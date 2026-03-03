import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

from regime_modeling.config import (
    N_REGIMES,
    DECISION_THRESHOLDS,
    OUTPUT_DIR
)
from regime_modeling.features import get_enhanced_features_for_model
from regime_modeling.models.regime import analyze_regime_characteristics

logger = logging.getLogger(__name__)


def _split_time_series_data(data: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split time-series data chronologically into train/test sets."""
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data, split_idx


def calculate_model_metrics(model: GaussianHMM, data_scaled: np.ndarray, n_features: int) -> Dict[str, Any]:
    """Calculate performance metrics for HMM model."""
    n_samples = len(data_scaled)
    n_states = model.n_components

    log_likelihood = model.score(data_scaled)
    n_params_means = n_states * n_features
    # Covariance params formula for 'full'. For 'diag'/'tied' it's different.
    n_params_covars = n_states * n_features * (n_features + 1) // 2
    n_params_transitions = n_states * (n_states - 1)
    n_params_init = n_states - 1   # startprob_ has n_states - 1 free params (sum to 1)
    n_params = n_params_means + n_params_covars + n_params_transitions + n_params_init

    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    return {
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'n_params': n_params,
        'n_features': n_features,
        'n_samples': n_samples
    }


def run_hmm_model(
    n_stocks: int,
    n_indices: int,
    volatility_window: int,
    rsi_period: int,
    momentum_period: int,
    include_returns: bool,
    include_volatility: bool,
    include_rsi: bool,
    include_momentum: bool,
    include_market_breadth: bool,
    n_iter: int,
    covariance_type: str,
    random_state: int,
    backtest: bool,
    train_ratio: float,
    generate_outputs: bool = True
) -> Dict[str, Any]:
    """
    Run HMM model with optional backtesting.
    
    Returns:
        Dictionary with backtest results and model metrics
    """
    mode_str = "BACKTEST" if backtest else "TRAIN"
    logger.info("\n" + "="*60)
    logger.info(f"RUNNING HMM MODEL - {mode_str} MODE")
    logger.info("="*60)

    logger.info("\nStep 1: Loading and engineering features...")
    data = get_enhanced_features_for_model(
        join_type='inner',
        n_stocks=n_stocks,
        n_indices=n_indices,
        volatility_window=volatility_window,
        rsi_period=rsi_period,
        momentum_period=momentum_period,
        save_to_csv=not backtest,
        csv_filename='merged_data_with_features.csv',
        include_returns=include_returns,
        include_volatility=include_volatility,
        include_rsi=include_rsi,
        include_momentum=include_momentum,
        include_market_breadth=include_market_breadth
    )

    logger.info(f"Feature matrix shape: {data.shape}")

    train_data, test_data, split_idx = _split_time_series_data(data, train_ratio)
    logger.info(f"Train: {len(train_data)} samples | Test: {len(test_data)} samples")
    logger.info(f"Split date: {data.index[split_idx]}")
    training_data = train_data

    logger.info("\nStep 2: Scaling features...")
    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(training_data)
    logger.info(f"Scaled data shape: {obs_scaled.shape}")

    n_states = N_REGIMES

    logger.info(f"\nStep 3: Training HMM with {n_states} states...")

    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=random_state)
    model.fit(obs_scaled)

    logger.info("Model trained!")

    logger.info("\nStep 4: Evaluating on test set...")

    train_log_prob = model.score(obs_scaled)

    test_scaled = scaler.transform(test_data)
    test_log_prob = model.score(test_scaled)

    _, test_states = model.decode(test_scaled, algorithm='viterbi')
    _, train_states = model.decode(obs_scaled, algorithm='viterbi')
    
    # Use robust method to calculate parameters instead of calculating manually based on full covars
    metrics = calculate_model_metrics(model, test_scaled, test_data.shape[1])
    test_aic = metrics['aic']
    test_bic = metrics['bic']

    train_log_prob_avg = train_log_prob / len(obs_scaled)
    test_log_prob_avg = test_log_prob / len(test_scaled)
    degradation = ((test_log_prob_avg - train_log_prob_avg) / train_log_prob_avg) * 100

    regime_changes = np.sum(np.diff(test_states) != 0)
    avg_regime_duration = len(test_states) / (regime_changes + 1)

    regime_types = analyze_regime_characteristics(test_states, test_data, n_states)
    unique_regimes = len(set(regime_types))
    n_unique_regimes = unique_regimes
    regime_diversity = unique_regimes / n_states  # Normalize by total possible states

    # Print concise results
    logger.info(f"Train LogL: {train_log_prob_avg:.2f}/sample ({train_log_prob:.0f} total)")
    logger.info(f"Test LogL: {test_log_prob_avg:.2f}/sample ({test_log_prob:.0f} total)")
    logger.info(f"Degradation: {degradation:.1f}%")
    logger.info(f"Test AIC: {test_aic:.0f} | Test BIC: {test_bic:.0f}")
    logger.info(f"Avg Regime Duration: {avg_regime_duration:.1f} periods | Changes: {regime_changes}")
    logger.info(f"Regimes: {' | '.join([f'{i}={regime_types[i]}' for i in range(n_states)])}")

    if degradation < DECISION_THRESHOLDS['ready_degradation'] and DECISION_THRESHOLDS['ready_min_duration'] <= avg_regime_duration <= DECISION_THRESHOLDS['ready_max_duration']:
        decision = "READY FOR PORTFOLIO"
    elif degradation < DECISION_THRESHOLDS['caution_degradation']:
        decision = "USE WITH CAUTION"
    else:
        decision = "NOT RECOMMENDED"
    logger.info(f"\nDecision: {decision}")

    logger.info("\n" + "="*60)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*60 + "\n")

    results = {
        'model': model,
        'scaler': scaler,
        'train_log_prob': train_log_prob,
        'test_log_prob': test_log_prob,
        'test_aic': test_aic,
        'test_bic': test_bic,
        'degradation': degradation,
        'n_regime_changes': regime_changes,
        'avg_regime_duration': avg_regime_duration,
        'regime_types': regime_types,
        'regime_diversity': regime_diversity,
        'n_unique_regimes': n_unique_regimes,
        'test_states': test_states,
        'test_data': test_data,
        'train_start': training_data.index[0].strftime('%Y-%m-%d'),
        'train_end': training_data.index[-1].strftime('%Y-%m-%d'),
        'test_start': test_data.index[0].strftime('%Y-%m-%d'),
        'test_end': test_data.index[-1].strftime('%Y-%m-%d'),
        'decision': decision
    }

    if generate_outputs:
        logger.info("\nGenerating dashboard outputs...")
        # Note: Delayed import to avoid circular dependency
        from regime_modeling.analysis.regime_stats import generate_all_outputs
        
        generate_all_outputs(
            model, 
            test_data, 
            test_states, 
            scaler, 
            regime_types,
            output_dir=OUTPUT_DIR
        )

    return results
