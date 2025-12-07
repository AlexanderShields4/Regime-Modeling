import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import norm
from features import get_enhanced_features_for_model
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import multiprocessing as mp
from itertools import product
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def _split_time_series_data(data, train_ratio):
    """Split time-series data chronologically into train/test sets."""
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data, split_idx


def _extract_return_columns(data):
    """
    Extract return columns from feature data.
    Returns raw price/return columns (stock_, index_, resource_), excluding engineered features.

    Returns:
        List of column names that represent returns
    """
    # Include all asset type prefixes (not just stock_)
    valid_prefixes = ('stock_', 'index_', 'resource_')

    # Exclude engineered feature suffixes
    exclude_patterns = ['_ma_', '_vol', 'momentum', '_mom', 'rsi', '_rsi',
                       '_ema', '_sma', '_ewma']

    return_cols = [
        col for col in data.columns
        if col.lower().startswith(valid_prefixes) and not any(
            pattern in col.lower() for pattern in exclude_patterns
        )
    ]

    return return_cols


def _analyze_regime_characteristics(states, data, n_states):
    """Analyze economic characteristics of each regime with 5-level classification."""
    
    return_cols = _extract_return_columns(data)

    regime_types = []
    if not return_cols:
        # No return columns found - cannot classify regimes
        import warnings
        warnings.warn(
            "⚠ No return columns found for regime classification. "
            "Possible causes:\n"
            "  • include_returns=False in configuration\n"
            "  • n_stocks=0, n_indices=0, and no resource data\n"
            "  • All return columns filtered by feature engineering\n"
            f"Available columns: {list(data.columns)[:10]}...\n"
            "All regimes will be labeled 'Unknown' with conservative allocation (60/30/10).",
            UserWarning,
            stacklevel=2
        )
        return ["Unknown"] * n_states

    # First pass: collect all regime statistics
    regime_stats = []
    for regime in range(n_states):
        mask = states == regime
        subset = data[mask]

        if len(subset) == 0:
            regime_stats.append({'mean_return': 0, 'std_return': 1, 'sharpe': 0, 'count': 0})
            continue

        regime_returns = subset[return_cols].mean(axis=1)
        mean_ret = regime_returns.mean()
        std_ret = regime_returns.std()
        sharpe = mean_ret / std_ret if std_ret > 0 else 0

        regime_stats.append({
            'mean_return': mean_ret,
            'std_return': std_ret,
            'sharpe': sharpe,
            'count': len(subset)
        })



    # Use relative ranking instead of absolute thresholds
    # Sort by mean returns to classify Bull (highest) vs Bear (lowest) vs Sideways (middle)
    valid_regimes = [(i, stats) for i, stats in enumerate(regime_stats) if stats['count'] > 0]

    if len(valid_regimes) == 0:
        return ["Unknown"] * n_states

    # Sort by mean return (descending) - highest return = Bull, lowest = Bear
    sorted_regimes = sorted(valid_regimes, key=lambda x: x[1]['mean_return'], reverse=True)

    # Assign labels based on relative performance
    regime_labels = {}
    if len(sorted_regimes) == 3:
        # Highest return = Bull, Lowest = Bear, Middle = Sideways
        regime_labels[sorted_regimes[0][0]] = "Bull"
        regime_labels[sorted_regimes[1][0]] = "Sideways"
        regime_labels[sorted_regimes[2][0]] = "Bear"
    elif len(sorted_regimes) == 2:
        # Highest = Bull, Lowest = Bear
        regime_labels[sorted_regimes[0][0]] = "Bull"
        regime_labels[sorted_regimes[1][0]] = "Bear"
    else:
        # Only 1 regime
        regime_labels[sorted_regimes[0][0]] = "Sideways"

    # Build final list in original order
    for regime in range(n_states):
        if regime in regime_labels:
            regime_types.append(regime_labels[regime])
        else:
            regime_types.append("Unknown")

    return regime_types

def _generate_all_outputs(model, test_data, test_states, scaler, regime_types, 
                          output_dir='dashboard_outputs'):
    """
    Generate dashboard outputs, risk metrics, and portfolio backtest in one unified place.
    
    Consolidates: dashboard generation, risk metrics calculation, and portfolio backtest.
    
    Args:
        model: Trained HMM model
        test_data: Test data
        test_states: Predicted states
        scaler: Fitted scaler
        regime_types: Regime labels (Bull/Bear/Sideways)
        output_dir: Output directory
    
    Returns:
        Backtest results dictionary (or None if backtest fails)
    """
    # Generate dashboard - pass empty dict for backtest_results if not available
    dashboard_info = generate_dashboard_outputs(
        model, test_data, test_states, {},
        regime_types=regime_types, 
        output_dir=output_dir
    )
    print(f"Dashboard files saved to: {dashboard_info['output_dir']}/")
    
    # Calculate and save risk metrics
    risk_metrics = calculate_risk_metrics(test_data, test_states, regime_types=regime_types)
    if risk_metrics:
        import json
        with open(f"{dashboard_info['output_dir']}/risk_metrics.json", 'w') as f:
            json.dump(risk_metrics, f, indent=2)
        print(f"Risk metrics saved")
    
    # Run portfolio backtest
    print("\nRunning portfolio backtest on test data...")
    try:
        backtest_results = run_portfolio_backtest(test_states, test_data)
        print(f"Portfolio backtest complete!")
        print(f"Best rebalancing frequency: {backtest_results['best_rebalance_freq']}")
        print(f"Visualizations saved to: {backtest_results['output_dir']}/charts/\n")
        return backtest_results
    except Exception as e:
        print(f"Portfolio backtest failed: {e}")
        print("Continuing without backtest results...\n")
        return None


def run_hmm_model(
    n_stocks,
    n_indices,
    volatility_window,
    rsi_period,
    momentum_period,
    include_returns,
    include_volatility,
    include_rsi,
    include_momentum,
    include_market_breadth,
    n_iter,
    covariance_type,
    random_state,
    backtest,
    train_ratio,
    generate_outputs=True):
    """
    Run HMM model with optional backtesting.

    Args:
        n_stocks: Number of stocks to include
        n_indices: Number of indices to include
        volatility_window: Window for volatility calculation
        rsi_period: Period for RSI calculation
        momentum_period: Period for momentum calculation
        include_returns: Include log returns features
        include_volatility: Include rolling volatility features
        include_rsi: Include RSI technical indicators
        include_momentum: Include momentum indicators
        include_market_breadth: Include market breadth indicators
        n_iter: Number of iterations for HMM training (recommended: 3000-7000)
        covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
        random_state: Random state for reproducibility (ensures consistent models)
        backtest: If True, perform train/test split backtesting
        train_ratio: Ratio of data for training (default 0.8)
        generate_outputs: If True, generate dashboard visualizations
        
    Note: HMM always uses n_components=3 (Bull, Bear, Sideways markets)

    Returns:
        If backtest=False: Tuple of (model, hidden_states, log_probability, features_df, scaler)
        If backtest=True: Dictionary with backtest_results and model metrics
    """
    mode_str = "BACKTEST" if backtest else "TRAIN"
    print("\n" + "="*60)
    print(f"RUNNING HMM MODEL - {mode_str} MODE")
    print("="*60)

    # Get enhanced features with customizable feature types
    print("\nStep 1: Loading and engineering features...")
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

    print(f"Feature matrix shape: {data.shape}")

    # Standard single split backtesting

    train_data, test_data, split_idx = _split_time_series_data(data, train_ratio)
    print(f"Train: {len(train_data)} samples | Test: {len(test_data)} samples")
    print(f"Split date: {data.index[split_idx]}")
    training_data = train_data

    # Scale the features
    print("\nStep 2: Scaling features...")
    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(training_data)
    print(f"Scaled data shape: {obs_scaled.shape}")

    # Define market states
    states = ["Bull Market", "Bear Market", "Sideways Market"]
    n_states = len(states)

    print(f"\nStep 3: Training HMM with {n_states} states...")

    # Train HMM model
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=random_state)
    model.fit(obs_scaled)

    print("Model trained!")

    # Backtesting mode - evaluate on test set
    print("\nStep 4: Evaluating on test set...")

    # Get training performance
    train_log_prob = model.score(obs_scaled)

    # Evaluate on test set
    test_scaled = scaler.transform(test_data)
    test_log_prob = model.score(test_scaled)

    # Use Viterbi algorithm for optimal state sequence (same as non-backtest mode)
    _, test_states = model.decode(test_scaled, algorithm='viterbi')

    # Decode training states using Viterbi for consistency
    _, train_states = model.decode(obs_scaled, algorithm='viterbi')

    # Calculate metrics
    n_features = test_data.shape[1]
    n_samples = len(test_data)
    n_params_means = n_states * n_features
    n_params_covars = n_states * n_features * (n_features + 1) // 2
    n_params_transitions = n_states * (n_states - 1)
    n_params = n_params_means + n_params_covars + n_params_transitions

    test_aic = -2 * test_log_prob + 2 * n_params
    test_bic = -2 * test_log_prob + n_params * np.log(n_samples)

    # Calculate degradation 
    train_log_prob_avg = train_log_prob / len(obs_scaled)
    test_log_prob_avg = test_log_prob / len(test_scaled)
    degradation = ((test_log_prob_avg - train_log_prob_avg) / train_log_prob_avg) * 100

    # Regime stability
    regime_changes = np.sum(np.diff(test_states) != 0)
    avg_regime_duration = len(test_states) / (regime_changes + 1)

    # Analyze regime characteristics (use test_data, not train_data - states match test data length)
    regime_types = _analyze_regime_characteristics(test_states, test_data, n_states)

    # Calculate regime diversity metrics
    unique_regimes = len(set(regime_types))
    n_unique_regimes = unique_regimes
    regime_diversity = unique_regimes / n_states  # Normalize by total possible states

    # Print concise results
    print(f"Train LogL: {train_log_prob_avg:.2f}/sample ({train_log_prob:.0f} total)")
    print(f"Test LogL: {test_log_prob_avg:.2f}/sample ({test_log_prob:.0f} total)")
    print(f"Degradation: {degradation:.1f}%")
    print(f"Test AIC: {test_aic:.0f} | Test BIC: {test_bic:.0f}")
    print(f"Avg Regime Duration: {avg_regime_duration:.1f} periods | Changes: {regime_changes}")
    print(f"Regimes: {' | '.join([f'{i}={regime_types[i]}' for i in range(n_states)])}")

    # Decision
    if degradation < 10 and 5 <= avg_regime_duration <= 50:
        decision = "READY FOR PORTFOLIO"
    elif degradation < 30:
        decision = "USE WITH CAUTION"
    else:
        decision = "NOT RECOMMENDED"
    print(f"\nDecision: {decision}")

    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60 + "\n")

    # Prepare results dictionary
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

    # Only generate outputs if requested (skip during grid search for performance)
    if generate_outputs:
        print("\nGenerating dashboard outputs...")
        _generate_all_outputs(model, test_data, test_states, scaler, regime_types, 
                              output_dir='dashboard_outputs')

    return results


def calculate_model_metrics(model, data_scaled, n_features):
    """
    Calculate performance metrics for HMM model.

    Args:
        model: Trained GaussianHMM model
        data_scaled: Scaled feature data
        n_features: Number of features used

    Returns:
        Dictionary of metrics (AIC, BIC, log_likelihood)
    """
    n_samples = len(data_scaled)
    n_states = model.n_components

    # Calculate log likelihood
    log_likelihood = model.score(data_scaled)

    # Calculate number of parameters
    n_params_means = n_states * n_features
    n_params_covars = n_states * n_features * (n_features + 1) // 2
    n_params_transitions = n_states * (n_states - 1)
    n_params = n_params_means + n_params_covars + n_params_transitions

    # Calculate AIC and BIC
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

def save_best_model(model, scaler, params, metrics, filename_prefix='best_hmm_model'):
    """
    Save the best HMM model, scaler, and configuration to disk.

    Args:
        model: Trained GaussianHMM model
        scaler: Fitted StandardScaler
        params: Dictionary of model parameters
        metrics: Dictionary of performance metrics
        filename_prefix: Prefix for saved files

    Returns:
        Dictionary with paths to saved files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Ensure all critical HMM parameters are in params dict
    model_params = {
        'n_components': 3,  # Fixed to 3 states
        'covariance_type': model.covariance_type,
        'n_iter': model.n_iter,
        'random_state': params.get('random_state', 42)
    }
    
    # Merge with existing params
    complete_params = {**params, **model_params}
    
    # Save model
    model_path = f'models/{filename_prefix}_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'params': complete_params,
            'metrics': metrics,
            'timestamp': timestamp
        }, f)

    # Also save as 'latest' for easy access
    latest_path = f'models/{filename_prefix}_latest.pkl'
    with open(latest_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'params': complete_params,
            'metrics': metrics,
            'timestamp': timestamp
        }, f)

    return {
        'model_path': model_path,
        'latest_path': latest_path
    }

def _calculate_composite_score(results):
    """
    Calculate composite score balancing portfolio performance and statistical quality.
    Lower score = better configuration.

    Args:
        results: Dictionary with HMM backtest results and portfolio metrics

    Returns:
        float: Composite score (lower is better)
    """
    # ========================================
    # PORTFOLIO PERFORMANCE METRICS
    # ========================================

    # 1. Returns penalty (invert CAGR - high CAGR = low penalty)
    cagr = results.get('portfolio_cagr', 0)
    returns_penalty = max(0, (0.40 - cagr) * 250)  # Weight: 0-150

    # 2. Sharpe ratio penalty (invert - high Sharpe = low penalty)
    sharpe = results.get('portfolio_sharpe', 0)
    sharpe_penalty = max(0, (3.0 - sharpe) * 25)  # Weight: 0-100

    # 3. Max drawdown penalty (absolute value - high DD = high penalty)
    max_dd = abs(results.get('portfolio_max_dd', -0.5))
    drawdown_penalty = max_dd * 150  # Weight: 7.5-90

    # 4. Sortino ratio penalty (invert - high Sortino = low penalty)
    sortino = results.get('portfolio_sortino', 0)
    sortino_penalty = max(0, (4.0 - sortino) * 15)  # Weight: 0-75

    # 5. Calmar ratio penalty (invert - high Calmar = low penalty)
    calmar = results.get('portfolio_calmar', 0)
    calmar_penalty = max(0, (3.0 - calmar) * 20)  # Weight: 0-100

    # Portfolio Performance Score (sum of penalties)
    portfolio_score = (
        returns_penalty +      # 0-150
        sharpe_penalty +       # 0-100
        drawdown_penalty +     # 7.5-90
        sortino_penalty +      # 0-75
        calmar_penalty         # 0-100
    )

    # ========================================
    # STATISTICAL QUALITY METRICS (40% weight)
    # ========================================

    test_states = results['test_states']
    n_regime_changes = results['n_regime_changes']
    avg_duration = results['avg_regime_duration']

    # Calculate regime diversity
    state_counts = np.bincount(test_states)
    state_probs = state_counts / len(test_states)
    regime_entropy = -np.sum(state_probs[state_probs > 0] * np.log(state_probs[state_probs > 0]))
    regime_diversity_score = regime_entropy / np.log(3)

    n_unique_regimes = len(np.unique(test_states))

    # Model quality penalties
    regime_usage_penalty = (3 - n_unique_regimes) * 30 if n_unique_regimes < 3 else 0

    switching_penalty = 0
    if n_regime_changes < 3:
        switching_penalty = (3 - n_regime_changes) * 20
    elif n_regime_changes > len(test_states) / 3:
        switching_penalty = (n_regime_changes - len(test_states) / 3) * 0.5

    duration_penalty = 0
    if avg_duration < 5:
        duration_penalty = (5 - avg_duration) * 25
    elif avg_duration > 50:
        duration_penalty = (avg_duration - 50) * 3

    degradation_penalty = abs(results['degradation']) * 8
    bic_score = results['test_bic'] / 1000
    diversity_penalty = (1 - regime_diversity_score) * 50

    # Statistical Quality Score
    statistical_score = (
        degradation_penalty +      # 8-16
        regime_usage_penalty +     # 0-60
        switching_penalty +        # 0-100
        duration_penalty +         # 0-135
        bic_score +                # 5-15
        diversity_penalty          # 0-50
    )

    # ========================================
    # COMPOSITE SCORE (60/40 split)
    # ========================================

    # Normalize both components to similar scales
    portfolio_normalized = portfolio_score / 250  # Normalize to ~1.0 average
    statistical_normalized = statistical_score / 150  # Normalize to ~1.0 average

    composite_score = (
        portfolio_normalized * 600 +     # 60% weight on portfolio
        statistical_normalized * 400     # 40% weight on statistics
    )

    # Store component scores for analysis
    results['portfolio_score_component'] = portfolio_score
    results['statistical_score_component'] = statistical_score
    results['regime_diversity'] = regime_diversity_score
    results['n_unique_regimes'] = n_unique_regimes

    return composite_score


def _run_single_config(args):
    """
    Helper function to run a single configuration for parallel processing.
    Returns results dict or None if error.
    """
    config_id, params = args

    try:
        # HARDCODED: Grid search ALWAYS uses backtest=True and requires stocks
        # Validate that stocks are included (n_stocks must be > 0)
        if params['n_stocks'] <= 0:
            raise ValueError(f"n_stocks must be > 0 for grid search (got {params['n_stocks']})")

        # Run backtest with these parameters
        # HARDCODED: backtest=True is mandatory for grid search evaluation
        results = run_hmm_model(
            n_stocks=params['n_stocks'],
            n_indices=params['n_indices'],
            volatility_window=params['volatility_window'],
            rsi_period=params['rsi_period'],
            momentum_period=params['momentum_period'],
            include_returns=params['include_returns'],
            include_volatility=params['include_volatility'],
            include_rsi=params['include_rsi'],
            include_momentum=params['include_momentum'],
            include_market_breadth=params['include_market_breadth'],
            backtest=True,  # HARDCODED: Always use backtest mode in grid search
            train_ratio=params.get('train_ratio', 0.8),
            n_iter=params['n_iter'],
            covariance_type=params['covariance_type'],
            random_state=params['random_state'],
            generate_outputs=False  # Skip dashboard generation during grid search
        )

        # Add config info to results
        results['config_id'] = config_id
        results['params'] = params
        # NEW: Run portfolio backtest to get investment performance metrics
        try:
            test_states = results['test_states']
            test_data = results['test_data']

            # Skip visualization generation for grid search (much faster)
            portfolio_results = run_portfolio_backtest(test_states, test_data, generate_visualizations=False)

            # Extract portfolio metrics for best rebalance frequency
            best_freq = portfolio_results['best_rebalance_freq']
            best_strategy = f'Regime-Based ({best_freq})'
            portfolio_metrics = portfolio_results['metrics'][best_strategy]

            # Add portfolio metrics to results
            results['portfolio_cagr'] = portfolio_metrics['CAGR']
            results['portfolio_sharpe'] = portfolio_metrics['Sharpe Ratio']
            results['portfolio_sortino'] = portfolio_metrics['Sortino Ratio']
            results['portfolio_max_dd'] = portfolio_metrics['Max Drawdown']
            results['portfolio_volatility'] = portfolio_metrics['Volatility']
            results['portfolio_calmar'] = portfolio_metrics['Calmar Ratio']
            results['portfolio_win_rate'] = portfolio_metrics['Win Rate']
            results['best_rebalance_freq'] = best_freq

        except Exception as e:
            # If portfolio backtest fails, use default poor values
            print(f"  Config {config_id}: Portfolio backtest failed ({str(e)}), using defaults")
            results['portfolio_cagr'] = 0.0
            results['portfolio_sharpe'] = 0.0
            results['portfolio_sortino'] = 0.0
            results['portfolio_max_dd'] = -0.5
            results['portfolio_volatility'] = 0.3
            results['portfolio_calmar'] = 0.0
            results['portfolio_win_rate'] = 0.5
            results['best_rebalance_freq'] = 'unknown'

        # Calculate new composite score (balances portfolio + statistical quality)
        results['score'] = _calculate_composite_score(results)

        return results

    except Exception as e:
        print(f"Config {config_id} failed: {str(e)}")
        return None


def grid_search_parameters(
    n_stocks_range=[5, 7, 10, 15],
    n_indices_range=[0, 3, 5],
    volatility_window_range=[10, 20, 30],
    rsi_period_range=[10, 14, 20],
    momentum_period_range=[5, 10, 15],
    n_iter_range=[5000],
    covariance_type_range=['full'],
    random_state=42,
    feature_combinations='auto',
    train_ratio=0.8,
    n_processes=None,
    top_n=10,
    show_progress=False
):
    """
    Perform parallel grid search over HMM parameters to find optimal configuration.

    Args:
        n_stocks_range: List of n_stocks values to test
        n_indices_range: List of n_indices values to test
        volatility_window_range: List of volatility window values to test
        rsi_period_range: List of RSI period values to test
        momentum_period_range: List of momentum period values to test
        n_iter_range: List of HMM iteration counts to test
        covariance_type_range: List of covariance types to test ('full', 'tied', 'diag', 'spherical')
        random_state: Random state for reproducibility
        feature_combinations: 'auto' for smart combinations, or list of dicts
        train_ratio: Train/test split ratio
        n_processes: Number of parallel processes (None = auto-detect)
        top_n: Number of top results to display

    Returns:
        DataFrame with all results sorted by score
    """

    print("\n" + "="*70)
    print("GRID SEARCH - PARALLEL PARAMETER OPTIMIZATION")
    print("="*70)

    # HARDCODED VALIDATION: Ensure stocks are always included in grid search
    if any(n <= 0 for n in n_stocks_range):
        raise ValueError(f"Grid search requires stocks: n_stocks_range must not contain values <= 0. Got: {n_stocks_range}")

    print("\n✓ Validated: Backtesting enabled and stocks required for all configurations")

    # Define feature combinations to test
    if feature_combinations == 'auto':
        # Smart subset - most promising combinations
        feature_combos = [
            # Core combinations
            {'returns': True, 'volatility': True, 'rsi': False, 'momentum': False, 'market_breadth': False},
            {'returns': True, 'volatility': True, 'rsi': True, 'momentum': True, 'market_breadth': False},
            {'returns': True, 'volatility': True, 'rsi': False, 'momentum': True, 'market_breadth': False},
            {'returns': True, 'volatility': True, 'rsi': True, 'momentum': False, 'market_breadth': False},
            # With market breadth
            {'returns': True, 'volatility': True, 'rsi': False, 'momentum': False, 'market_breadth': True},
            {'returns': True, 'volatility': True, 'rsi': True, 'momentum': True, 'market_breadth': True},
        ]
    else:
        feature_combos = feature_combinations

    # Generate all parameter combinations
    param_grid = []
    config_id = 0

    for n_stocks, n_indices, vol_window, rsi_period, mom_period, n_iter, cov_type, features in product(
        n_stocks_range,
        n_indices_range,
        volatility_window_range,
        rsi_period_range,
        momentum_period_range,
        n_iter_range,
        covariance_type_range,
        feature_combos
    ):
        params = {
            'n_stocks': n_stocks,
            'n_indices': n_indices,
            'volatility_window': vol_window,
            'rsi_period': rsi_period,
            'momentum_period': mom_period,
            'n_iter': n_iter,
            'covariance_type': cov_type,
            'random_state': random_state,
            'include_returns': features['returns'],
            'include_volatility': features['volatility'],
            'include_rsi': features['rsi'],
            'include_momentum': features['momentum'],
            'include_market_breadth': features['market_breadth'],
            'train_ratio': train_ratio,
            'n_components': 3  # Fixed to 3 states for consistency
        }
        param_grid.append((config_id, params))
        config_id += 1

    total_configs = len(param_grid)
    print(f"\nTotal configurations to test: {total_configs}")
    print(f"Parameter ranges:")
    print(f"  n_stocks: {n_stocks_range}")
    print(f"  n_indices: {n_indices_range}")
    print(f"  volatility_window: {volatility_window_range}")
    print(f"  rsi_period: {rsi_period_range}")
    print(f"  momentum_period: {momentum_period_range}")
    print(f"  n_iter: {n_iter_range}")
    print(f"  covariance_type: {covariance_type_range}")
    print(f"  Feature combinations: {len(feature_combos)}")

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() // 2)  # Use half the cores

    estimated_minutes_low = total_configs / n_processes * 0.5
    estimated_minutes_high = total_configs / n_processes * 2
    expected_completion = datetime.now() + timedelta(minutes=estimated_minutes_high)

    print(f"\nUsing {n_processes} parallel processes (half of available cores)")
    print(f"Estimated time: {estimated_minutes_low:.1f}-{estimated_minutes_high:.1f} minutes ({estimated_minutes_low/60:.1f}-{estimated_minutes_high/60:.1f} hours)")
    print(f"Expected completion: {expected_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nStarting grid search...\n")

    # Run parallel grid search
    results_list = []

    if n_processes > 1:
        # Parallel execution
        with mp.Pool(processes=n_processes) as pool:
            # Use imap_unordered for progress tracking
            iterator = pool.imap_unordered(_run_single_config, param_grid)
            
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, total=total_configs, desc="Grid Search")
                except ImportError:
                    pass
            
            for result in iterator:
                if result is not None:
                    results_list.append(result)
    else:
        # Serial execution (for debugging)
        iterator = param_grid
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Grid Search")
            except ImportError:
                pass
        
        for config in iterator:
            result = _run_single_config(config)
            if result is not None:
                results_list.append(result)

    print(f"\nCompleted! {len(results_list)}/{total_configs} configurations succeeded")

    # Check if any configurations succeeded
    if len(results_list) == 0:
        print("\n" + "="*70)
        print("ERROR: All configurations failed!")
        print("="*70)
        print("\nPlease check:")
        print("  1. Data quality and completeness")
        print("  2. Feature engineering parameters")
        print("  3. Error messages above for specific failures")
        print("="*70 + "\n")
        return None

    # Convert to DataFrame
    summary_data = []
    for r in results_list:
        p = r['params']

        # Create feature string
        features = []
        if p['include_returns']: features.append('Ret')
        if p['include_volatility']: features.append('Vol')
        if p['include_rsi']: features.append('RSI')
        if p['include_momentum']: features.append('Mom')
        if p['include_market_breadth']: features.append('MB')
        feature_str = '+'.join(features)

        # Calculate decision based on portfolio metrics
        sharpe = r.get('portfolio_sharpe', 0)
        max_dd = r.get('portfolio_max_dd', -0.5)
        cagr = r.get('portfolio_cagr', 0)
        degradation = abs(r['degradation'])

        if sharpe > 1.5 and max_dd > -0.25 and cagr > 0.12 and degradation < 10:
            decision = '✅ EXCELLENT'
        elif sharpe > 1.0 and max_dd > -0.30 and cagr > 0.08 and degradation < 15:
            decision = '✓ GOOD'
        elif sharpe > 0.5 and max_dd > -0.40 and cagr > 0.04:
            decision = '○ ACCEPTABLE'
        else:
            decision = '✗ POOR'

        summary_data.append({
            'Config_ID': r['config_id'],
            'Score': r['score'],
            # Portfolio Performance Metrics (NEW)
            'CAGR%': r.get('portfolio_cagr', 0) * 100,
            'Sharpe': r.get('portfolio_sharpe', 0),
            'Sortino': r.get('portfolio_sortino', 0),
            'MaxDD%': r.get('portfolio_max_dd', -0.5) * 100,
            'Calmar': r.get('portfolio_calmar', 0),
            'Volatility%': r.get('portfolio_volatility', 0.3) * 100,
            'WinRate%': r.get('portfolio_win_rate', 0.5) * 100,
            'BestRebalance': r.get('best_rebalance_freq', 'unknown'),
            # Statistical Quality Metrics
            'Regime_Diversity': r.get('regime_diversity', 0),
            'N_Regimes_Used': r.get('n_unique_regimes', 0),
            'Regime_Changes': r['n_regime_changes'],
            'Avg_Duration': r['avg_regime_duration'],
            'Degradation%': r['degradation'],
            'Test_LogL': r['test_log_prob'],
            'Test_BIC': r['test_bic'],
            # Configuration Parameters
            'n_stocks': p['n_stocks'],
            'n_indices': p['n_indices'],
            'vol_window': p['volatility_window'],
            'Features': feature_str,
            'Regimes': '|'.join(r['regime_types']),
            'Decision': decision
        })

    results_df = pd.DataFrame(summary_data)
    results_df = results_df.sort_values('Score')  # Lower score is better

    # Display top results
    print("\n" + "="*70)
    print(f"TOP {top_n} CONFIGURATIONS (Lower Score = Better)")
    print("="*70)
    print(results_df.head(top_n).to_string(index=False))

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'grid_search_results_{timestamp}.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Full results saved to: {csv_filename}")

    # Display best configuration details
    best = results_df.iloc[0]
    print("\n" + "="*70)
    print("BEST CONFIGURATION DETAILS")
    print("="*70)
    print(f"Config ID: {best['Config_ID']}")
    print(f"Composite Score: {best['Score']:.2f} (lower is better)")
    print(f"\nPORTFOLIO PERFORMANCE:")
    print(f"  CAGR: {best['CAGR%']:.2f}%")
    print(f"  Sharpe Ratio: {best['Sharpe']:.2f}")
    print(f"  Sortino Ratio: {best['Sortino']:.2f}")
    print(f"  Max Drawdown: {best['MaxDD%']:.2f}%")
    print(f"  Calmar Ratio: {best['Calmar']:.2f}")
    print(f"  Volatility: {best['Volatility%']:.2f}%")
    print(f"  Win Rate: {best['WinRate%']:.1f}%")
    print(f"  Best Rebalance Freq: {best['BestRebalance']}")
    print(f"\nPARAMETERS:")
    print(f"  n_stocks: {best['n_stocks']}")
    print(f"  n_indices: {best['n_indices']}")
    print(f"  volatility_window: {best['vol_window']}")
    print(f"  Features: {best['Features']}")
    print(f"\nREGIME QUALITY:")
    print(f"  Regimes Used: {best['N_Regimes_Used']}/3")
    print(f"  Regime Diversity: {best['Regime_Diversity']:.3f} (1.0 = perfectly balanced)")
    print(f"  Regime Changes: {best['Regime_Changes']}")
    print(f"  Avg Regime Duration: {best['Avg_Duration']:.1f} periods")
    print(f"  Regime Types: {best['Regimes']}")
    print(f"\nSTATISTICAL QUALITY:")
    print(f"  Degradation: {best['Degradation%']:.1f}%")
    print(f"  Test Log-Likelihood: {best['Test_LogL']:.0f}")
    print(f"  Test BIC: {best['Test_BIC']:.0f}")
    print(f"\nOVERALL DECISION: {best['Decision']}")
    print("="*70 + "\n")

    # Save best config to file for easy reference
    best_config_file = 'best_config.txt'
    best_params = [r for r in results_list if r['config_id'] == best['Config_ID']][0]['params']

    with open(best_config_file, 'w') as f:
        f.write("# Best HMM Configuration (from grid search)\n")
        f.write(f"# Generated: {timestamp}\n\n")
        f.write("run_hmm_model(\n")
        f.write(f"    n_stocks={best_params['n_stocks']},\n")
        f.write(f"    n_indices={best_params['n_indices']},\n")
        f.write(f"    volatility_window={best_params['volatility_window']},\n")
        f.write(f"    rsi_period={best_params['rsi_period']},\n")
        f.write(f"    momentum_period={best_params['momentum_period']},\n")
        f.write(f"    include_returns={best_params['include_returns']},\n")
        f.write(f"    include_volatility={best_params['include_volatility']},\n")
        f.write(f"    include_rsi={best_params['include_rsi']},\n")
        f.write(f"    include_momentum={best_params['include_momentum']},\n")
        f.write(f"    include_market_breadth={best_params['include_market_breadth']},\n")
        f.write(f"    n_iter={best_params['n_iter']},\n")
        f.write(f"    covariance_type='{best_params['covariance_type']}',\n")
        f.write(f"    random_state={best_params['random_state']},\n")
        f.write(f"    backtest=True\n")
        f.write(")\n")

    print(f"✓ Best configuration saved to: {best_config_file}")

    # Save the best model (if available)
    best_result = [r for r in results_list if r['config_id'] == best['Config_ID']][0]

    if best_result.get('model') is not None and best_result.get('scaler') is not None:
        print("\nSaving best model...")

        # Prepare metrics dictionary
        best_metrics = {
            'score': best_result['score'],
            'degradation': best_result['degradation'],
            'train_log_prob': best_result['train_log_prob'],
            'test_log_prob': best_result['test_log_prob'],
            'test_aic': best_result['test_aic'],
            'test_bic': best_result['test_bic'],
            'avg_regime_duration': best_result['avg_regime_duration'],
            'n_regime_changes': best_result['n_regime_changes'],
            'regime_types': best_result['regime_types'],
            'decision': best_result['decision']
        }

        saved_paths = save_best_model(
            model=best_result['model'],
            scaler=best_result['scaler'],
            params=best_result['params'],
            metrics=best_metrics
        )

        print(f"✓ Best model saved to: {saved_paths['model_path']}")
        print(f"✓ Latest model link: {saved_paths['latest_path']}")
        print("\nTo load the model later:")
        print("  from model import load_model")
        print("  data = load_model()")
        print("  model = data['model']")
        print("  scaler = data['scaler']\n")
    else:
        print("    To train the best config, re-run with the saved parameters in best_config.txt\n")

    # Portfolio Performance Summary
    print("\n" + "="*70)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\nBest Sharpe Ratio: {results_df['Sharpe'].max():.2f} (Config {results_df.loc[results_df['Sharpe'].idxmax(), 'Config_ID']})")
    print(f"Best CAGR: {results_df['CAGR%'].max():.1f}% (Config {results_df.loc[results_df['CAGR%'].idxmax(), 'Config_ID']})")
    print(f"Best Max Drawdown: {results_df['MaxDD%'].max():.1f}% (Config {results_df.loc[results_df['MaxDD%'].idxmax(), 'Config_ID']})")
    print(f"Best Calmar Ratio: {results_df['Calmar'].max():.2f} (Config {results_df.loc[results_df['Calmar'].idxmax(), 'Config_ID']})")

    print(f"\n Configuration Quality Distribution:")
    decision_counts = results_df['Decision'].value_counts()
    for decision in ['✅ EXCELLENT', '✓ GOOD', '○ ACCEPTABLE', '✗ POOR']:
        count = decision_counts.get(decision, 0)
        pct = count / len(results_df) * 100
        print(f"  {decision}: {count}/{len(results_df)} ({pct:.1f}%)")

    excellent_count = (results_df['Decision'] == '✅ EXCELLENT').sum()
    good_count = (results_df['Decision'] == '✓ GOOD').sum()
    ready_count = excellent_count + good_count

    print(f"\nConfigs ready for live trading: {ready_count}/{len(results_df)} ({ready_count/len(results_df)*100:.1f}%)")
    print(f"  (Sharpe > 1.0, MaxDD > -30%, CAGR > 8%, Degradation < 15%)")

    print("\n" + "="*70)
    print("SCORING BREAKDOWN (for top config)")
    print("="*70)
    best_result = [r for r in results_list if r['config_id'] == best['Config_ID']][0]
    print(f"Composite Score: {best_result['score']:.2f}")
    print(f"  Portfolio Component: {best_result.get('portfolio_score_component', 0):.2f} (normalized × 600 = {best_result.get('portfolio_score_component', 0) / 250 * 600:.2f})")
    print(f"  Statistical Component: {best_result.get('statistical_score_component', 0):.2f} (normalized × 400 = {best_result.get('statistical_score_component', 0) / 150 * 400:.2f})")
    print(f"\nScore Interpretation:")
    print(f"  Lower score = better overall quality")
    print(f"  60% weight on portfolio performance (returns, risk, Sharpe)")
    print(f"  40% weight on statistical quality (BIC, degradation, regime diversity)")
    print("="*70 + "\n")

    return results_df


def generate_dashboard_outputs(model, data, hidden_states, backtest_results, regime_types, output_dir='dashboard_outputs'):
    import os
    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import plotly.graph_objects as go
    import plotly.express as px
    import shutil

    # Create output directory, removing old files if they exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Regime names & colors
    # -------------------------
    # ISSUE 6 FIX: Use actual regime types from analysis if provided
    if regime_types is not None:
        regime_names = regime_types
    else:
        # Fallback to analyzing if not provided
        n_states = len(np.unique(hidden_states))
        regime_names = _analyze_regime_characteristics(hidden_states, data, n_states)

    # color map keyed by regime name (rgba for vrect translucency)
    color_map = {
        'Bull': 'rgba(0,200,0,0.25)',      # bright green
        'Bear': 'rgba(220,0,0,0.25)',      # bright red
        'Sideways': 'rgba(120,120,120,0.25)'       # grey
    }

    # marker colors for legend (opaque)
    marker_map = {
        'Bull': 'green',
        'Bear': 'darkred',
        'Sideways': 'gray'
    }

    # -------------------------
    # Regime timeline (Plotly)
    # -------------------------
    fig_regime = go.Figure()

    # Aggregate contiguous regime spans
    if len(hidden_states) == 0:
        # Nothing to plot
        fig_regime.add_annotation(text="No regime data", xref="paper", yref="paper", showarrow=False)
    else:
        regime_changes = np.where(np.diff(hidden_states) != 0)[0]
        starts = np.insert(regime_changes + 1, 0, 0)
        ends = np.append(regime_changes, len(hidden_states) - 1)

        for s, e in zip(starts, ends):
            regime_idx = int(hidden_states[s])
            # guard: if index out of range, fallback to 'Sideways'
            if 0 <= regime_idx < len(regime_names):
                regime_name = regime_names[regime_idx]
            else:
                regime_name = 'Sideways'

            fill_color = color_map.get(regime_name, 'rgba(120,120,120,0.25)')

            fig_regime.add_vrect(
                x0=data.index[s], x1=data.index[e],
                fillcolor=fill_color, layer="below", line_width=0,
                annotation_text=regime_name if (e - s) <= 100 else None,  # optional label for short spans
                annotation_position="top left"
            )

    # Add invisible/dummy traces purely for legend (one per regime)
    for name in regime_names:
        fig_regime.add_trace(go.Scatter(
            x=[data.index[0] if len(data.index) else None],
            y=[0],
            mode='markers',
            marker=dict(size=10, color=marker_map.get(name, 'gray')),
            name=f"{name} Market",
            visible=True,
            hoverinfo='none'  # these are just legend entries
        ))

    fig_regime.update_layout(
        title='Market Regime Timeline',
        xaxis_title='Date',
        yaxis=dict(visible=False),  # timeline is background colored; hide numeric axis
        hovermode='x',
        height=400,
        showlegend=True,
        margin=dict(t=50, b=40, l=40, r=20)
    )

    # -------------------------
    # Regime statistics
    # -------------------------
    total_periods = len(hidden_states)
    
    # Count periods in each regime by name
    regime_counts = {name: 0 for name in ['Bull', 'Bear', 'Sideways']}
    for i, name in enumerate(regime_names):
        mask = hidden_states == i
        regime_counts[name] = int(np.sum(mask))

    regime_stats = {
        'Total Periods': total_periods,
        'Bull Periods': regime_counts['Bull'],
        'Bear Periods': regime_counts['Bear'],
        'Sideways Periods': regime_counts['Sideways'],
        'Regime Changes': int(np.sum(np.diff(hidden_states) != 0)),
        'Average Duration': float(total_periods / (np.sum(np.diff(hidden_states) != 0) + 1))
    }

    # percentages
    for key in ['Bull', 'Bear', 'Sideways']:
        regime_stats[f'{key} %'] = f"{(regime_stats[f'{key} Periods'] / total_periods * 100):.1f}%" if total_periods > 0 else "0.0%"

    # -------------------------
    # Save regime_stats JSON
    # -------------------------
    with open(f'{output_dir}/regime_stats.json', 'w') as f:
        json.dump(regime_stats, f, indent=2)

    # -------------------------
    # Performance metrics (backtest results)
    # -------------------------
    if backtest_results:
        performance_metrics = {
            'Train Log-Likelihood': float(backtest_results.get('train_log_prob', np.nan)),
            'Test Log-Likelihood': float(backtest_results.get('test_log_prob', np.nan)),
            'Test AIC': float(backtest_results.get('test_aic', np.nan)),
            'Test BIC': float(backtest_results.get('test_bic', np.nan)),
            'Degradation %': float(backtest_results.get('degradation', np.nan)),
            'Average Regime Duration': float(backtest_results.get('avg_regime_duration', np.nan)),
            'Number of Regime Changes': int(backtest_results.get('n_regime_changes', 0)),
            'Decision': backtest_results.get('decision', '')
        }

        with open(f'{output_dir}/performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=2)

        # Generate performance summary CSV
        metrics_df = pd.DataFrame([performance_metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv(f'{output_dir}/performance_summary.csv')

    # -------------------------
    # Return metadata
    # -------------------------
    return {
        'output_dir': output_dir,
        'regime_stats': regime_stats,
        'files_created': os.listdir(output_dir)
    }

def calculate_risk_metrics(data, hidden_states, regime_types=None):
    # Extract return columns (only price/return columns; exclude engineered features)
    return_cols = _extract_return_columns(data)

    if len(return_cols) == 0:
        return {}

    # Calculate metrics for each regime
    risk_metrics = {}

    if regime_types is None:
        # Fallback: analyze regime characteristics
        n_states = len(np.unique(hidden_states))
        regime_types = _analyze_regime_characteristics(hidden_states, data, n_states)

    for regime, name in enumerate(regime_types):
        mask = hidden_states == regime
        if np.sum(mask) == 0:
            continue

        regime_data = data[mask][return_cols]

        # regime_data holds per-day returns for each asset in return_cols
        # Compute daily portfolio-like return as the cross-sectional mean across assets
        daily_regime_returns = regime_data.mean(axis=1).dropna()

        # Calculate average daily return and daily volatility
        mean_daily = daily_regime_returns.mean() if len(daily_regime_returns) > 0 else 0.0
        std_daily = daily_regime_returns.std()

        # Use geometric daily risk-free rate consistent with PortfolioMetrics
        annual_rf = 0.02
        daily_rf = (1 + annual_rf) ** (1/252) - 1
        
        # Annualized Sharpe
        sharpe = ((mean_daily - daily_rf) / std_daily) * np.sqrt(252)

        # Calculate max drawdown
        cumulative_returns = (1 + regime_data.mean(axis=1)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        risk_metrics[name] = {
            'Average Return': float(mean_daily),
            'Volatility': float(std_daily),
            'Sharpe Ratio': float(sharpe) if not np.isnan(sharpe) else None,
            'Max Drawdown': float(max_drawdown),
            'Periods': int(np.sum(mask))
        }

    return risk_metrics


def run_portfolio_backtest(hidden_states, data, output_dir='dashboard_outputs/backtest_results', generate_visualizations=True):
    """
    Run portfolio backtest using HMM regime predictions.

    Args:
        hidden_states: Array of regime predictions
        data: DataFrame with feature data
        output_dir: Directory for output files
        generate_visualizations: If False, skip visualization generation (faster for grid search)

    Returns:
        Dictionary with backtest results and metrics
    """
    import json
    from data.data_utils import get_bond_data, get_indices
    from portfolio_backtest import PortfolioBacktester
    import shutil

    # Create output directories only if generating visualizations
    if generate_visualizations:
        from portfolio_visualizer import PortfolioVisualizer
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/charts', exist_ok=True)
        os.makedirs(f'{output_dir}/data', exist_ok=True)

    # Fetch bond data (TLT) for same date range
    bond_data = get_bond_data("TLT", interval="1d")
    if bond_data.empty:
        raise ValueError("Failed to fetch bond data (TLT)")

    # Get stock data (use S&P 500 ^GSPC as proxy)
    stock_data = get_indices(["^GSPC"], interval="1d")
    if stock_data.empty:
        raise ValueError("Failed to fetch S&P 500 data")

    # Align dates with feature data
    feature_dates = data.index
    bond_data = bond_data.loc[bond_data.index.isin(feature_dates)]
    stock_data = stock_data.loc[stock_data.index.isin(feature_dates)]

    # Further align all three datasets
    common_dates = feature_dates.intersection(bond_data.index).intersection(stock_data.index)
    bond_data = bond_data.loc[common_dates]
    stock_data = stock_data.loc[common_dates]

    # hidden_states is aligned with feature_dates, need to extract values for common_dates
    hidden_states_series = pd.Series(hidden_states, index=feature_dates)
    aligned_regime_predictions = hidden_states_series.loc[common_dates].values

    # Analyze regime characteristics to get labels based on thresholds
    # This determines which state index corresponds to which regime label
    n_states = len(np.unique(hidden_states))
    regime_types = _analyze_regime_characteristics(aligned_regime_predictions, data.loc[common_dates], n_states)

    # Validate regime types - warn if Unknown regimes detected
    if "Unknown" in regime_types:
        unknown_count = regime_types.count("Unknown")
        import warnings
        warnings.warn(
            f"⚠ Portfolio backtest contains {unknown_count} 'Unknown' regime(s) out of {n_states} total. "
            f"Unknown regimes use conservative allocation (60% stocks, 30% bonds, 10% cash). "
            f"Detected regimes: {regime_types}\n"
            f"To fix: Ensure include_returns=True and n_stocks>0 or n_indices>0 in configuration.",
            UserWarning,
            stacklevel=2
        )

    # Create mapping from state index to label
    state_to_label = {i: regime_types[i] for i in range(n_states)}
    
    # Define allocations by LABEL (not by state index) - this allows thresholds to affect allocations
    allocation_by_label = {
        'Bull': {'stocks': 1.0, 'bonds': 0.0, 'cash': 0.0},  # Bull: 100% stocks
        'Bear': {'stocks': 0.3, 'bonds': 0.5, 'cash': 0.2},  # Bear: 50% bonds, 50% cash (defensive but not flat)
        'Sideways': {'stocks': 0.8, 'bonds': 0.1, 'cash': 0.1},  # Sideways: 70% stocks, 20% bonds, 10% cash
        'Unknown': {'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1},  # Unknown: conservative balanced (same as Sideways)
    }
    
    # Map state indices to allocations based on their labels
    # This ensures allocations change when thresholds change the labeling
    regime_allocations = {}
    for state_idx in range(n_states):
        label = state_to_label[state_idx]  # Should always exist for valid state indices
        if label not in allocation_by_label:
            raise ValueError(f"Regime label '{label}' not found in allocation_by_label. Valid labels: {list(allocation_by_label.keys())}")
        regime_allocations[state_idx] = allocation_by_label[label]

    # Initialize backtester
    backtester = PortfolioBacktester(
        stock_data=stock_data.iloc[:, 0] if isinstance(stock_data, pd.DataFrame) else stock_data,
        bond_data=bond_data.iloc[:, 0] if isinstance(bond_data, pd.DataFrame) else bond_data,
        regime_predictions=aligned_regime_predictions,
        initial_capital=100000
    )

    # Run all strategies and compare
    comparison_results = backtester.compare_all_strategies(regime_allocations)

    strategy_results = comparison_results['strategy_results']
    metrics = comparison_results['metrics']
    best_rebalance_freq = comparison_results['best_rebalance_freq']

    # Generate visualizations and save files (only if requested)
    visualization_paths = []
    if generate_visualizations:
        visualizer = PortfolioVisualizer(output_dir=f'{output_dir}/charts')
        visualization_paths = visualizer.save_all_visualizations(
            comparison_results,
            regime_predictions=aligned_regime_predictions,
            dates=common_dates,
            stock_data=backtester.stock_prices
        )

        # Save strategy results to CSV
        strategy_df = pd.DataFrame(strategy_results)
        strategy_df.index.name = 'Date'
        strategy_df.to_csv(f'{output_dir}/data/strategy_results.csv')

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'Strategy'
        metrics_df.to_csv(f'{output_dir}/data/metrics_summary.csv')

        # Save best strategy info to JSON
        best_strategy_info = {
            'best_rebalance_freq': best_rebalance_freq,
            'best_sharpe': comparison_results['best_sharpe'],
            'best_strategy_name': f'Regime-Based ({best_rebalance_freq})' if best_rebalance_freq else None
        }
        with open(f'{output_dir}/data/best_strategy.json', 'w') as f:
            json.dump(best_strategy_info, f, indent=2)

        # Save regime allocations history
        regime_allocation_history = []
        for i, date in enumerate(common_dates):
            regime = int(aligned_regime_predictions[i])
            allocation = regime_allocations[regime]
            regime_label = state_to_label.get(regime, 'Unknown')
            regime_allocation_history.append({
                'Date': date,
                'Regime': regime,
                'Regime_Name': regime_label,  # Use actual label from threshold analysis
                'Stock_Pct': allocation['stocks'],
                'Bond_Pct': allocation.get('bonds', 0.0),
                'Cash_Pct': allocation.get('cash', 0.0)
            })
        regime_alloc_df = pd.DataFrame(regime_allocation_history)
        regime_alloc_df.to_csv(f'{output_dir}/data/regime_allocations.csv', index=False)

    chart_files = [os.path.basename(p) for p in visualization_paths] if visualization_paths else []

    # Save metadata
    metadata = {
        'run_timestamp': datetime.now().isoformat(),
        'date_range': {'start': str(common_dates[0]), 'end': str(common_dates[-1])},
        'strategies': list(strategy_results.keys()),
        'rebalancing_frequencies': ['regime_change', 'monthly', 'quarterly'],
        'best_strategy': f'Regime-Based ({best_rebalance_freq})' if best_rebalance_freq else None,
        'transaction_cost': 0.001,
        'regime_allocations': allocation_by_label,  # Use label-based allocations
        'state_to_label_mapping': state_to_label,  # Show which state maps to which label
        'chart_files': chart_files
    }

    try:
        with open(f'{output_dir}/backtest_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {output_dir}/backtest_metadata.json")
    except Exception as e:
        print(f"Failed to save metadata: {e}")


    return {
        'strategy_results': strategy_results,
        'metrics': metrics,
        'best_rebalance_freq': best_rebalance_freq,
        'visualization_paths': visualization_paths,
        'output_dir': output_dir
    }


def main(
    n_stocks=18,
    n_indices=0,
    volatility_window=10,
    rsi_period=14,
    momentum_period=10,
    include_returns=True,
    include_volatility=True,
    include_rsi=False,
    include_momentum=False,
    include_market_breadth=False,
    n_iter=5000,
    covariance_type='full',
    random_state=42,
    backtest=True,
    train_ratio=.8):
    """Run HMM model with customizable parameters."""
    run_hmm_model(
        n_stocks=n_stocks,
        n_indices=n_indices,
        volatility_window=volatility_window,
        rsi_period=rsi_period,
        momentum_period=momentum_period,
        include_returns=include_returns,
        include_volatility=include_volatility,
        include_rsi=include_rsi,
        include_momentum=include_momentum,
        include_market_breadth=include_market_breadth,
        n_iter=n_iter,
        covariance_type=covariance_type,
        random_state=random_state,
        backtest=backtest,
        train_ratio=train_ratio)

if __name__ == "__main__":
    main()