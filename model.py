import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import norm
from features import get_enhanced_features_for_model
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import multiprocessing as mp
from itertools import product
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def _split_time_series_data(data, train_ratio=0.8):
    """Split time-series data chronologically into train/test sets."""
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data, split_idx


def _analyze_regime_characteristics(states, data, n_states):
    """Analyze economic characteristics of each regime."""
    return_cols = [col for col in data.columns if 'stock_' in col.lower() and '_ma_' not in col]
    regime_types = []

    for regime in range(n_states):
        mask = states == regime
        if len(data[mask]) == 0 or len(return_cols) == 0:
            regime_types.append("Unknown")
            continue

        regime_returns = data[mask][return_cols].mean(axis=1)
        mean_return = regime_returns.mean()
        std_return = regime_returns.std()

        if mean_return > 0.001 and std_return < data[return_cols].std().mean():
            regime_types.append("Bull")
        elif mean_return < -0.001:
            regime_types.append("Bear")
        else:
            regime_types.append("Sideways")

    return regime_types


def _save_backtest_results(results, params):
    """Append backtest results to Results.md file."""
    results_file = 'Results.md'

    try:
        with open(results_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        content = "# HMM Backtesting Results\n\n"

    # Create feature string
    features = []
    if params['include_returns']: features.append('Ret')
    if params['include_volatility']: features.append('Vol')
    if params['include_rsi']: features.append('RSI')
    if params['include_momentum']: features.append('Mom')
    if params['include_market_breadth']: features.append('MB')
    feature_str = '+'.join(features)

    # Format dates
    train_date = f"{results['train_start']} to {results['train_end']}"
    test_date = f"{results['test_start']} to {results['test_end']}"

    # Portfolio ready status
    degradation = results['degradation']
    avg_duration = results['avg_regime_duration']

    if degradation < 10 and 5 <= avg_duration <= 50:
        portfolio_ready = "✅ YES"
    elif degradation < 30:
        portfolio_ready = "⚠️ CAUTION"
    else:
        portfolio_ready = "❌ NO"

    # Create regime allocation string
    regime_info = []
    for i, rtype in enumerate(results['regime_types']):
        if rtype == "Bull":
            alloc = "80/20"
        elif rtype == "Bear":
            alloc = "30/70"
        else:
            alloc = "50/50"
        regime_info.append(f"R{i}:{rtype}({alloc})")
    regime_str = " | ".join(regime_info)

    # Append new row to backtesting section
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_entry = f"\n## Backtest Run - {timestamp}\n"
    new_entry += f"**Config:** stocks={params['n_stocks']}, indices={params['n_indices']}, features={feature_str}\n"
    new_entry += f"**Train:** {train_date}\n"
    new_entry += f"**Test:** {test_date}\n"
    new_entry += f"**Performance:** Train LogL={results['train_log_prob']:.0f}, Test LogL={results['test_log_prob']:.0f}, Degradation={degradation:.1f}%\n"
    new_entry += f"**Metrics:** AIC={results['test_aic']:.0f}, BIC={results['test_bic']:.0f}\n"
    new_entry += f"**Trading:** Avg Duration={avg_duration:.1f} periods, Changes={results['n_regime_changes']}\n"
    new_entry += f"**Regimes:** {regime_str}\n"
    new_entry += f"**Portfolio Ready:** {portfolio_ready}\n"
    new_entry += f"---\n"

    with open(results_file, 'a') as f:
        f.write(new_entry)


def run_hmm_model(n_stocks=10, n_indices=5, volatility_window=20,
                  rsi_period=14, momentum_period=10,
                  include_returns=True, include_volatility=True,
                  include_rsi=False, include_momentum=False,
                  include_market_breadth=False, backtest=False, train_ratio=0.8,
                  n_iter=5000, covariance_type='full', random_state=42):
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
        backtest: If True, perform train/test split backtesting
        train_ratio: Ratio of data for training (default 0.8)
        n_iter: Number of iterations for HMM training
        covariance_type: Type of covariance parameters to use ('full', 'tied', 'diag', 'spherical')
        random_state: Random state for reproducibility

    Returns:
        If backtest=False: Tuple of (model, hidden_states, log_probability, features_df, scaler)
        If backtest=True: Dictionary with backtest results
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

    # Split data if backtesting
    if backtest:
        train_data, test_data, split_idx = _split_time_series_data(data, train_ratio)
        print(f"Train: {len(train_data)} samples | Test: {len(test_data)} samples")
        print(f"Split date: {data.index[split_idx]}")
        training_data = train_data
    else:
        training_data = data

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

    # If not backtesting, return standard results
    if not backtest:
        # Decode hidden states using Viterbi algorithm
        print("\nStep 4: Decoding hidden states...")
        log_probability, hidden_states = model.decode(
            obs_scaled,
            lengths=len(obs_scaled),
            algorithm='viterbi'
        )

        print(f'\nLog Probability: {log_probability:.2f}')

        # Print state distribution
        state_counts = np.bincount(hidden_states)
        print("\nState distribution:")
        for i, count in enumerate(state_counts):
            print(f"  State {i}: {count} periods ({count/len(hidden_states)*100:.1f}%)")

        # Generate dashboard outputs
        print("\nStep 5: Generating dashboard outputs...")
        dashboard_info = generate_dashboard_outputs(model, data, hidden_states, scaler)
        print(f"Dashboard files saved to: {dashboard_info['output_dir']}/")
        print(f"Files created: {', '.join(dashboard_info['files_created'])}")

        # Calculate and save risk metrics
        risk_metrics = calculate_risk_metrics(data, hidden_states)
        if risk_metrics:
            import json
            with open(f"{dashboard_info['output_dir']}/risk_metrics.json", 'w') as f:
                json.dump(risk_metrics, f, indent=2)
            print(f"Risk metrics saved")

        # Run portfolio backtest
        print("\nStep 6: Running portfolio backtest...")
        try:
            backtest_results = run_portfolio_backtest(hidden_states, data)
            print(f"Portfolio backtest complete!")
            print(f"Best rebalancing frequency: {backtest_results['best_rebalance_freq']}")
            print(f"Visualizations saved to: {backtest_results['output_dir']}/charts/")
        except Exception as e:
            print(f"Portfolio backtest failed: {e}")
            print("Continuing without backtest results...")

        print("\n" + "="*60)
        print("HMM MODEL COMPLETE")
        print("="*60 + "\n")

        return model, hidden_states, log_probability, data, scaler

    # Backtesting mode - evaluate on test set
    print("\nStep 4: Evaluating on test set...")

    # Get training performance
    train_log_prob = model.score(obs_scaled)

    # Evaluate on test set
    test_scaled = scaler.transform(test_data)
    test_log_prob = model.score(test_scaled)
    test_states = model.predict(test_scaled)

    # Calculate metrics
    n_features = test_data.shape[1]
    n_samples = len(test_data)
    n_params_means = n_states * n_features
    n_params_covars = n_states * n_features * (n_features + 1) // 2
    n_params_transitions = n_states * (n_states - 1)
    n_params = n_params_means + n_params_covars + n_params_transitions

    test_aic = -2 * test_log_prob + 2 * n_params
    test_bic = -2 * test_log_prob + n_params * np.log(n_samples)

    # Calculate degradation (normalize by sample size for fair comparison)
    # Log-likelihoods are negative, so if test is worse (more negative per sample),
    # degradation will be positive
    train_log_prob_avg = train_log_prob / len(obs_scaled)
    test_log_prob_avg = test_log_prob / len(test_scaled)
    degradation = ((test_log_prob_avg - train_log_prob_avg) / train_log_prob_avg) * 100

    # Regime stability
    regime_changes = np.sum(np.diff(test_states) != 0)
    avg_regime_duration = len(test_states) / (regime_changes + 1)

    # Analyze regime characteristics
    regime_types = _analyze_regime_characteristics(test_states, test_data, n_states)

    # Print concise results
    print(f"Train LogL: {train_log_prob_avg:.2f}/sample ({train_log_prob:.0f} total)")
    print(f"Test LogL: {test_log_prob_avg:.2f}/sample ({test_log_prob:.0f} total)")
    print(f"Degradation: {degradation:.1f}%")
    print(f"Test AIC: {test_aic:.0f} | Test BIC: {test_bic:.0f}")
    print(f"Avg Regime Duration: {avg_regime_duration:.1f} periods | Changes: {regime_changes}")
    print(f"Regimes: {' | '.join([f'{i}={regime_types[i]}' for i in range(n_states)])}")

    # Decision
    if degradation < 10 and 5 <= avg_regime_duration <= 50:
        decision = "✅ READY FOR PORTFOLIO"
    elif degradation < 30:
        decision = "⚠️ USE WITH CAUTION"
    else:
        decision = "❌ NOT RECOMMENDED"
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
        'test_states': test_states,
        'train_start': training_data.index[0].strftime('%Y-%m-%d'),
        'train_end': training_data.index[-1].strftime('%Y-%m-%d'),
        'test_start': test_data.index[0].strftime('%Y-%m-%d'),
        'test_end': test_data.index[-1].strftime('%Y-%m-%d'),
        'decision': decision
    }

    # Save to Results.md
    params = {
        'n_stocks': n_stocks,
        'n_indices': n_indices,
        'include_returns': include_returns,
        'include_volatility': include_volatility,
        'include_rsi': include_rsi,
        'include_momentum': include_momentum,
        'include_market_breadth': include_market_breadth
    }
    _save_backtest_results(results, params)
    print("✓ Results saved to Results.md")

    # Generate dashboard outputs for backtest
    print("\nGenerating dashboard outputs...")
    dashboard_info = generate_dashboard_outputs(model, test_data, test_states, scaler, results)
    print(f"Dashboard files saved to: {dashboard_info['output_dir']}/")

    # Calculate and save risk metrics
    risk_metrics = calculate_risk_metrics(test_data, test_states)
    if risk_metrics:
        import json
        with open(f"{dashboard_info['output_dir']}/risk_metrics.json", 'w') as f:
            json.dump(risk_metrics, f, indent=2)
        print(f"Risk metrics saved")

    # Run portfolio backtest on test data
    print("\nRunning portfolio backtest on test data...")
    try:
        backtest_results = run_portfolio_backtest(test_states, test_data)
        print(f"Portfolio backtest complete!")
        print(f"Best rebalancing frequency: {backtest_results['best_rebalance_freq']}")
        print(f"Visualizations saved to: {backtest_results['output_dir']}/charts/\n")
    except Exception as e:
        print(f"Portfolio backtest failed: {e}")
        print("Continuing without backtest results...\n")

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
    # For full covariance: n_states * n_features * (n_features + 1) / 2
    # Plus transition matrix: n_states * (n_states - 1)
    # Plus means: n_states * n_features
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


def compare_feature_combinations(n_stocks=15, n_indices=5, volatility_window=20,
                                 rsi_period=14, momentum_period=10):
    """
    Compare HMM model performance across different feature combinations.

    Args:
        n_stocks: Number of stocks to include
        n_indices: Number of indices to include
        volatility_window: Window for volatility calculation
        rsi_period: Period for RSI calculation
        momentum_period: Period for momentum calculation

    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*60)
    print("COMPARING FEATURE COMBINATIONS")
    print("="*60)

    # Define feature combinations to test
    feature_combinations = [
        {
            'name': 'All Features',
            'returns': True, 'volatility': True, 'rsi': True,
            'momentum': True, 'market_breadth': True
        },
        {
            'name': 'Returns Only',
            'returns': True, 'volatility': False, 'rsi': False,
            'momentum': False, 'market_breadth': False
        },
        {
            'name': 'Returns + Volatility',
            'returns': True, 'volatility': True, 'rsi': False,
            'momentum': False, 'market_breadth': False
        },
        {
            'name': 'Returns + Market Breadth',
            'returns': True, 'volatility': False, 'rsi': False,
            'momentum': False, 'market_breadth': True
        },
        {
            'name': 'Returns + Technical (RSI + Momentum)',
            'returns': True, 'volatility': False, 'rsi': True,
            'momentum': True, 'market_breadth': False
        },
        {
            'name': 'Returns + Volatility + Market Breadth',
            'returns': True, 'volatility': True, 'rsi': False,
            'momentum': False, 'market_breadth': True
        },
        {
            'name': 'No Technical Indicators',
            'returns': True, 'volatility': True, 'rsi': False,
            'momentum': False, 'market_breadth': True
        },
        {
            'name': 'Technical Only (No Breadth)',
            'returns': True, 'volatility': True, 'rsi': True,
            'momentum': True, 'market_breadth': False
        }
    ]

    results = []

    for i, combo in enumerate(feature_combinations, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(feature_combinations)}: {combo['name']}")
        print(f"{'='*60}")

        try:
            # Run model with this feature combination
            model, hidden_states, log_prob, data, scaler = run_hmm_model(
                n_stocks=n_stocks,
                n_indices=n_indices,
                volatility_window=volatility_window,
                rsi_period=rsi_period,
                momentum_period=momentum_period,
                include_returns=combo['returns'],
                include_volatility=combo['volatility'],
                include_rsi=combo['rsi'],
                include_momentum=combo['momentum'],
                include_market_breadth=combo['market_breadth']
            )

            # Scale data for metrics calculation
            from sklearn.preprocessing import StandardScaler
            scaler_temp = StandardScaler()
            data_scaled = scaler_temp.fit_transform(data)

            # Calculate metrics
            metrics = calculate_model_metrics(model, data_scaled, data.shape[1])

            # Store results
            result = {
                'Feature Set': combo['name'],
                'N Features': metrics['n_features'],
                'Log Likelihood': metrics['log_likelihood'],
                'AIC': metrics['aic'],
                'BIC': metrics['bic'],
                'N Parameters': metrics['n_params']
            }
            results.append(result)

            print(f"\nMetrics for {combo['name']}:")
            print(f"  Features: {metrics['n_features']}")
            print(f"  Log Likelihood: {metrics['log_likelihood']:.2f}")
            print(f"  AIC: {metrics['aic']:.2f}")
            print(f"  BIC: {metrics['bic']:.2f}")

        except Exception as e:
            print(f"\nError with {combo['name']}: {str(e)}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by BIC (lower is better)
    results_df = results_df.sort_values('BIC')

    print("\n" + "="*60)
    print("COMPARISON RESULTS (sorted by BIC - lower is better)")
    print("="*60)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv('feature_comparison_results.csv', index=False)
    print(f"\nResults saved to: feature_comparison_results.csv")

    print("\n" + "="*60)
    print("Best performing feature set (by BIC):")
    print(f"  {results_df.iloc[0]['Feature Set']}")
    print(f"  BIC: {results_df.iloc[0]['BIC']:.2f}")
    print(f"  AIC: {results_df.iloc[0]['AIC']:.2f}")
    print(f"  Features: {results_df.iloc[0]['N Features']}")
    print("="*60 + "\n")

    return results_df


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

    # Save model
    model_path = f'models/{filename_prefix}_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'params': params,
            'metrics': metrics,
            'timestamp': timestamp
        }, f)

    # Also save as 'latest' for easy access
    latest_path = f'models/{filename_prefix}_latest.pkl'
    with open(latest_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'params': params,
            'metrics': metrics,
            'timestamp': timestamp
        }, f)

    return {
        'model_path': model_path,
        'latest_path': latest_path
    }


def load_model(filepath='models/best_hmm_model_latest.pkl'):
    """
    Load a saved HMM model from disk.

    Args:
        filepath: Path to the saved model file

    Returns:
        Dictionary with model, scaler, params, metrics, and timestamp
    """
    with open(filepath, 'rb') as f:
        saved_data = pickle.load(f)

    return saved_data


def _run_single_config(args):
    """
    Helper function to run a single configuration for parallel processing.
    Returns results dict or None if error.
    """
    config_id, params = args

    try:
        # Run backtest with these parameters
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
            backtest=True,
            train_ratio=params.get('train_ratio', 0.8),
            n_iter=params['n_iter'],
            covariance_type=params['covariance_type'],
            random_state=params['random_state']
        )

        # Add config info to results
        results['config_id'] = config_id
        results['params'] = params

        # Calculate regime quality metrics
        test_states = results['test_states']
        unique_states = np.unique(test_states)
        n_unique_regimes = len(unique_states)
        n_regime_changes = results['n_regime_changes']

        # Calculate regime diversity (how balanced are the regimes?)
        state_counts = np.bincount(test_states)
        state_probs = state_counts / len(test_states)
        # Entropy as a measure of diversity (higher is better, max log(3) for 3 states)
        regime_entropy = -np.sum(state_probs[state_probs > 0] * np.log(state_probs[state_probs > 0]))
        max_entropy = np.log(3)  # For 3 states
        regime_diversity_score = regime_entropy / max_entropy  # 0 to 1, higher is better

        # Penalize if not using all 3 regimes
        regime_usage_penalty = 0
        if n_unique_regimes < 3:
            regime_usage_penalty = (3 - n_unique_regimes) * 100  # Heavy penalty

        # Penalize if regimes don't switch enough (want meaningful regime changes)
        switching_penalty = 0
        if n_regime_changes < 5:
            switching_penalty = (5 - n_regime_changes) * 30  # Not enough switching
        elif n_regime_changes > len(test_states) / 3:
            switching_penalty = (n_regime_changes - len(test_states) / 3) * 0.5  # Too much switching

        # Regime duration penalty (want 5-50 days average)
        duration_penalty = 0
        avg_duration = results['avg_regime_duration']
        if avg_duration < 5:
            duration_penalty = (5 - avg_duration) * 40  # Too noisy
        elif avg_duration > 50:
            duration_penalty = (avg_duration - 50) * 5  # Too slow

        # Model performance penalty
        degradation_penalty = abs(results['degradation']) * 8  # Weight on out-of-sample performance

        # Normalize BIC (lower is better)
        bic_score = results['test_bic'] / 1000

        # COMPOSITE SCORE (lower is better)
        # Prioritizes: regime diversity, proper switching, good duration, model performance
        results['score'] = (
            degradation_penalty +           # Model generalization (weight ~8-16)
            regime_usage_penalty +          # Must use all regimes (weight 0-200)
            switching_penalty +             # Appropriate switching (weight 0-150)
            duration_penalty +              # Good regime duration (weight 0-200)
            bic_score +                     # Model fit (weight ~5-15)
            (1 - regime_diversity_score) * 50  # Regime balance (weight 0-50)
        )

        # Store additional metrics for analysis
        results['regime_diversity'] = regime_diversity_score
        results['n_unique_regimes'] = n_unique_regimes

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
    top_n=10
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
            'train_ratio': train_ratio
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
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one core free

    print(f"\nUsing {n_processes} parallel processes")
    print(f"Estimated time: {total_configs / n_processes * 0.5:.1f}-{total_configs / n_processes * 2:.1f} minutes")
    print("\nStarting grid search...\n")

    # Run parallel grid search
    results_list = []

    if n_processes > 1:
        # Parallel execution
        with mp.Pool(processes=n_processes) as pool:
            # Use imap_unordered for progress tracking
            completed = 0
            for result in pool.imap_unordered(_run_single_config, param_grid):
                if result is not None:
                    results_list.append(result)
                completed += 1
                if completed % max(1, total_configs // 20) == 0:  # Update every 5%
                    print(f"Progress: {completed}/{total_configs} ({completed/total_configs*100:.1f}%)")
    else:
        # Serial execution (for debugging)
        for i, config in enumerate(param_grid):
            result = _run_single_config(config)
            if result is not None:
                results_list.append(result)
            if (i + 1) % max(1, total_configs // 20) == 0:
                print(f"Progress: {i+1}/{total_configs} ({(i+1)/total_configs*100:.1f}%)")

    print(f"\nCompleted! {len(results_list)}/{total_configs} configurations succeeded")

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

        summary_data.append({
            'Config_ID': r['config_id'],
            'Score': r['score'],
            'Regime_Diversity': r.get('regime_diversity', 0),
            'N_Regimes_Used': r.get('n_unique_regimes', 0),
            'Regime_Changes': r['n_regime_changes'],
            'Avg_Duration': r['avg_regime_duration'],
            'Degradation%': r['degradation'],
            'Test_LogL': r['test_log_prob'],
            'Test_BIC': r['test_bic'],
            'n_stocks': p['n_stocks'],
            'n_indices': p['n_indices'],
            'vol_window': p['volatility_window'],
            'Features': feature_str,
            'Regimes': '|'.join(r['regime_types']),
            'Decision': r['decision']
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
    print(f"Score: {best['Score']:.2f}")
    print(f"\nParameters:")
    print(f"  n_stocks: {best['n_stocks']}")
    print(f"  n_indices: {best['n_indices']}")
    print(f"  volatility_window: {best['vol_window']}")
    print(f"  Features: {best['Features']}")
    print(f"\nRegime Quality:")
    print(f"  Regimes Used: {best['N_Regimes_Used']}/3")
    print(f"  Regime Diversity: {best['Regime_Diversity']:.3f} (1.0 = perfectly balanced)")
    print(f"  Regime Changes: {best['Regime_Changes']}")
    print(f"  Avg Regime Duration: {best['Avg_Duration']:.1f} periods")
    print(f"  Regime Types: {best['Regimes']}")
    print(f"\nModel Performance:")
    print(f"  Degradation: {best['Degradation%']:.1f}%")
    print(f"  Test Log-Likelihood: {best['Test_LogL']:.0f}")
    print(f"  Test BIC: {best['Test_BIC']:.0f}")
    print(f"\nDecision: {best['Decision']}")
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

    # Save the best model
    print("\nSaving best model...")
    best_result = [r for r in results_list if r['config_id'] == best['Config_ID']][0]

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

    return results_df


def generate_dashboard_outputs(model, data, hidden_states, scaler=None, backtest_results=None, output_dir='dashboard_outputs'):
    # Create output directory, removing old files if they exist
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate regime timeline visualization
    fig_regime = go.Figure()

    # Add regime background colors
    # Aggregate contiguous regime spans to speed up plotting
    regime_changes = np.where(np.diff(hidden_states) != 0)[0]
    starts = np.insert(regime_changes + 1, 0, 0)
    ends = np.append(regime_changes, len(hidden_states) - 1)
    for s, e in zip(starts, ends):
        regime = hidden_states[s]
        color = ['rgba(0,255,0,0.2)', 'rgba(255,0,0,0.2)', 'rgba(128,128,128,0.2)'][regime]
        fig_regime.add_vrect(
            x0=data.index[s], x1=data.index[e],
            fillcolor=color, layer="below", line_width=0
        )


    # Add a dummy trace for each regime to show in legend
    fig_regime.add_trace(go.Scatter(
        x=[data.index[0]], y=[0], mode='markers',
        marker=dict(size=10, color='green'), name='Bull Market'
    ))
    fig_regime.add_trace(go.Scatter(
        x=[data.index[0]], y=[0], mode='markers',
        marker=dict(size=10, color='red'), name='Bear Market'
    ))
    fig_regime.add_trace(go.Scatter(
        x=[data.index[0]], y=[0], mode='markers',
        marker=dict(size=10, color='gray'), name='Sideways Market'
    ))

    fig_regime.update_layout(
        title='Market Regime Timeline',
        xaxis_title='Date',
        yaxis_title='Regime',
        hovermode='x',
        height=400,
        showlegend=True
    )
    fig_regime.write_html(f'{output_dir}/regime_timeline.html')

    # Generate regime statistics
    regime_stats = {
        'Total Periods': len(hidden_states),
        'Bull Periods': int(np.sum(hidden_states == 0)),
        'Bear Periods': int(np.sum(hidden_states == 1)),
        'Sideways Periods': int(np.sum(hidden_states == 2)),
        'Regime Changes': int(np.sum(np.diff(hidden_states) != 0)),
        'Average Duration': float(len(hidden_states) / (np.sum(np.diff(hidden_states) != 0) + 1))
    }

    # Calculate regime percentages
    regime_stats['Bull %'] = f"{regime_stats['Bull Periods'] / regime_stats['Total Periods'] * 100:.1f}%"
    regime_stats['Bear %'] = f"{regime_stats['Bear Periods'] / regime_stats['Total Periods'] * 100:.1f}%"
    regime_stats['Sideways %'] = f"{regime_stats['Sideways Periods'] / regime_stats['Total Periods'] * 100:.1f}%"

    # Generate regime distribution pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Bull Market', 'Bear Market', 'Sideways Market'],
        values=[regime_stats['Bull Periods'], regime_stats['Bear Periods'], regime_stats['Sideways Periods']],
        marker=dict(colors=['green', 'red', 'gray'])
    )])
    fig_pie.update_layout(title='Regime Distribution', height=400)
    fig_pie.write_html(f'{output_dir}/regime_distribution.html')

    # Generate portfolio allocation recommendations
    allocations = pd.DataFrame({
        'Regime': ['Bull Market', 'Bear Market', 'Sideways Market'],
        'Stock %': [80, 30, 50],
        'Bond %': [20, 70, 50],
        'Description': [
            'High growth potential, allocate more to stocks',
            'Risk aversion, allocate more to bonds',
            'Balanced approach for uncertain markets'
        ]
    })
    allocations.to_csv(f'{output_dir}/portfolio_allocations.csv', index=False)

    # Generate allocation visualization
    fig_alloc = go.Figure(data=[
        go.Bar(name='Stocks', x=allocations['Regime'], y=allocations['Stock %'], marker_color='green'),
        go.Bar(name='Bonds', x=allocations['Regime'], y=allocations['Bond %'], marker_color='blue')
    ])
    fig_alloc.update_layout(
        title='Recommended Portfolio Allocation by Regime',
        xaxis_title='Regime',
        yaxis_title='Allocation %',
        barmode='stack',
        height=400
    )
    fig_alloc.write_html(f'{output_dir}/portfolio_allocation_chart.html')

    # Save regime statistics
    with open(f'{output_dir}/regime_stats.json', 'w') as f:
        import json
        json.dump(regime_stats, f, indent=2)

    # If backtest results available, generate performance metrics
    if backtest_results:
        performance_metrics = {
            'Train Log-Likelihood': float(backtest_results['train_log_prob']),
            'Test Log-Likelihood': float(backtest_results['test_log_prob']),
            'Test AIC': float(backtest_results['test_aic']),
            'Test BIC': float(backtest_results['test_bic']),
            'Degradation %': float(backtest_results['degradation']),
            'Average Regime Duration': float(backtest_results['avg_regime_duration']),
            'Number of Regime Changes': int(backtest_results['n_regime_changes']),
            'Decision': backtest_results['decision']
        }

        with open(f'{output_dir}/performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=2)

        # Generate performance summary table
        metrics_df = pd.DataFrame([performance_metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv(f'{output_dir}/performance_summary.csv')

    # Generate feature importance visualization if model has means
    try:
        n_states = model.n_components
        n_features = model.means_.shape[1]

        # Create heatmap of feature means per regime
        feature_cols = data.columns[:n_features] if len(data.columns) >= n_features else data.columns

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=model.means_,
            x=feature_cols,
            y=['Bull', 'Bear', 'Sideways'],
            colorscale='RdBu',
            zmid=0
        ))
        fig_heatmap.update_layout(
            title='Feature Means by Regime (Normalized)',
            xaxis_title='Features',
            yaxis_title='Regime',
            height=500
        )
        fig_heatmap.write_html(f'{output_dir}/feature_importance.html')
    except:
        pass

    return {
        'output_dir': output_dir,
        'regime_stats': regime_stats,
        'files_created': os.listdir(output_dir)
    }


def calculate_risk_metrics(data, hidden_states):
    # Extract return columns
    return_cols = [col for col in data.columns if 'stock_' in col.lower() and '_ma_' not in col]

    if len(return_cols) == 0:
        return {}

    # Calculate metrics for each regime
    risk_metrics = {}

    for regime, name in enumerate(['Bull', 'Bear', 'Sideways']):
        mask = hidden_states == regime
        if np.sum(mask) == 0:
            continue

        regime_data = data[mask][return_cols]

        # Calculate average returns
        avg_returns = regime_data.mean(axis=1).mean()

        # Calculate volatility (std of returns)
        volatility = regime_data.mean(axis=1).std()

        # Calculate Sharpe ratio (assuming risk-free rate of 0.02/252 per day)
        risk_free_rate = 0.02 / 252
        sharpe = (avg_returns - risk_free_rate) / volatility if volatility > 0 else 0

        # Calculate max drawdown
        cumulative_returns = (1 + regime_data.mean(axis=1)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        risk_metrics[name] = {
            'Average Return': float(avg_returns),
            'Volatility': float(volatility),
            'Sharpe Ratio': float(sharpe),
            'Max Drawdown': float(max_drawdown),
            'Periods': int(np.sum(mask))
        }

    return risk_metrics


def run_portfolio_backtest(hidden_states, data, output_dir='dashboard_outputs/backtest_results'):
    # Run portfolio backtest using HMM regime predictions
    import json
    from data.data_utils import get_bond_data, get_indices
    from portfolio_backtest import PortfolioBacktester
    from portfolio_visualizer import PortfolioVisualizer
    import shutil

    # Create output directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/charts', exist_ok=True)
    os.makedirs(f'{output_dir}/data', exist_ok=True)

    # Fetch bond data (TLT) for same date range
    bond_data = get_bond_data("TLT", period="15y", interval="1d")
    if bond_data.empty:
        raise ValueError("Failed to fetch bond data (TLT)")

    # Get stock data (use S&P 500 ^GSPC as proxy)
    stock_data = get_indices(["^GSPC"], period="15y", interval="1d")
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
    aligned_regime_predictions = hidden_states[:len(common_dates)]

    # Initialize backtester
    backtester = PortfolioBacktester(
        stock_data=stock_data.iloc[:, 0] if isinstance(stock_data, pd.DataFrame) else stock_data,
        bond_data=bond_data.iloc[:, 0] if isinstance(bond_data, pd.DataFrame) else bond_data,
        regime_predictions=aligned_regime_predictions,
        transaction_cost=0.001,  # 0.1%
        initial_capital=100000
    )

    # Define regime allocations (Bull: 80/20, Bear: 30/70, Sideways: 50/50)
    regime_allocations = {
        0: {'stocks': 0.80, 'bonds': 0.20},  # Bull
        1: {'stocks': 0.30, 'bonds': 0.70},  # Bear
        2: {'stocks': 0.50, 'bonds': 0.50}   # Sideways
    }

    # Run all strategies and compare
    comparison_results = backtester.compare_all_strategies(regime_allocations)

    strategy_results = comparison_results['strategy_results']
    metrics = comparison_results['metrics']
    best_rebalance_freq = comparison_results['best_rebalance_freq']

    # Generate visualizations
    visualizer = PortfolioVisualizer(output_dir=f'{output_dir}/charts')
    visualization_paths = visualizer.save_all_visualizations(
        comparison_results,
        regime_predictions=aligned_regime_predictions,
        dates=common_dates
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
        regime = aligned_regime_predictions[i]
        allocation = regime_allocations[regime]
        regime_allocation_history.append({
            'Date': date,
            'Regime': regime,
            'Regime_Name': ['Bull', 'Bear', 'Sideways'][regime],
            'Stock_Pct': allocation['stocks'],
            'Bond_Pct': allocation['bonds']
        })
    regime_alloc_df = pd.DataFrame(regime_allocation_history)
    regime_alloc_df.to_csv(f'{output_dir}/data/regime_allocations.csv', index=False)

    # Save metadata
    metadata = {
        'run_timestamp': datetime.now().isoformat(),
        'date_range': {
            'start': str(common_dates[0]),
            'end': str(common_dates[-1])
        },
        'strategies': list(strategy_results.keys()),
        'rebalancing_frequencies': ['regime_change', 'monthly', 'quarterly'],
        'best_strategy': f'Regime-Based ({best_rebalance_freq})' if best_rebalance_freq else None,
        'transaction_cost': 0.001,
        'regime_allocations': {
            'Bull': {'stocks': 0.8, 'bonds': 0.2},
            'Bear': {'stocks': 0.3, 'bonds': 0.7},
            'Sideways': {'stocks': 0.5, 'bonds': 0.5}
        },
        'chart_files': [os.path.basename(p) for p in visualization_paths]
    }
    with open(f'{output_dir}/backtest_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'strategy_results': strategy_results,
        'metrics': metrics,
        'best_rebalance_freq': best_rebalance_freq,
        'visualization_paths': visualization_paths,
        'output_dir': output_dir
    }


def main():
    run_hmm_model()

if __name__ == "__main__":
    main()