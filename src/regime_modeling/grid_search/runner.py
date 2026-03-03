import multiprocessing as mp
from itertools import product
from datetime import datetime, timedelta
import logging

from regime_modeling.config import DEFAULT_HMM_PARAMS
from regime_modeling.models.hmm import run_hmm_model
from regime_modeling.portfolio.backtest import run_portfolio_backtest
from regime_modeling.analysis.scoring import calculate_composite_score

logger = logging.getLogger(__name__)


def _run_single_config(args):
    """Run single configuration for parallel processing."""
    config_id, params = args

    try:
        if params['n_stocks'] <= 0:
            raise ValueError(f"n_stocks must be > 0 for grid search (got {params['n_stocks']})")

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
            random_state=params['random_state'],
            generate_outputs=False
        )

        results['config_id'] = config_id
        results['params'] = params
        try:
            test_states = results['test_states']
            test_data = results['test_data']

            portfolio_results = run_portfolio_backtest(test_states, test_data, generate_visualizations=False)

            best_freq = portfolio_results['best_rebalance_freq']
            best_strategy = f'Regime-Based ({best_freq})'
            portfolio_metrics = portfolio_results['metrics'][best_strategy]
            results['portfolio_cagr'] = portfolio_metrics['CAGR']
            results['portfolio_sharpe'] = portfolio_metrics['Sharpe Ratio']
            results['portfolio_sortino'] = portfolio_metrics['Sortino Ratio']
            results['portfolio_max_dd'] = portfolio_metrics['Max Drawdown']
            results['portfolio_volatility'] = portfolio_metrics['Volatility']
            results['portfolio_calmar'] = portfolio_metrics['Calmar Ratio']
            results['portfolio_win_rate'] = portfolio_metrics['Win Rate']
            results['best_rebalance_freq'] = best_freq

        except Exception as e:
            logger.warning(f"  Config {config_id}: Portfolio backtest failed ({str(e)}), using defaults")
            results['portfolio_cagr'] = 0.0
            results['portfolio_sharpe'] = 0.0
            results['portfolio_sortino'] = 0.0
            results['portfolio_max_dd'] = -0.5
            results['portfolio_volatility'] = 0.3
            results['portfolio_calmar'] = 0.0
            results['portfolio_win_rate'] = 0.5
            results['best_rebalance_freq'] = 'unknown'

        results['score'] = calculate_composite_score(results)

        return results

    except Exception as e:
        logger.error(f"Config {config_id} failed: {str(e)}")
        return None


def run_grid_search(
    n_stocks_range=None,
    n_indices_range=None,
    volatility_window_range=None,
    rsi_period_range=None,
    momentum_period_range=None,
    n_iter_range=None,
    covariance_type_range=None,
    random_state=DEFAULT_HMM_PARAMS['random_state'],
    feature_combinations='auto',
    train_ratio=DEFAULT_HMM_PARAMS['train_ratio'],
    n_processes=None,
    show_progress=False
):
    """
    Execute parallel grid search over HMM parameters.
    
    Returns:
        List of configuration results (dictionaries)
    """
    from regime_modeling.config import DEFAULT_GRID_SEARCH_RANGES as _DGSR
    from regime_modeling.config import N_REGIMES
    
    if n_stocks_range is None:
        n_stocks_range = _DGSR['n_stocks_range']
    if n_indices_range is None:
        n_indices_range = _DGSR['n_indices_range']
    if volatility_window_range is None:
        volatility_window_range = _DGSR['volatility_window_range']
    if rsi_period_range is None:
        rsi_period_range = _DGSR['rsi_period_range']
    if momentum_period_range is None:
        momentum_period_range = _DGSR['momentum_period_range']
    if n_iter_range is None:
        n_iter_range = _DGSR['n_iter_range']
    if covariance_type_range is None:
        covariance_type_range = _DGSR['covariance_type_range']

    logger.info("\n" + "="*70)
    logger.info("GRID SEARCH - PARALLEL PARAMETER OPTIMIZATION")
    logger.info("="*70)

    if any(n <= 0 for n in n_stocks_range):
        raise ValueError(f"Grid search requires stocks: n_stocks_range must not contain values <= 0. Got: {n_stocks_range}")

    logger.info("\n✓ Validated: Backtesting enabled and stocks required for all configurations")

    if feature_combinations == 'auto':
        feature_combos = [
            {'returns': True, 'volatility': True, 'rsi': False, 'momentum': False, 'market_breadth': False},
            {'returns': True, 'volatility': True, 'rsi': True, 'momentum': True, 'market_breadth': False},
            {'returns': True, 'volatility': True, 'rsi': False, 'momentum': True, 'market_breadth': False},
            {'returns': True, 'volatility': True, 'rsi': True, 'momentum': False, 'market_breadth': False},
            {'returns': True, 'volatility': True, 'rsi': False, 'momentum': False, 'market_breadth': True},
            {'returns': True, 'volatility': True, 'rsi': True, 'momentum': True, 'market_breadth': True},
        ]
    else:
        feature_combos = feature_combinations

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
            'n_components': N_REGIMES
        }
        param_grid.append((config_id, params))
        config_id += 1

    total_configs = len(param_grid)
    logger.info(f"\nTotal configurations to test: {total_configs}")
    logger.info(f"Parameter ranges:")
    logger.info(f"  n_stocks: {n_stocks_range}")
    logger.info(f"  n_indices: {n_indices_range}")
    logger.info(f"  volatility_window: {volatility_window_range}")
    logger.info(f"  rsi_period: {rsi_period_range}")
    logger.info(f"  momentum_period: {momentum_period_range}")
    logger.info(f"  n_iter: {n_iter_range}")
    logger.info(f"  covariance_type: {covariance_type_range}")
    logger.info(f"  Feature combinations: {len(feature_combos)}")

    if n_processes is None:
        n_processes = max(1, mp.cpu_count() // 2)

    estimated_minutes_low = total_configs / n_processes * 0.5
    estimated_minutes_high = total_configs / n_processes * 2
    expected_completion = datetime.now() + timedelta(minutes=estimated_minutes_high)

    logger.info(f"\nUsing {n_processes} parallel processes (half of available cores)")
    logger.info(f"Estimated time: {estimated_minutes_low:.1f}-{estimated_minutes_high:.1f} minutes ({estimated_minutes_low/60:.1f}-{estimated_minutes_high/60:.1f} hours)")
    logger.info(f"Expected completion: {expected_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\nStarting grid search...\n")

    results_list = []

    if n_processes > 1:
        with mp.Pool(processes=n_processes) as pool:
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

    logger.info(f"\nCompleted! {len(results_list)}/{total_configs} configurations succeeded")

    if len(results_list) == 0:
        logger.error("\n" + "="*70)
        logger.error("ERROR: All configurations failed!")
        logger.error("="*70)
        logger.error("\nPlease check:")
        logger.error("  1. Data quality and completeness")
        logger.error("  2. Feature engineering parameters")
        logger.error("  3. Error messages above for specific failures")
        logger.error("="*70 + "\n")
        return []

    return results_list
