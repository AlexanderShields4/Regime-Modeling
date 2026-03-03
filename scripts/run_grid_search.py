"""
Comprehensive overnight grid search for optimal HMM parameters.
Expected runtime: 1-4 hours.

Run cache_data.py first to avoid API rate limiting.
"""

import logging
from regime_modeling import setup_logging
from regime_modeling.grid_search.reporting import grid_search_parameters
from regime_modeling.features import cache_exists
import multiprocessing as mp
from datetime import datetime, timedelta
import sys
from tqdm import tqdm

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE OVERNIGHT GRID SEARCH")
    logger.info("="*70)

    if not cache_exists():
        logger.error("\n" + "="*70)
        logger.error("ERROR: No cached data found!")
        logger.error("="*70)
        logger.info("\nRun cache_data.py first:")
        logger.info("    python cache_data.py")
        logger.info("="*70 + "\n")
        sys.exit(1)

    logger.info("\n✓ Cached data found - grid search will be fast!\n")

    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Display system info
    n_cores = mp.cpu_count()
    logger.info(f"Available CPU cores: {n_cores}")
    logger.info(f"Using {n_cores // 2} cores (half of available cores)")

    logger.info("\n" + "-"*70)
    logger.info("PARAMETER RANGES")
    logger.info("-"*70)

    n_stocks_range = [5, 7, 10, 12, 15, 18, 20, 25]
    n_indices_range = [0, 2, 3, 5, 7]

    volatility_window_range = [10, 20, 30]
    rsi_period_range = [14]
    momentum_period_range = [10]

    # Lower bound raised from 3000 (may not converge) to 5000.
    # Upper bound raised to 10000 to match the known-best config.
    n_iter_range = [5000, 7000, 10000]
    # 'tied' forces identical covariance across all regimes, which contradicts
    # the regime-switching premise. Only 'full' is searched.
    covariance_type_range = ['full']
    total_configs = (len(n_stocks_range) * len(n_indices_range) *
                    len(volatility_window_range) * len(rsi_period_range) *
                    len(momentum_period_range) * len(n_iter_range) *
                    len(covariance_type_range) * 6)

    logger.info(f"\nn_stocks: {n_stocks_range}")
    logger.info(f"n_indices: {n_indices_range}")
    logger.info(f"volatility_window: {volatility_window_range}")
    logger.info(f"rsi_period: {rsi_period_range}")
    logger.info(f"momentum_period: {momentum_period_range}")
    logger.info(f"n_iter: {n_iter_range}")
    logger.info(f"covariance_type: {covariance_type_range}")
    logger.info(f"feature_combinations: 6 (auto-selected)")

    logger.info(f"\nTotal configurations to test: {total_configs:,}")

    avg_time_per_config = 15
    total_seconds = (total_configs * avg_time_per_config) / (n_cores // 2)
    hours = total_seconds / 3600

    expected_completion_low = datetime.now() + timedelta(seconds=total_seconds)
    expected_completion_high = datetime.now() + timedelta(seconds=total_seconds * 2)

    logger.info(f"\nEstimated time: {hours:.1f} - {hours*2:.1f} hours")
    logger.info(f"Expected completion: {expected_completion_low.strftime('%Y-%m-%d %H:%M:%S')} - {expected_completion_high.strftime('%H:%M:%S')}")

    logger.info("\n" + "="*70)
    logger.info("STARTING COMPREHENSIVE SEARCH...")
    logger.info("="*70)
    logger.info("\nYou can safely minimize this window.")
    logger.info("Results will be saved automatically when complete.\n")

    results = grid_search_parameters(
        n_stocks_range=n_stocks_range,
        n_indices_range=n_indices_range,
        volatility_window_range=volatility_window_range,
        rsi_period_range=rsi_period_range,
        momentum_period_range=momentum_period_range,
        n_iter_range=n_iter_range,
        covariance_type_range=covariance_type_range,
        feature_combinations='auto',
        train_ratio=0.8,
        n_processes=None,
        top_n=20,
        show_progress=True
    )

    end_time = datetime.now()
    elapsed = end_time - start_time
    hours_elapsed = elapsed.total_seconds() / 3600

    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE SEARCH COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed: {hours_elapsed:.2f} hours ({elapsed.total_seconds()/60:.1f} minutes)")
    logger.info(f"\nConfigurations tested: {len(results)}/{total_configs}")
    logger.info(f"Success rate: {len(results)/total_configs*100:.1f}%")

    logger.info("\n" + "="*70)
    logger.info("GENERATED FILES")
    logger.info("="*70)
    logger.info("\n1. grid_search_results_*.csv")
    logger.info("   → Full results table with all configurations")
    logger.info("   → Sorted by score (lower = better)")
    logger.info("\n2. best_config.txt")
    logger.info("   → Ready-to-use code for best configuration")

    logger.info("="*70)
    logger.info("Grid search complete! Check the files above.")
    logger.info("="*70 + "\n")

    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    logger.info(f"\nBest Score: {results['Score'].min():.2f}")
    logger.info(f"Worst Score: {results['Score'].max():.2f}")
    logger.info(f"Average Score: {results['Score'].mean():.2f}")
    logger.info(f"\nBest Degradation: {results['Degradation%'].min():.1f}%")
    logger.info(f"Worst Degradation: {results['Degradation%'].max():.1f}%")
    logger.info(f"Average Degradation: {results['Degradation%'].mean():.1f}%")

    ready_count = results['Decision'].str.contains('✅').sum()
    logger.info(f"\nConfigurations ready for portfolio: {ready_count}/{len(results)} ({ready_count/len(results)*100:.1f}%)")

    logger.info("\n" + "="*70 + "\n")
