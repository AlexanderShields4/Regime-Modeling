"""
Comprehensive overnight grid search for optimal HMM parameters.
Expected runtime: 1-4 hours.

Run cache_data.py first to avoid API rate limiting.
"""

from model import grid_search_parameters
from features import cache_exists
import multiprocessing as mp
from datetime import datetime, timedelta
import sys
from tqdm import tqdm

if __name__ == "__main__":

    print("\n" + "="*70)
    print("COMPREHENSIVE OVERNIGHT GRID SEARCH")
    print("="*70)

    if not cache_exists():
        print("\n" + "="*70)
        print("ERROR: No cached data found!")
        print("="*70)
        print("\nRun cache_data.py first:")
        print("    python cache_data.py")
        print("="*70 + "\n")
        sys.exit(1)

    print("\n✓ Cached data found - grid search will be fast!\n")

    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Display system info
    n_cores = mp.cpu_count()
    print(f"Available CPU cores: {n_cores}")
    print(f"Using {n_cores // 2} cores (half of available cores)")

    print("\n" + "-"*70)
    print("PARAMETER RANGES")
    print("-"*70)

    n_stocks_range = [5, 7, 10, 12, 15, 18, 20]
    n_indices_range = [0, 2, 3, 5, 7]

    volatility_window_range = [10, 20, 30]
    rsi_period_range = [14]
    momentum_period_range = [10]

    n_iter_range = [3000, 5000, 7000]
    covariance_type_range = ['full', 'tied']
    total_configs = (len(n_stocks_range) * len(n_indices_range) *
                    len(volatility_window_range) * len(rsi_period_range) *
                    len(momentum_period_range) * len(n_iter_range) *
                    len(covariance_type_range) * 6)

    print(f"\nn_stocks: {n_stocks_range}")
    print(f"n_indices: {n_indices_range}")
    print(f"volatility_window: {volatility_window_range}")
    print(f"rsi_period: {rsi_period_range}")
    print(f"momentum_period: {momentum_period_range}")
    print(f"n_iter: {n_iter_range}")
    print(f"covariance_type: {covariance_type_range}")
    print(f"feature_combinations: 6 (auto-selected)")

    print(f"\nTotal configurations to test: {total_configs:,}")

    avg_time_per_config = 15
    total_seconds = (total_configs * avg_time_per_config) / (n_cores // 2)
    hours = total_seconds / 3600

    expected_completion_low = datetime.now() + timedelta(seconds=total_seconds)
    expected_completion_high = datetime.now() + timedelta(seconds=total_seconds * 2)

    print(f"\nEstimated time: {hours:.1f} - {hours*2:.1f} hours")
    print(f"Expected completion: {expected_completion_low.strftime('%Y-%m-%d %H:%M:%S')} - {expected_completion_high.strftime('%H:%M:%S')}")

    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE SEARCH...")
    print("="*70)
    print("\nYou can safely minimize this window.")
    print("Results will be saved automatically when complete.\n")

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

    print("\n" + "="*70)
    print("COMPREHENSIVE SEARCH COMPLETE!")
    print("="*70)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed: {hours_elapsed:.2f} hours ({elapsed.total_seconds()/60:.1f} minutes)")
    print(f"\nConfigurations tested: {len(results)}/{total_configs}")
    print(f"Success rate: {len(results)/total_configs*100:.1f}%")

    print("\n" + "="*70)
    print("GENERATED FILES")
    print("="*70)
    print("\n1. grid_search_results_*.csv")
    print("   → Full results table with all configurations")
    print("   → Sorted by score (lower = better)")
    print("\n2. best_config.txt")
    print("   → Ready-to-use code for best configuration")

    print("="*70)
    print("Grid search complete! Check the files above.")
    print("="*70 + "\n")

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nBest Score: {results['Score'].min():.2f}")
    print(f"Worst Score: {results['Score'].max():.2f}")
    print(f"Average Score: {results['Score'].mean():.2f}")
    print(f"\nBest Degradation: {results['Degradation%'].min():.1f}%")
    print(f"Worst Degradation: {results['Degradation%'].max():.1f}%")
    print(f"Average Degradation: {results['Degradation%'].mean():.1f}%")

    ready_count = results['Decision'].str.contains('✅').sum()
    print(f"\nConfigurations ready for portfolio: {ready_count}/{len(results)} ({ready_count/len(results)*100:.1f}%)")

    print("\n" + "="*70 + "\n")
