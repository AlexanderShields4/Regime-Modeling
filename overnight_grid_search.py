"""
COMPREHENSIVE OVERNIGHT GRID SEARCH

This script runs an exhaustive search across many parameter combinations.
Expected runtime: 1-4 hours depending on your CPU cores.

IMPORTANT: Run cache_data.py FIRST to avoid API rate limiting!
    python cache_data.py

Then run this script overnight:
    python overnight_grid_search.py

Total configurations: ~900 (customizable below)
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

    # Check if cache exists
    if not cache_exists():
        print("\n" + "="*70)
        print("ERROR: No cached data found!")
        print("="*70)
        print("\nYou must run cache_data.py first to download and cache data:")
        print("    python cache_data.py")
        print("\nThis ensures fast grid search without API rate limiting.")
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

    # Define comprehensive parameter ranges
    # Expanded ranges for more thorough exploration
    
    # Asset selection - more granular options
    n_stocks_range = [5, 7, 10, 12, 15, 18, 20]
    n_indices_range = [0, 2, 3, 5, 7]
    
    # Feature engineering windows - reduced for efficiency
    volatility_window_range = [10, 20, 30]  # Reduced from 6 to 3 values
    rsi_period_range = [14]  # Reduced to standard period only
    momentum_period_range = [10]  # Reduced to standard period only
    
    # HMM training parameters
    n_iter_range = [3000, 5000, 7000]  # More iterations for better convergence
    covariance_type_range = ['full', 'tied']  # Test different covariance structures

    # Calculate total configurations
    # Using 'auto' feature combinations = 6 combos
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

    # Estimate time
    avg_time_per_config = 15  # seconds (conservative estimate)
    total_seconds = (total_configs * avg_time_per_config) / (n_cores // 2)  # Updated for half cores
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

    # Run comprehensive grid search
    results = grid_search_parameters(
        n_stocks_range=n_stocks_range,
        n_indices_range=n_indices_range,
        volatility_window_range=volatility_window_range,
        rsi_period_range=rsi_period_range,
        momentum_period_range=momentum_period_range,
        n_iter_range=n_iter_range,
        covariance_type_range=covariance_type_range,
        feature_combinations='auto',     # 6 smart feature combinations
        train_ratio=0.8,
        n_processes=None,                # Use all available cores minus 1
        top_n=20,                        # Show top 20 results
        show_progress=True               # Use tqdm progress bar
    )

    # Calculate elapsed time
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
    print("   → Ready-to-use Python code for best configuration")
    print("   → Just copy and run!")


    print("="*70)
    print("Grid search complete! Check the files above.")
    print("="*70 + "\n")

    # Display summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nBest Score: {results['Score'].min():.2f}")
    print(f"Worst Score: {results['Score'].max():.2f}")
    print(f"Average Score: {results['Score'].mean():.2f}")
    print(f"\nBest Degradation: {results['Degradation%'].min():.1f}%")
    print(f"Worst Degradation: {results['Degradation%'].max():.1f}%")
    print(f"Average Degradation: {results['Degradation%'].mean():.1f}%")

    # Count how many are ready for portfolio
    ready_count = results['Decision'].str.contains('✅').sum()
    print(f"\nConfigurations ready for portfolio: {ready_count}/{len(results)} ({ready_count/len(results)*100:.1f}%)")

    print("\n" + "="*70 + "\n")
