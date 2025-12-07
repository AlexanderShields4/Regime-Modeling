"""
TEST GRID SEARCH WITH PORTFOLIO OPTIMIZATION

This script runs a small grid search (10-20 configs) to test the new
portfolio-optimized scoring function before running the full overnight search.

Expected runtime: 5-10 minutes
"""

from model import grid_search_parameters
from features import cache_exists
from tqdm import tqdm
import sys

if __name__ == "__main__":

    print("\n" + "="*70)
    print("TEST GRID SEARCH - PORTFOLIO OPTIMIZATION")
    print("="*70)

    # Check if cache exists
    if not cache_exists():
        print("\n" + "="*70)
        print("ERROR: No cached data found!")
        print("="*70)
        print("\nYou must run cache_data.py first:")
        print("    python cache_data.py")
        print("="*70 + "\n")
        sys.exit(1)

    print("\n✓ Cached data found\n")

    # Small parameter ranges for quick testing
    n_stocks_range = [7, 10]  # 2 values
    n_indices_range = [0, 3]  # 2 values
    volatility_window_range = [10, 20, 30]  # 3 values (matches overnight)
    rsi_period_range = [14]  # 1 value
    momentum_period_range = [10]  # 1 value
    n_iter_range = [3000]  # 1 value (faster convergence)
    covariance_type_range = ['full']  # 1 value

    # Using 'auto' = 6 feature combinations
    # Total configs = 2 * 2 * 2 * 1 * 1 * 1 * 1 * 6 = 48 configs

    total_configs = (len(n_stocks_range) * len(n_indices_range) *
                    len(volatility_window_range) * len(rsi_period_range) *
                    len(momentum_period_range) * len(n_iter_range) *
                    len(covariance_type_range) * 6)

    print("Test Configuration:")
    print(f"  Total configs: {total_configs}")
    print(f"  n_stocks: {n_stocks_range}")
    print(f"  n_indices: {n_indices_range}")
    print(f"  volatility_window: {volatility_window_range}")
    print(f"  n_iter: {n_iter_range}")
    print(f"  Expected runtime: ~5-10 minutes\n")

    print("="*70)
    print("STARTING TEST SEARCH...")
    print("="*70 + "\n")

    # Run test grid search
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
        n_processes=None,  # Use all available cores
        top_n=10,
        show_progress=True
    )

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)

    # Display portfolio performance distribution
    print("\nPortfolio Performance Distribution:")
    print(f"  Sharpe Ratio: min={results['Sharpe'].min():.2f}, max={results['Sharpe'].max():.2f}, mean={results['Sharpe'].mean():.2f}")
    print(f"  CAGR: min={results['CAGR%'].min():.1f}%, max={results['CAGR%'].max():.1f}%, mean={results['CAGR%'].mean():.1f}%")
    print(f"  Max DD: min={results['MaxDD%'].min():.1f}%, max={results['MaxDD%'].max():.1f}%, mean={results['MaxDD%'].mean():.1f}%")

    # Check if scoring is working correctly
    print("\nVerifying Scoring:")
    top_config = results.iloc[0]
    print(f"  Top config has:")
    print(f"    Score: {top_config['Score']:.2f} (should be lowest)")
    print(f"    Sharpe: {top_config['Sharpe']:.2f}")
    print(f"    CAGR: {top_config['CAGR%']:.1f}%")
    print(f"    Decision: {top_config['Decision']}")

    bottom_config = results.iloc[-1]
    print(f"  Bottom config has:")
    print(f"    Score: {bottom_config['Score']:.2f} (should be highest)")
    print(f"    Sharpe: {bottom_config['Sharpe']:.2f}")
    print(f"    CAGR: {bottom_config['CAGR%']:.1f}%")
    print(f"    Decision: {bottom_config['Decision']}")

    # Sanity check
    if top_config['Score'] > bottom_config['Score']:
        print("\n⚠️  WARNING: Scoring might be inverted! Top score should be LOWER than bottom.")
    else:
        print("\n✓ Scoring verified: Lower score = better config")

    # Check for configs ready for trading
    excellent = (results['Decision'] == '✅ EXCELLENT').sum()
    good = (results['Decision'] == '✓ GOOD').sum()
    acceptable = (results['Decision'] == '○ ACCEPTABLE').sum()
    poor = (results['Decision'] == '✗ POOR').sum()

    print(f"\nDecision Distribution:")
    print(f"  ✅ EXCELLENT: {excellent}/{len(results)} ({excellent/len(results)*100:.1f}%)")
    print(f"  ✓ GOOD: {good}/{len(results)} ({good/len(results)*100:.1f}%)")
    print(f"  ○ ACCEPTABLE: {acceptable}/{len(results)} ({acceptable/len(results)*100:.1f}%)")
    print(f"  ✗ POOR: {poor}/{len(results)} ({poor/len(results)*100:.1f}%)")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Review the test results:
   - Check grid_search_results_*.csv
   - Verify portfolio metrics are populated
   - Confirm scoring makes sense

2. If results look good, run full overnight search:
   - python overnight_grid_search.py

3. If issues found:
   - Check error messages
   - Verify portfolio backtest is working
   - Adjust scoring weights if needed
""")

    print("="*70 + "\n")
