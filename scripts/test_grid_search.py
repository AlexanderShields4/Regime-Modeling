"""
Test grid search with portfolio optimization.
Expected runtime: 5-10 minutes.
"""

import logging
from regime_modeling import setup_logging
from regime_modeling.grid_search.reporting import grid_search_parameters
from regime_modeling.features import cache_exists
from tqdm import tqdm
import sys

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*70)
    logger.info("TEST GRID SEARCH - PORTFOLIO OPTIMIZATION")
    logger.info("="*70)

    if not cache_exists():
        logger.error("\n" + "="*70)
        logger.error("ERROR: No cached data found!")
        logger.error("="*70)
        logger.info("\nRun cache_data.py first:")
        logger.info("    python cache_data.py")
        logger.info("="*70 + "\n")
        sys.exit(1)

    logger.info("\n✓ Cached data found\n")

    n_stocks_range = [7, 10]
    n_indices_range = [0, 3]
    volatility_window_range = [10, 20, 30]
    rsi_period_range = [14]
    momentum_period_range = [10]
    n_iter_range = [3000]
    covariance_type_range = ['full']

    total_configs = (len(n_stocks_range) * len(n_indices_range) *
                    len(volatility_window_range) * len(rsi_period_range) *
                    len(momentum_period_range) * len(n_iter_range) *
                    len(covariance_type_range) * 6)

    logger.info("Test Configuration:")
    logger.info(f"  Total configs: {total_configs}")
    logger.info(f"  n_stocks: {n_stocks_range}")
    logger.info(f"  n_indices: {n_indices_range}")
    logger.info(f"  volatility_window: {volatility_window_range}")
    logger.info(f"  n_iter: {n_iter_range}")
    logger.info(f"  Expected runtime: ~5-10 minutes\n")

    logger.info("="*70)
    logger.info("STARTING TEST SEARCH...")
    logger.info("="*70 + "\n")

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
        top_n=10,
        show_progress=True
    )

    logger.info("\n" + "="*70)
    logger.info("TEST COMPLETE!")
    logger.info("="*70)

    logger.info("\nPortfolio Performance Distribution:")
    logger.info(f"  Sharpe Ratio: min={results['Sharpe'].min():.2f}, max={results['Sharpe'].max():.2f}, mean={results['Sharpe'].mean():.2f}")
    logger.info(f"  CAGR: min={results['CAGR%'].min():.1f}%, max={results['CAGR%'].max():.1f}%, mean={results['CAGR%'].mean():.1f}%")
    logger.info(f"  Max DD: min={results['MaxDD%'].min():.1f}%, max={results['MaxDD%'].max():.1f}%, mean={results['MaxDD%'].mean():.1f}%")

    logger.info("\nVerifying Scoring:")
    top_config = results.iloc[0]
    logger.info(f"  Top config has:")
    logger.info(f"    Score: {top_config['Score']:.2f} (should be lowest)")
    logger.info(f"    Sharpe: {top_config['Sharpe']:.2f}")
    logger.info(f"    CAGR: {top_config['CAGR%']:.1f}%")
    logger.info(f"    Decision: {top_config['Decision']}")

    bottom_config = results.iloc[-1]
    logger.info(f"  Bottom config has:")
    logger.info(f"    Score: {bottom_config['Score']:.2f} (should be highest)")
    logger.info(f"    Sharpe: {bottom_config['Sharpe']:.2f}")
    logger.info(f"    CAGR: {bottom_config['CAGR%']:.1f}%")
    logger.info(f"    Decision: {bottom_config['Decision']}")

    if top_config['Score'] > bottom_config['Score']:
        logger.warning("\n⚠️  WARNING: Scoring might be inverted! Top score should be LOWER than bottom.")
    else:
        logger.info("\n✓ Scoring verified: Lower score = better config")
    excellent = (results['Decision'] == '✅ EXCELLENT').sum()
    good = (results['Decision'] == '✓ GOOD').sum()
    acceptable = (results['Decision'] == '○ ACCEPTABLE').sum()
    poor = (results['Decision'] == '✗ POOR').sum()

    logger.info(f"\nDecision Distribution:")
    logger.info(f"  ✅ EXCELLENT: {excellent}/{len(results)} ({excellent/len(results)*100:.1f}%)")
    logger.info(f"  ✓ GOOD: {good}/{len(results)} ({good/len(results)*100:.1f}%)")
    logger.info(f"  ○ ACCEPTABLE: {acceptable}/{len(results)} ({acceptable/len(results)*100:.1f}%)")
    logger.info(f"  ✗ POOR: {poor}/{len(results)} ({poor/len(results)*100:.1f}%)")

    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("""
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

    logger.info("="*70 + "\n")
