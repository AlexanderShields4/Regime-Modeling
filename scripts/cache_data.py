"""
Cache stock data from APIs for offline use in grid searches.
"""

import argparse
import logging
from regime_modeling import setup_logging
from regime_modeling.features import get_merged_data, cache_exists, clear_cache, CACHE_DIR
from datetime import datetime

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description='Cache stock data for HMM modeling')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh cache (ignore existing)')
    parser.add_argument('--clear', action='store_true',
                       help='Clear existing cache and exit')
    args = parser.parse_args()

    logger.info("\n" + "="*70)
    logger.info("DATA CACHING UTILITY")
    logger.info("="*70)

    if args.clear:
        logger.info("\nClearing cache...")
        clear_cache()
        logger.info("\n" + "="*70)
        logger.info("CACHE CLEARED")
        logger.info("="*70 + "\n")
        return

    if not args.refresh and cache_exists():
        logger.warning("\n⚠️  Cache already exists!")
        logger.info(f"    Location: {CACHE_DIR}/")
        response = input("\n    Do you want to refresh? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("\n    Keeping existing cache. Use --refresh to force update.\n")
            return
        args.refresh = True

    logger.info("\nDownloading data from APIs...")
    logger.info("This may take a few minutes...\n")

    start_time = datetime.now()

    try:
        merged_data = get_merged_data(
            join_type='outer',
            use_cache=False if args.refresh else True,
            force_refresh=args.refresh
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*70)
        logger.info("DATA CACHED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"  Data shape: {merged_data.shape}")
        logger.info(f"  Date range: {merged_data.index[0]} to {merged_data.index[-1]}")
        logger.info(f"  Cache location: {CACHE_DIR}/")
        logger.info(f"  Time elapsed: {elapsed:.1f} seconds")
        logger.info("\n  ✓ Grid searches will now use cached data (much faster!)")
        logger.info("  ✓ To refresh data in the future, run: python cache_data.py --refresh")
        logger.info("="*70 + "\n")

    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("ERROR: Failed to cache data")
        logger.error("="*70)
        logger.error(f"  {str(e)}")
        logger.error("\n  Try again later or check your internet connection.")
        logger.error("="*70 + "\n")
        raise


if __name__ == "__main__":
    main()
