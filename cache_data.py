"""
Cache Data Script

Run this once to download and cache stock data from APIs.
This cached data will be used by all grid searches and model runs,
avoiding repeated API calls and rate limiting.

"""

import argparse
from features import get_merged_data, cache_exists, clear_cache, CACHE_DIR
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Cache stock data for HMM modeling')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh cache (ignore existing)')
    parser.add_argument('--clear', action='store_true',
                       help='Clear existing cache and exit')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("DATA CACHING UTILITY")
    print("="*70)

    # Clear cache if requested
    if args.clear:
        print("\nClearing cache...")
        clear_cache()
        print("\n" + "="*70)
        print("CACHE CLEARED")
        print("="*70 + "\n")
        return

    # Check if cache exists
    if not args.refresh and cache_exists():
        print("\n⚠️  Cache already exists!")
        print(f"    Location: {CACHE_DIR}/")
        response = input("\n    Do you want to refresh? (y/N): ").strip().lower()
        if response != 'y':
            print("\n    Keeping existing cache. Use --refresh to force update.\n")
            return
        args.refresh = True

    # Download and cache data
    print("\nDownloading data from APIs...")
    print("This may take a few minutes...\n")

    start_time = datetime.now()

    try:
        merged_data = get_merged_data(
            join_type='inner',
            use_cache=False if args.refresh else True,
            force_refresh=args.refresh
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        print("\n" + "="*70)
        print("DATA CACHED SUCCESSFULLY")
        print("="*70)
        print(f"  Data shape: {merged_data.shape}")
        print(f"  Date range: {merged_data.index[0]} to {merged_data.index[-1]}")
        print(f"  Cache location: {CACHE_DIR}/")
        print(f"  Time elapsed: {elapsed:.1f} seconds")
        print("\n  ✓ Grid searches will now use cached data (much faster!)")
        print("  ✓ To refresh data in the future, run: python cache_data.py --refresh")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: Failed to cache data")
        print("="*70)
        print(f"  {str(e)}")
        print("\n  Try again later or check your internet connection.")
        print("="*70 + "\n")
        raise


if __name__ == "__main__":
    main()
