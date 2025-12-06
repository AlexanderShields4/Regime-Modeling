import pandas as pd
import numpy as np
from pages_utils import fetch_all_data
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = 'data_cache'
RAW_DATA_CACHE = os.path.join(CACHE_DIR, 'raw_merged_data.pkl')
RAW_DATA_CSV = os.path.join(CACHE_DIR, 'raw_merged_data.csv')


def save_raw_data_cache(merged_df, cache_dir=CACHE_DIR):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Save data as pickle with metadata for fast loading
    cache_path = os.path.join(cache_dir, 'raw_merged_data.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'data': merged_df,
            'timestamp': datetime.now(),
            'shape': merged_df.shape,
            'date_range': {
                'start': merged_df.index.min(),
                'end': merged_df.index.max()
            }
        }, f)

    # Also save as CSV for human inspection
    csv_path = os.path.join(cache_dir, 'raw_merged_data.csv')
    merged_df.to_csv(csv_path, index=True)

    print(f"✓ Cache saved: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    print(f"  Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    print(f"  Cache timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return cache_path


def load_raw_data_cache(cache_path=RAW_DATA_CACHE, max_age_hours=24):
    """
    Load cached data if it exists and is not too old.
    
    Args:
        cache_path: Path to cache file
        max_age_hours: Maximum age of cache in hours before it's considered stale
                       Set to None to disable expiry check
    
    Returns:
        DataFrame if cache is valid, None otherwise
    """
    # Return None if cache doesn't exist
    if not os.path.exists(cache_path):
        print("ℹ No cache found")
        return None

    # Load and check cache age
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    cache_timestamp = cache_data.get('timestamp', datetime.min)
    cache_age = datetime.now() - cache_timestamp
    
    print(f"ℹ Cache found:")
    print(f"  Created: {cache_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Age: {cache_age.total_seconds() / 3600:.1f} hours")
    print(f"  Shape: {cache_data['shape']}")
    if 'date_range' in cache_data:
        print(f"  Date range: {cache_data['date_range']['start']} to {cache_data['date_range']['end']}")
    
    # Check if cache is too old (if expiry is enabled)
    if max_age_hours is not None and cache_age > timedelta(hours=max_age_hours):
        print(f"⚠ Cache is stale (older than {max_age_hours} hours)")
        return None
    
    print("✓ Using cached data")
    return cache_data['data']


def cache_exists(cache_path=RAW_DATA_CACHE):
    # Check if cache file exists
    return os.path.exists(cache_path)


def clear_cache(cache_dir=CACHE_DIR):
    # Remove entire cache directory if it exists
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✓ Cache cleared: {cache_dir}")


def get_merged_data(join_type='inner', use_cache=True, force_refresh=False, 
                    cache_max_age_hours=24):
    """
    Get merged market data with smart caching.
    
    Args:
        join_type: How to merge dataframes ('inner' or 'outer')
        use_cache: Whether to use cached data if available
        force_refresh: If True, bypass cache and fetch fresh data
        cache_max_age_hours: Maximum cache age in hours (None = never expire)
    
    Returns:
        DataFrame with merged market data
    """
    print("\n" + "="*60)
    print("LOADING MARKET DATA")
    print("="*60)
    
    # Force refresh: clear cache and fetch new data
    if force_refresh:
        print("🔄 Force refresh enabled - fetching fresh data...")
        clear_cache()
        use_cache = False
    
    # Try to load from cache if enabled
    cached_data = None
    if use_cache:
        cached_data = load_raw_data_cache(max_age_hours=cache_max_age_hours)
    
    # Return cached data if valid
    if cached_data is not None:
        print("="*60 + "\n")
        return cached_data
    
    # Fetch fresh data if no valid cache
    print("\n📡 Fetching fresh data from sources...")
    stocks_df, indices_df, resources_df, stock_moving_averages_df, \
        indice_moving_averages_df, resource_moving_averages_df = fetch_all_data()

    # Merge all dataframes with prefixes
    print("🔗 Merging dataframes...")
    merged_df = merge_all_dataframes(
        stocks_df, indices_df, resources_df,
        stock_moving_averages_df, indice_moving_averages_df,
        resource_moving_averages_df, join_type=join_type
    )

    # Clean missing values
    print("🧹 Cleaning missing values...")
    merged_df = handle_missing_values(merged_df)

    # Save merged data to CSV for inspection
    merged_df.to_csv('merged_data.csv', index=True)
    print("✓ Saved to merged_data.csv")

    # Save to cache for faster future loads
    if use_cache:
        print("\n💾 Saving to cache...")
        save_raw_data_cache(merged_df)

    print("="*60 + "\n")
    return merged_df


def merge_all_dataframes(stocks_df, indices_df, resources_df,
                         stock_moving_averages_df, indice_moving_averages_df,
                         resource_moving_averages_df, join_type='outer'):
    # Organize all dataframes with descriptive keys
    df_dict = {
        'stock': stocks_df,
        'index': indices_df,
        'resource': resources_df,
        'stock_ma': stock_moving_averages_df,
        'index_ma': indice_moving_averages_df,
        'resource_ma': resource_moving_averages_df
    }

    # Add prefixes to column names to identify data source
    prefixed_dfs = [df.add_prefix(f'{prefix}_') for prefix, df in df_dict.items()]

    # Concatenate all dataframes horizontally with specified join type
    combined_df = pd.concat(prefixed_dfs, axis=1, join=join_type)

    # Sort by date index for chronological order
    combined_df.sort_index(inplace=True)

    return combined_df


def handle_missing_values(df):
    df_clean = df.copy()

    # Remove columns with more than 50% missing data
    nan_fraction = df_clean.isnull().sum() / len(df_clean)
    cols_to_keep = nan_fraction[nan_fraction <= 0.5].index
    df_clean = df_clean[cols_to_keep]

    # Fill missing values using forward fill only (no backward fill to avoid data leakage)
    # Forward fill uses past data, which is safe for time series
    ma_cols = [col for col in df_clean.columns if '_ma_' in col]
    if ma_cols:
        df_clean[ma_cols] = df_clean[ma_cols].fillna(method='ffill')

    # Fill price columns with forward fill only
    price_cols = [col for col in df_clean.columns if '_ma_' not in col]
    if price_cols:
        df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill')

    # Drop any rows still containing NaN values (safer than interpolation which uses future data)
    df_clean = df_clean.dropna(axis=0)

    return df_clean


# ... rest of your functions remain the same ...

def calculate_returns(df, method='log'):
    if method == 'log':
        returns = np.log(df / df.shift(1))
    else:
        returns = df.pct_change()
    return returns.dropna()


def calculate_volatility(returns_df, window=20):
    return returns_df.rolling(window=window).std()


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_momentum(prices, period=10):
    return prices.pct_change(periods=period)


def calculate_market_breadth(returns_df):
    return pd.DataFrame({
        'pct_positive': (returns_df > 0).sum(axis=1) / returns_df.shape[1],
        'avg_return': returns_df.mean(axis=1),
        'cross_sectional_vol': returns_df.std(axis=1)
    })


def select_key_features(df, n_stocks=10, n_indices=5):
    selected_cols = []
    key_indices = [col for col in df.columns if any(idx in col for idx in ['GSPC', 'IXIC', 'DJI', 'RUT', 'VIX'])]
    selected_cols.extend(key_indices[:n_indices])
    
    stock_cols = [col for col in df.columns if col.startswith('stock_') and '_ma_' not in col]
    priority_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'XOM', 'JNJ', 'WMT']
    selected_stocks = [col for col in stock_cols if any(stock in col for stock in priority_stocks)][:n_stocks]
    selected_cols.extend(selected_stocks)
    
    resource_cols = [col for col in df.columns if col.startswith('resource_') and '_ma_' not in col]
    selected_cols.extend(resource_cols[:5])
    
    return df[selected_cols]


def create_enhanced_features_custom(df, include_returns=True, include_volatility=True,
                                   include_rsi=True, include_momentum=True,
                                   include_market_breadth=True,
                                   volatility_window=20, rsi_period=14, momentum_period=10):
    features_list = []
    returns = calculate_returns(df, method='log')
    
    if include_returns:
        features_list.append(returns)
    
    if include_volatility:
        volatility = calculate_volatility(returns, window=volatility_window)
        volatility.columns = [f"{col}_vol" for col in volatility.columns]
        key_vol_cols = [col for col in volatility.columns if any(x in col for x in ['GSPC', 'IXIC', 'AAPL', 'MSFT', 'VIX'])][:10]
        features_list.append(volatility[key_vol_cols])
    
    if include_market_breadth:
        features_list.append(calculate_market_breadth(returns))
    
    if include_rsi or include_momentum:
        key_price_cols = [col for col in df.columns if any(x in col for x in ['GSPC', 'IXIC', 'AAPL', 'MSFT', 'VIX'])][:10]
        
        if include_rsi:
            rsi_features = [calculate_rsi(df[col], period=rsi_period).rename(f"{col}_rsi")
                           for col in key_price_cols if col in df.columns]
            if rsi_features:
                features_list.append(pd.concat(rsi_features, axis=1))
        
        if include_momentum:
            momentum_features = [calculate_momentum(df[col], period=momentum_period).rename(f"{col}_momentum")
                                for col in key_price_cols if col in df.columns]
            if momentum_features:
                features_list.append(pd.concat(momentum_features, axis=1))
    
    return pd.concat(features_list, axis=1).dropna()


def get_enhanced_features_for_model(join_type='inner', n_stocks=15, n_indices=5,
                                   volatility_window=20, rsi_period=14, momentum_period=10,
                                   save_to_csv=True, csv_filename='merged_data_with_features.csv',
                                   include_returns=True, include_volatility=True,
                                   include_rsi=True, include_momentum=True,
                                   include_market_breadth=True,
                                   force_refresh=False, cache_max_age_hours=24):
    """
    Get enhanced features with smart caching.
    
    Args:
        force_refresh: If True, fetch fresh data
        cache_max_age_hours: Maximum cache age (24 = daily refresh, None = never expire)
    """
    # Load with cache control
    raw_data = get_merged_data(
        join_type=join_type, 
        force_refresh=force_refresh,
        cache_max_age_hours=cache_max_age_hours
    )
    
    raw_data_selected = select_key_features(raw_data, n_stocks=n_stocks, n_indices=n_indices)
    
    enhanced_features = create_enhanced_features_custom(
        raw_data_selected,
        include_returns=include_returns,
        include_volatility=include_volatility,
        include_rsi=include_rsi,
        include_momentum=include_momentum,
        include_market_breadth=include_market_breadth,
        volatility_window=volatility_window,
        rsi_period=rsi_period,
        momentum_period=momentum_period
    )
    
    if save_to_csv:
        enhanced_features.to_csv(csv_filename, index=True)
    
    return enhanced_features