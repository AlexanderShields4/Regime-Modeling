import pandas as pd
import numpy as np
from pages_utils import fetch_all_data
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle
import os
from datetime import datetime
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
            'shape': merged_df.shape
        }, f)

    # Also save as CSV for human inspection
    csv_path = os.path.join(cache_dir, 'raw_merged_data.csv')
    merged_df.to_csv(csv_path, index=True)

    return cache_path


def load_raw_data_cache(cache_path=RAW_DATA_CACHE):
    # Return None if cache doesn't exist
    if not os.path.exists(cache_path):
        return None

    # Load and return the cached DataFrame
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    return cache_data['data']


def cache_exists(cache_path=RAW_DATA_CACHE):
    # Check if cache file exists
    return os.path.exists(cache_path)


def clear_cache(cache_dir=CACHE_DIR):
    # Remove entire cache directory if it exists
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)


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

    # Fill moving average columns with backward then forward fill
    ma_cols = [col for col in df_clean.columns if '_ma_' in col]
    if ma_cols:
        df_clean[ma_cols] = df_clean[ma_cols].fillna(method='bfill').fillna(method='ffill')

    # Fill price columns with forward then backward fill (prefer recent data)
    price_cols = [col for col in df_clean.columns if '_ma_' not in col]
    if price_cols:
        df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill').fillna(method='bfill')

    # Interpolate any remaining gaps linearly
    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.interpolate(method='linear', axis=0)

    # Drop any rows still containing NaN values
    df_clean = df_clean.dropna(axis=0)

    return df_clean


def get_merged_data(join_type='inner', use_cache=True, force_refresh=False):
    # Try to load from cache if enabled and not forcing refresh
    if use_cache and not force_refresh:
        cached_data = load_raw_data_cache()
        if cached_data is not None:
            return cached_data

    # Fetch all raw data from data sources
    stocks_df, indices_df, resources_df, stock_moving_averages_df, indice_moving_averages_df, resource_moving_averages_df = fetch_all_data()

    # Merge all dataframes with prefixes
    merged_df = merge_all_dataframes(stocks_df, indices_df, resources_df,
                                     stock_moving_averages_df, indice_moving_averages_df,
                                     resource_moving_averages_df, join_type=join_type)

    # Clean missing values
    merged_df = handle_missing_values(merged_df)

    # Save merged data to CSV for inspection
    merged_df.to_csv('merged_data.csv', index=True)

    # Save to cache for faster future loads
    if use_cache:
        save_raw_data_cache(merged_df)

    return merged_df


def calculate_returns(df, method='log'):
    # Calculate log returns (more suitable for HMM) or simple percentage returns
    if method == 'log':
        returns = np.log(df / df.shift(1))
    else:
        returns = df.pct_change()

    # Remove first row which will be NaN due to shift
    return returns.dropna()


def calculate_volatility(returns_df, window=20):
    # Rolling standard deviation of returns as volatility measure
    return returns_df.rolling(window=window).std()


def calculate_rsi(prices, period=14):
    # Relative Strength Index: momentum oscillator (0-100)
    # Measures speed and magnitude of price changes
    delta = prices.diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculate relative strength and RSI
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_momentum(prices, period=10):
    # Rate of change over specified period
    return prices.pct_change(periods=period)


def calculate_market_breadth(returns_df):
    # Aggregate market health indicators
    return pd.DataFrame({
        'pct_positive': (returns_df > 0).sum(axis=1) / returns_df.shape[1],  # % of assets with positive returns
        'avg_return': returns_df.mean(axis=1),  # Average return across all assets
        'cross_sectional_vol': returns_df.std(axis=1)  # Dispersion of returns
    })


def select_key_features(df, n_stocks=10, n_indices=5):
    selected_cols = []

    # Select key market indices (S&P 500, NASDAQ, Dow Jones, Russell 2000, VIX)
    key_indices = [col for col in df.columns if any(idx in col for idx in ['GSPC', 'IXIC', 'DJI', 'RUT', 'VIX'])]
    selected_cols.extend(key_indices[:n_indices])

    # Select priority stocks from major sectors (tech, finance, energy, healthcare, retail)
    stock_cols = [col for col in df.columns if col.startswith('stock_') and '_ma_' not in col]
    priority_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'XOM', 'JNJ', 'WMT']
    selected_stocks = [col for col in stock_cols if any(stock in col for stock in priority_stocks)][:n_stocks]
    selected_cols.extend(selected_stocks)

    # Select commodity/resource columns (gold, oil, etc.)
    resource_cols = [col for col in df.columns if col.startswith('resource_') and '_ma_' not in col]
    selected_cols.extend(resource_cols[:5])

    return df[selected_cols]


def create_enhanced_features_custom(df, include_returns=True, include_volatility=True,
                                   include_rsi=True, include_momentum=True,
                                   include_market_breadth=True,
                                   volatility_window=20, rsi_period=14, momentum_period=10):
    # Build feature set based on flags (allows flexible feature engineering)
    features_list = []

    # Calculate log returns (stationary and suitable for HMM)
    returns = calculate_returns(df, method='log')

    # Add returns if requested
    if include_returns:
        features_list.append(returns)

    # Add rolling volatility for key assets
    if include_volatility:
        volatility = calculate_volatility(returns, window=volatility_window)
        volatility.columns = [f"{col}_vol" for col in volatility.columns]
        # Select subset to avoid too many features
        key_vol_cols = [col for col in volatility.columns if any(x in col for x in ['GSPC', 'IXIC', 'AAPL', 'MSFT', 'VIX'])][:10]
        features_list.append(volatility[key_vol_cols])

    # Add market breadth indicators (cross-sectional stats)
    if include_market_breadth:
        features_list.append(calculate_market_breadth(returns))

    # Add technical indicators (RSI and momentum) for key price columns
    if include_rsi or include_momentum:
        key_price_cols = [col for col in df.columns if any(x in col for x in ['GSPC', 'IXIC', 'AAPL', 'MSFT', 'VIX'])][:10]

        if include_rsi:
            # Calculate RSI for each key asset
            rsi_features = [calculate_rsi(df[col], period=rsi_period).rename(f"{col}_rsi")
                           for col in key_price_cols if col in df.columns]
            if rsi_features:
                features_list.append(pd.concat(rsi_features, axis=1))

        if include_momentum:
            # Calculate momentum for each key asset
            momentum_features = [calculate_momentum(df[col], period=momentum_period).rename(f"{col}_momentum")
                                for col in key_price_cols if col in df.columns]
            if momentum_features:
                features_list.append(pd.concat(momentum_features, axis=1))

    # Combine all features and drop rows with any NaN values
    return pd.concat(features_list, axis=1).dropna()




def apply_pca(features_df, n_components=20, variance_threshold=0.95):
    # Normalize features before PCA to ensure all features contribute equally
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    # Choose PCA mode: auto-select components by variance threshold or use fixed number
    if n_components is None:
        pca = PCA(n_components=variance_threshold, svd_solver='full')
    else:
        # Ensure we don't request more components than available features
        pca = PCA(n_components=min(n_components, features_df.shape[1]))

    # Transform features to principal components
    features_pca = pca.fit_transform(features_scaled)

    # Convert to DataFrame with descriptive column names (PC1, PC2, etc.)
    pca_df = pd.DataFrame(
        features_pca,
        index=features_df.index,
        columns=[f'PC{i+1}' for i in range(features_pca.shape[1])]
    )

    # Return PCA features, model, scaler, and variance explained by each component
    return pca_df, pca, scaler, pca.explained_variance_ratio_


def normalize_features(features_df, method='standard'):
    # Select appropriate scaler based on method
    # standard: zero mean, unit variance (sensitive to outliers)
    # robust: uses median and IQR (resistant to outliers)
    # minmax: scales to [0,1] range (sensitive to outliers)
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard', 'robust', or 'minmax'")

    # Fit scaler and transform features
    features_normalized = scaler.fit_transform(features_df)

    # Convert back to DataFrame to preserve index and column names
    normalized_df = pd.DataFrame(
        features_normalized,
        index=features_df.index,
        columns=features_df.columns
    )

    # Return normalized features and fitted scaler for future transforms
    return normalized_df, scaler


def prepare_hmm_features(join_type='inner', use_pca=True, n_components=20,
                         scaling_method='robust', select_features=True,
                         n_stocks=15, n_indices=5,
                         volatility_window=20, rsi_period=14, momentum_period=10,
                         save_to_csv=True, csv_filename='enhanced_features.csv'):
    # Load and merge all market data (stocks, indices, resources, moving averages)
    raw_data = get_merged_data(join_type=join_type)

    # Optionally reduce feature set to key assets to reduce noise and computation
    if select_features:
        raw_data = select_key_features(raw_data, n_stocks=n_stocks, n_indices=n_indices)

    # Engineer features: returns, volatility, RSI, momentum, market breadth
    features = create_enhanced_features_custom(
        raw_data,
        include_returns=True,
        include_volatility=True,
        include_rsi=True,
        include_momentum=True,
        include_market_breadth=True,
        volatility_window=volatility_window,
        rsi_period=rsi_period,
        momentum_period=momentum_period
    )

    # Save enhanced features before dimensionality reduction for later analysis
    if save_to_csv:
        features.to_csv(csv_filename, index=True)

    # Optionally reduce dimensionality with PCA to capture key variance
    pca_model = None
    pca_scaler = None
    if use_pca:
        features, pca_model, pca_scaler, _ = apply_pca(
            features,
            n_components=n_components
        )

    # Final normalization to standardize feature scales for HMM
    features_normalized, final_scaler = normalize_features(features, method=scaling_method)

    # Return normalized features, scaler, and PCA model (if used)
    return features_normalized, final_scaler, pca_model, pca_scaler


def get_enhanced_features_for_model(join_type='inner', n_stocks=15, n_indices=5,
                                   volatility_window=20, rsi_period=14, momentum_period=10,
                                   save_to_csv=True, csv_filename='merged_data_with_features.csv',
                                   include_returns=True, include_volatility=True,
                                   include_rsi=True, include_momentum=True,
                                   include_market_breadth=True):
    # Load and merge all market data
    raw_data = get_merged_data(join_type=join_type)

    # Reduce to key features to minimize noise
    raw_data_selected = select_key_features(raw_data, n_stocks=n_stocks, n_indices=n_indices)

    # Create enhanced features with configurable feature types
    # This provides raw features without normalization or PCA for direct analysis
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

    # Optionally save enhanced features to CSV for inspection
    if save_to_csv:
        enhanced_features.to_csv(csv_filename, index=True)

    return enhanced_features

