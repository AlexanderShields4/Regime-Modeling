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
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'raw_merged_data.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'data': merged_df,
            'timestamp': datetime.now(),
            'shape': merged_df.shape
        }, f)
    csv_path = os.path.join(cache_dir, 'raw_merged_data.csv')
    merged_df.to_csv(csv_path, index=True)
    return cache_path


def load_raw_data_cache(cache_path=RAW_DATA_CACHE):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    return cache_data['data']


def cache_exists(cache_path=RAW_DATA_CACHE):
    return os.path.exists(cache_path)


def clear_cache(cache_dir=CACHE_DIR):
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)


def merge_all_dataframes(stocks_df, indices_df, resources_df,
                         stock_moving_averages_df, indice_moving_averages_df,
                         resource_moving_averages_df, join_type='outer'):
    df_dict = {
        'stock': stocks_df,
        'index': indices_df,
        'resource': resources_df,
        'stock_ma': stock_moving_averages_df,
        'index_ma': indice_moving_averages_df,
        'resource_ma': resource_moving_averages_df
    }
    prefixed_dfs = [df.add_prefix(f'{prefix}_') for prefix, df in df_dict.items()]
    combined_df = pd.concat(prefixed_dfs, axis=1, join=join_type)
    combined_df.sort_index(inplace=True)
    return combined_df


def handle_missing_values(df):
    df_clean = df.copy()
    nan_fraction = df_clean.isnull().sum() / len(df_clean)
    cols_to_keep = nan_fraction[nan_fraction <= 0.5].index
    df_clean = df_clean[cols_to_keep]

    ma_cols = [col for col in df_clean.columns if '_ma_' in col]
    if ma_cols:
        df_clean[ma_cols] = df_clean[ma_cols].fillna(method='bfill').fillna(method='ffill')

    price_cols = [col for col in df_clean.columns if '_ma_' not in col]
    if price_cols:
        df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill').fillna(method='bfill')

    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.interpolate(method='linear', axis=0)

    df_clean = df_clean.dropna(axis=0)
    return df_clean


def get_merged_data(join_type='inner', use_cache=True, force_refresh=False):
    if use_cache and not force_refresh:
        cached_data = load_raw_data_cache()
        if cached_data is not None:
            return cached_data

    stocks_df, indices_df, resources_df, stock_moving_averages_df, indice_moving_averages_df, resource_moving_averages_df = fetch_all_data()
    merged_df = merge_all_dataframes(stocks_df, indices_df, resources_df,
                                     stock_moving_averages_df, indice_moving_averages_df,
                                     resource_moving_averages_df, join_type=join_type)
    merged_df = handle_missing_values(merged_df)
    merged_df.to_csv('merged_data.csv', index=True)

    if use_cache:
        save_raw_data_cache(merged_df)

    return merged_df


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


def create_enhanced_features(df, include_technical_indicators=True,
                             volatility_window=20, rsi_period=14,
                             momentum_period=10):
    """
    Create comprehensive feature set for HMM model.

    Args:
        df: Raw price DataFrame
        include_technical_indicators: Whether to include RSI, momentum, etc.
        volatility_window: Window for volatility calculation
        rsi_period: Period for RSI calculation
        momentum_period: Period for momentum calculation

    Returns:
        Enhanced feature DataFrame
    """
    print("Creating enhanced features...")

    # 1. Calculate returns (stationary data)
    print("  - Calculating returns...")
    returns = calculate_returns(df, method='log')

    # 2. Calculate volatility
    print("  - Calculating volatility...")
    volatility = calculate_volatility(returns, window=volatility_window)
    volatility.columns = [f"{col}_vol" for col in volatility.columns]

    # 3. Market breadth indicators
    print("  - Calculating market breadth...")
    breadth = calculate_market_breadth(returns)

    # Combine features
    features = returns.copy()

    if include_technical_indicators:
        print("  - Calculating technical indicators...")

        # Select key price columns for technical indicators
        key_price_cols = [col for col in df.columns if any(x in col for x in ['GSPC', 'IXIC', 'AAPL', 'MSFT', 'VIX'])][:10]

        # Calculate RSI for key assets
        for col in key_price_cols:
            if col in df.columns:
                rsi = calculate_rsi(df[col], period=rsi_period)
                features[f"{col}_rsi"] = rsi

        # Calculate momentum for key assets
        for col in key_price_cols:
            if col in df.columns:
                momentum = calculate_momentum(df[col], period=momentum_period)
                features[f"{col}_momentum"] = momentum

    # Add volatility features (select subset to avoid too many features)
    key_vol_cols = [col for col in volatility.columns if any(x in col for x in ['GSPC', 'IXIC', 'AAPL', 'MSFT', 'VIX'])][:10]
    features = pd.concat([features, volatility[key_vol_cols]], axis=1)

    # Add market breadth
    features = pd.concat([features, breadth], axis=1)

    # Drop any remaining NaN (from indicator calculations)
    features = features.dropna()

    print(f"  - Final feature shape: {features.shape}")
    print(f"  - Features created: returns, volatility, breadth, RSI, momentum")

    return features


def apply_pca(features_df, n_components=20, variance_threshold=0.95):
    """
    Apply PCA for dimensionality reduction.

    Args:
        features_df: DataFrame of features
        n_components: Number of components (if int) or None to auto-select
        variance_threshold: Cumulative variance to retain (if n_components is None)

    Returns:
        Tuple of (transformed_data, pca_model, explained_variance_ratio)
    """
    print(f"Applying PCA...")
    print(f"  - Input shape: {features_df.shape}")

    # First normalize the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    # Apply PCA
    if n_components is None:
        # Auto-select components based on variance threshold
        pca = PCA(n_components=variance_threshold, svd_solver='full')
    else:
        pca = PCA(n_components=min(n_components, features_df.shape[1]))

    features_pca = pca.fit_transform(features_scaled)

    print(f"  - Output shape: {features_pca.shape}")
    print(f"  - Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  - Number of components: {pca.n_components_}")

    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(
        features_pca,
        index=features_df.index,
        columns=[f'PC{i+1}' for i in range(features_pca.shape[1])]
    )

    return pca_df, pca, scaler, pca.explained_variance_ratio_


def normalize_features(features_df, method='standard'):
    """
    Normalize features using specified method.

    Args:
        features_df: DataFrame of features
        method: 'standard', 'robust', or 'minmax'

    Returns:
        Tuple of (normalized_data, scaler)
    """
    print(f"Normalizing features using {method} scaling...")

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard', 'robust', or 'minmax'")

    features_normalized = scaler.fit_transform(features_df)

    # Create DataFrame
    normalized_df = pd.DataFrame(
        features_normalized,
        index=features_df.index,
        columns=features_df.columns
    )

    return normalized_df, scaler


def prepare_hmm_features(join_type='inner', use_pca=True, n_components=20,
                         scaling_method='robust', select_features=True,
                         n_stocks=15, n_indices=5,
                         volatility_window=20, rsi_period=14, momentum_period=10,
                         save_to_csv=True, csv_filename='enhanced_features.csv'):
    """
    Complete pipeline to prepare features for HMM model.

    Args:
        join_type: 'inner' or 'outer' for data merging
        use_pca: Whether to apply PCA dimensionality reduction
        n_components: Number of PCA components (if use_pca=True)
        scaling_method: 'standard', 'robust', or 'minmax'
        select_features: Whether to pre-select key features before processing
        n_stocks: Number of stocks to select (if select_features=True)
        n_indices: Number of indices to select (if select_features=True)
        volatility_window: Window for volatility calculation
        rsi_period: Period for RSI calculation
        momentum_period: Period for momentum calculation
        save_to_csv: Whether to save features to CSV
        csv_filename: Name of CSV file to save features

    Returns:
        Tuple of (features_df, scaler, pca_model)
    """
    print("\n" + "="*60)
    print("PREPARING HMM FEATURES")
    print("="*60)

    # 1. Get merged raw data
    print("\n1. Loading raw data...")
    raw_data = get_merged_data(join_type=join_type)
    print(f"   Raw data shape: {raw_data.shape}")

    # 2. Select key features if requested (reduces noise)
    if select_features:
        print("\n2. Selecting key features...")
        raw_data = select_key_features(raw_data, n_stocks=n_stocks, n_indices=n_indices)
        print(f"   Selected features shape: {raw_data.shape}")

    # 3. Create enhanced features (returns, volatility, technical indicators)
    print("\n3. Engineering features...")
    features = create_enhanced_features(
        raw_data,
        include_technical_indicators=True,
        volatility_window=volatility_window,
        rsi_period=rsi_period,
        momentum_period=momentum_period
    )

    # Save features before normalization/PCA (for analysis and debugging)
    if save_to_csv:
        print(f"\n   Saving enhanced features to {csv_filename}...")
        features.to_csv(csv_filename, index=True)
        print(f"   Saved {features.shape[0]} rows and {features.shape[1]} columns")

    # 4. Apply PCA if requested
    pca_model = None
    pca_scaler = None
    if use_pca:
        print(f"\n4. Applying PCA (n_components={n_components})...")
        features, pca_model, pca_scaler, explained_var = apply_pca(
            features,
            n_components=n_components
        )
        print(f"   Top 5 components explain: {explained_var[:5].sum():.2%} of variance")

    # 5. Final normalization
    print(f"\n5. Final normalization ({scaling_method})...")
    features_normalized, final_scaler = normalize_features(features, method=scaling_method)

    print("\n" + "="*60)
    print(f"FINAL FEATURE MATRIX: {features_normalized.shape}")
    print(f"Date range: {features_normalized.index[0]} to {features_normalized.index[-1]}")
    print("="*60 + "\n")

    return features_normalized, final_scaler, pca_model, pca_scaler


def get_enhanced_features_for_model(join_type='inner', n_stocks=15, n_indices=5,
                                   volatility_window=20, rsi_period=14, momentum_period=10,
                                   save_to_csv=True, csv_filename='merged_data_with_features.csv',
                                   include_returns=True, include_volatility=True,
                                   include_rsi=True, include_momentum=True,
                                   include_market_breadth=True):
    """
    Get enhanced features for the HMM model without PCA or normalization.
    This returns the raw enhanced features (returns, volatility, market breadth, RSI, momentum)
    that can be inspected and used directly.

    Args:
        join_type: 'inner' or 'outer' for data merging
        n_stocks: Number of stocks to select
        n_indices: Number of indices to select
        volatility_window: Window for volatility calculation
        rsi_period: Period for RSI calculation
        momentum_period: Period for momentum calculation
        save_to_csv: Whether to save to CSV
        csv_filename: Filename for CSV output
        include_returns: Include log returns features
        include_volatility: Include rolling volatility features
        include_rsi: Include RSI technical indicators
        include_momentum: Include momentum indicators
        include_market_breadth: Include market breadth indicators

    Returns:
        DataFrame with enhanced features (returns, volatility, market breadth, etc.)
    """
    print("\n" + "="*60)
    print("GENERATING ENHANCED FEATURES FOR HMM MODEL")
    print("="*60)

    # 1. Get merged raw data
    print("\n1. Loading raw data...")
    raw_data = get_merged_data(join_type=join_type)
    print(f"   Raw data shape: {raw_data.shape}")

    # 2. Select key features (reduces noise and computational burden)
    print("\n2. Selecting key features...")
    raw_data_selected = select_key_features(raw_data, n_stocks=n_stocks, n_indices=n_indices)
    print(f"   Selected features shape: {raw_data_selected.shape}")

    # 3. Create enhanced features based on selected feature types
    print("\n3. Engineering features...")
    print(f"   Feature types enabled:")
    print(f"     - Returns: {include_returns}")
    print(f"     - Volatility: {include_volatility}")
    print(f"     - RSI: {include_rsi}")
    print(f"     - Momentum: {include_momentum}")
    print(f"     - Market Breadth: {include_market_breadth}")

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

    # 4. Save to CSV if requested
    if save_to_csv:
        print(f"\n4. Saving enhanced features to {csv_filename}...")
        enhanced_features.to_csv(csv_filename, index=True)
        print(f"   Saved {enhanced_features.shape[0]} rows and {enhanced_features.shape[1]} columns")
        print(f"   File location: ./{csv_filename}")

        # Print feature breakdown
        feature_types = {
            'Returns': len([col for col in enhanced_features.columns if not any(x in col for x in ['_vol', '_rsi', '_momentum']) and col not in ['pct_positive', 'avg_return', 'cross_sectional_vol']]),
            'Volatility': len([col for col in enhanced_features.columns if '_vol' in col and col != 'cross_sectional_vol']),
            'RSI': len([col for col in enhanced_features.columns if '_rsi' in col]),
            'Momentum': len([col for col in enhanced_features.columns if '_momentum' in col]),
            'Market Breadth': len([col for col in enhanced_features.columns if col in ['pct_positive', 'avg_return', 'cross_sectional_vol']])
        }
        print("\n   Feature breakdown:")
        for feat_type, count in feature_types.items():
            print(f"     - {feat_type}: {count} features")

    print("\n" + "="*60)
    print(f"ENHANCED FEATURES READY: {enhanced_features.shape}")
    print(f"Date range: {enhanced_features.index[0]} to {enhanced_features.index[-1]}")
    print("="*60 + "\n")

    return enhanced_features

