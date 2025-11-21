#feature engineering function for HMM model.
import pandas as pd
import numpy as np
from pages_utils import fetch_all_data


def merge_all_dataframes(
    stocks_df,
    indices_df,
    resources_df,
    stock_moving_averages_df,
    indice_moving_averages_df,
    resource_moving_averages_df,
    join_type='outer'
):
   
    df_dict = {
        'stock': stocks_df,
        'index': indices_df,
        'resource': resources_df,
        'stock_ma': stock_moving_averages_df,
        'index_ma': indice_moving_averages_df,
        'resource_ma': resource_moving_averages_df
    }

    # Add prefixes to distinguish column sources
    prefixed_dfs = [df.add_prefix(f'{prefix}_') for prefix, df in df_dict.items()]

    # Concatenate along columns (axis=1)
    combined_df = pd.concat(prefixed_dfs, axis=1, join=join_type)

    # Sort by date index
    combined_df.sort_index(inplace=True)

    return combined_df


def handle_missing_values(df):
    """Handle missing values in the merged DataFrame using hybrid strategy"""
    df_clean = df.copy()

    # Drop columns with >50% missing data
    nan_fraction = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = nan_fraction[nan_fraction > 0.5].index
    if len(cols_to_drop) > 0:
        print(f"  Dropping {len(cols_to_drop)} columns with >50% NaN: {list(cols_to_drop)[:5]}...")
    cols_to_keep = nan_fraction[nan_fraction <= 0.5].index
    df_clean = df_clean[cols_to_keep]

    #For moving averages (which naturally have NaN at start), use bfill then ffill
    ma_cols = [col for col in df_clean.columns if '_ma_' in col]
    if ma_cols:
        df_clean[ma_cols] = df_clean[ma_cols].fillna(method='bfill').fillna(method='ffill')

    #For price data, use forward fill (assumes price carries forward)
    price_cols = [col for col in df_clean.columns if '_ma_' not in col]
    if price_cols:
        df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill').fillna(method='bfill')

    #Any remaining NaNs get interpolated
    remaining_nans = df_clean.isnull().sum().sum()
    if remaining_nans > 0:
        df_clean = df_clean.interpolate(method='linear', axis=0)

    #Final cleanup - drop rows with any remaining NaNs
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(axis=0)
    rows_dropped = rows_before - len(df_clean)
    
    print(f"  Dropped {rows_dropped} rows with remaining NaN values")

    print(f"  Final shape: {df_clean.shape}")
    print(f"  Final NaN count: {df_clean.isnull().sum().sum()}")

    return df_clean


def get_merged_data(join_type='inner',):
    """
    Fetch and merge all data sources into a single DataFrame."""
    stocks_df, indices_df, resources_df, stock_moving_averages_df, indice_moving_averages_df, resource_moving_averages_df = fetch_all_data()

    # Merge into single DataFrame
    merged_df = merge_all_dataframes(
        stocks_df,
        indices_df,
        resources_df,
        stock_moving_averages_df,
        indice_moving_averages_df,
        resource_moving_averages_df,
        join_type=join_type
    )

    # Handle NaN values
    merged_df = handle_missing_values(merged_df)

    return merged_df

