from regime_modeling.data.data_utils import get_individual_stocks, get_indices, get_natural_resources
from regime_modeling.data.moving_averages import calculate_moving_averages


def load_individual_stocks(tickers=None, period=None, interval="1d", start="2000-01-01", end=None):
    return get_individual_stocks(tickers=tickers, period=period, interval=interval, start=start, end=end)


def load_indices(period=None, interval="1d", start="2000-01-01", end=None):
    df = get_indices(period=period, interval=interval, start=start, end=end)
    return df


def load_resources(period=None, interval="1d", start="2000-01-01", end=None):
    df = get_natural_resources(period=period, interval=interval, start=start, end=end)
    return df


def fetch_all_data():
    """Fetch all data sources and return DataFrames."""
    stocks_df = load_individual_stocks(interval="1d")
    indices_df = load_indices(interval="1d")
    resources_df = load_resources(interval="1d")

    stock_moving_averages_df = calculate_moving_averages(stocks_df)
    indice_moving_averages_df = calculate_moving_averages(indices_df)
    resource_moving_averages_df = calculate_moving_averages(resources_df)

    return stocks_df, indices_df, resources_df, stock_moving_averages_df, indice_moving_averages_df, resource_moving_averages_df
