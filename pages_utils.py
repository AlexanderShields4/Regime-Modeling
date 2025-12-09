def fetch_all_data():
    """Fetch all data sources and return DataFrames."""
    from data.ind_stocks import load_individual_stocks
    from data.indices import load_indices
    from data.natural_resources import load_resources
    from data.moving_averages import calculate_moving_averages

    stocks_df = load_individual_stocks(interval="1d")
    indices_df = load_indices(interval="1d")
    resources_df = load_resources(interval="1d")

    stock_moving_averages_df = calculate_moving_averages(stocks_df)
    indice_moving_averages_df = calculate_moving_averages(indices_df)
    resource_moving_averages_df = calculate_moving_averages(resources_df)

    return stocks_df, indices_df, resources_df, stock_moving_averages_df, indice_moving_averages_df, resource_moving_averages_df
    