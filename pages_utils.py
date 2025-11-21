def fetch_all_data():
    """Fetch all data sources and return a dictionary of DataFrames."""
    from data.ind_stocks import load_individual_stocks
    from data.indices import load_indices
    from data.natural_resources import load_resources
    from data.moving_averages import calculate_moving_averages

    # Load data (all functions already return DataFrames)
    stocks_df = load_individual_stocks(period="15y", interval="1d")
    indices_df = load_indices(period="15y", interval="1d")
    resources_df = load_resources(period="15y", interval="1d")

    # Calculate moving averages (returns DataFrames)
    stock_moving_averages_df = calculate_moving_averages(stocks_df)
    indice_moving_averages_df = calculate_moving_averages(indices_df)
    resource_moving_averages_df = calculate_moving_averages(resources_df)

    return stocks_df, indices_df, resources_df, stock_moving_averages_df, indice_moving_averages_df, resource_moving_averages_df
    