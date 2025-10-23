import streamlit as st
import pandas as pd
from pages_utils import fetch_all_data, calculate_moving_average_mse, calculate_moving_averages


# ======== Moving Averages & MSE Page ========
st.title("📉 Moving Averages & MSE")

# Sidebar controls
st.sidebar.header("Moving Averages Settings")
st.sidebar.write("Select options for moving averages and MSE visualization.")

data = fetch_all_data()

view_option = st.sidebar.selectbox("Choose dataset:", ["Stocks", "Indices", "Resources"])
available_cols = list(data[view_option.lower()].columns)

selected_tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=available_cols,
    default=available_cols[:5]
)

selected_window = st.sidebar.slider("Select moving average window:", min_value=5, max_value=50, value=20)

if not selected_tickers:
    st.warning("Please select at least one ticker to display plots.")
else:
    df_to_display = data[view_option.lower()][selected_tickers]

    # Moving averages
    moving_avg_df = calculate_moving_averages(df_to_display, window=selected_window)
    st.subheader(f"{view_option} Moving Averages (window={selected_window})")
    st.line_chart(moving_avg_df)

    # MSE for moving-average forecast
    mse_df = calculate_moving_average_mse(df_to_display, window=selected_window)
    st.subheader(f"{view_option} Moving-Average Forecast MSE (window={selected_window})")
    st.line_chart(mse_df)

    st.subheader(f"{view_option} MSE Data (last 10 rows)")
    st.dataframe(mse_df.tail(10))