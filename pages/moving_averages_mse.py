import streamlit as st
from pages_utils import fetch_all_data
import pandas as pd
from data.data_utils import calculate_moving_average_mse, calculate_moving_averages

# ========== Moving Averages MSE Page ==========
st.title("📉 Moving Averages MSE")

# Sidebar controls
st.sidebar.header("Moving Averages MSE Settings")
st.sidebar.write("Select options for moving average MSE visualization.")

data = fetch_all_data()

view_option = st.sidebar.selectbox("Choose dataset:", ["Stocks", "Indices", "Resources"])
selected_tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=list(data[view_option.lower()].columns),
    default=list(data[view_option.lower()].columns)[:5]
)

# Main content
st.subheader(f"{view_option} Moving Averages MSE")

selected_window = st.sidebar.slider("Select moving average window:", min_value=5, max_value=50, value=20)

df_to_display = data[view_option.lower()][selected_tickers]

# Calculate rolling MSE using the util
mse_df = calculate_moving_average_mse(df_to_display, window=selected_window)

st.line_chart(mse_df)

st.subheader(f"{view_option} MSE Data")
st.dataframe(mse_df.tail(10))
import streamlit as st
from pages_utils import fetch_all_data
import pandas as pd

# ========== Moving Averages Page ==========
st.title("📉 Moving Averages")

# Sidebar controls
st.sidebar.header("Moving Averages Settings")
st.sidebar.write("Select options for moving averages visualization.")

data = fetch_all_data()

view_option = st.sidebar.selectbox("Choose dataset:", ["Stocks", "Indices", "Resources"])
selected_tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=list(data[view_option.lower()].columns),
    default=list(data[view_option.lower()].columns)[:5]
)

# Main content
st.subheader(f"{view_option} Moving Averages")

# Calculate moving averages
def calculate_moving_averages(df, window=20):
    return df.rolling(window=window).mean()

selected_window = st.sidebar.slider("Select moving average window:", min_value=5, max_value=50, value=20)
df_to_display = data[view_option.lower()][selected_tickers]
moving_avg_df = calculate_moving_averages(df_to_display, window=selected_window)

st.line_chart(moving_avg_df)

st.subheader(f"{view_option} Raw Data")
st.dataframe(moving_avg_df.tail(10))