import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import plotly.graph_objects as go
from datetime import datetime, timedelta


# -----------------------------------------------------
# Fetch data
# -----------------------------------------------------
def fetch_intraday_data(symbol, interval='1d', days=3650):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data


# -----------------------------------------------------
# Rate of Change Feature
# -----------------------------------------------------
def calculate_roc(data, window=12):
    return data['Close'].pct_change(periods=window)


# -----------------------------------------------------
# Fit HMM Model
# -----------------------------------------------------
def fit_hmm_predict(series, n_components=3):
    clean = series.dropna()
    X = clean.values.reshape(-1, 1)

    model = hmm.GaussianHMM(n_components=n_components, n_iter=50, random_state=42)
    model.fit(X)

    hidden = model.predict(X)
    hidden = pd.Series(hidden.astype(int), index=clean.index)

    return hidden


# -----------------------------------------------------
# Generate Signals
# -----------------------------------------------------
def generate_signals(hidden_states):
    signals = pd.Series(0, index=hidden_states.index)
    signals[hidden_states == 2] = 1
    signals[hidden_states == 0] = -1
    return signals


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    symbol = "SPY"
    data = fetch_intraday_data(symbol)

    roc = calculate_roc(data)

    hidden_states = fit_hmm_predict(roc)
    signals = generate_signals(hidden_states)

    # ---------------------------
    # FORCE 1-D SHAPES (fix root cause)
    # ---------------------------
    hidden_states = hidden_states.reindex(data.index).astype("float").to_numpy().reshape(-1)
    signals = signals.reindex(data.index).astype("float").to_numpy().reshape(-1)
    roc = roc.to_numpy().reshape(-1)
    close = data['Close'].to_numpy().reshape(-1)

    # Debug shapes
    print("Close shape:", close.shape)
    print("ROC shape:", roc.shape)
    print("Regime shape:", hidden_states.shape)
    print("Signal shape:", signals.shape)

    # ---------------------------
    # Create clean DataFrame
    # ---------------------------
    result = pd.DataFrame({
        'Close': close,
        'ROC': roc,
        'Regime': hidden_states,
        'Signal': signals
    }, index=data.index).dropna()

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=result.index, y=result['Close'], mode='lines', name='Nifty Close'))

    buy = result[result['Signal'] == 1]
    sell = result[result['Signal'] == -1]

    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers',
                             marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='Buy Signal'))

    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers',
                             marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='Sell Signal'))

    fig.update_layout(
        title='S&P500 HMM Strategy',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )

    fig.show()

    print(f"Total data points: {len(result)}")
    print(f"Buy signals: {len(buy)}")
    print(f"Sell signals: {len(sell)}")


if __name__ == "__main__":
    main()
