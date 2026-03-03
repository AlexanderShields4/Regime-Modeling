import yfinance as yf
import pandas as pd

from regime_modeling.config import (
    STOCK_TICKERS,
    INDEX_TICKERS,
    NATURAL_RESOURCES_TICKERS,
    BOND_ETF_TICKERS,
    DEFAULT_START_DATE,
    DEFAULT_INTERVAL,
)


def _fetch_data(tickers: list[str], period: str = None, interval: str = DEFAULT_INTERVAL,
                start: str = DEFAULT_START_DATE, end: str = None) -> pd.DataFrame:
    """Download and format ticker data from Yahoo Finance."""
    if period is not None:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )
    else:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close_data = {}
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                close_data[ticker] = data[ticker]["Close"]
        df = pd.DataFrame(close_data)
    else:
        if len(tickers) == 1 and "Close" in data.columns:
            df = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            df = pd.DataFrame()

    df.index.name = "Date"
    return df

def get_individual_stocks(tickers=None, period=None, interval=DEFAULT_INTERVAL, start=DEFAULT_START_DATE, end=None):
    """Fetch historical stock price data."""
    tickers = tickers or STOCK_TICKERS
    return _fetch_data(tickers, period, interval, start, end)

def get_indices(tickers=None, period=None, interval=DEFAULT_INTERVAL, start=DEFAULT_START_DATE, end=None):
    """Fetch historical index price data."""
    tickers = tickers or INDEX_TICKERS
    return _fetch_data(tickers, period, interval, start, end)

def get_natural_resources(tickers=None, period=None, interval=DEFAULT_INTERVAL, start=DEFAULT_START_DATE, end=None):
    """Fetch historical commodity and resource data."""
    tickers = tickers or NATURAL_RESOURCES_TICKERS
    return _fetch_data(tickers, period, interval, start, end)

def get_volume_data(tickers=None, period=None, interval=DEFAULT_INTERVAL, start=DEFAULT_START_DATE, end=None):
    """Fetch volume data for tickers."""
    tickers = tickers or STOCK_TICKERS

    if period is not None:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )
    else:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        volume_data = {}
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                volume_data[ticker] = data[ticker]["Volume"]
        df = pd.DataFrame(volume_data)
    else:
        if len(tickers) == 1 and "Volume" in data.columns:
            df = pd.DataFrame({tickers[0]: data["Volume"]})
        else:
            df = pd.DataFrame()

    df.index.name = "Date"
    return df


def get_bond_data(ticker="TLT", period=None, interval=DEFAULT_INTERVAL, start=DEFAULT_START_DATE, end=None):
    """Fetch bond ETF data with fallback options."""
    _BOND_FALLBACKS = ["IEF", "AGG", "BND"]

    def _select_price_series(df, ticker_symbol):
        """Extract price series from DataFrame."""
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            fields = df.columns.get_level_values(1)
            if 'Adj Close' in fields:
                try:
                    return df.xs('Adj Close', axis=1, level=1)[ticker_symbol]
                except Exception:
                    adj = df.xs('Adj Close', axis=1, level=1)
                    return adj.iloc[:, 0]
            elif 'Close' in fields:
                try:
                    return df.xs('Close', axis=1, level=1)[ticker_symbol]
                except Exception:
                    close = df.xs('Close', axis=1, level=1)
                    return close.iloc[:, 0]

        if ticker_symbol in df.columns:
            return df[ticker_symbol]

        if 'Adj Close' in df.columns:
            return df['Adj Close']
        elif 'Close' in df.columns:
            return df['Close']

        try:
            return df.iloc[:, 0]
        except Exception:
            return pd.DataFrame()

    try:
        data = _fetch_data([ticker], period, interval, start, end)
        if not data.empty:
            return _select_price_series(data, ticker)
    except Exception:
        pass

    for fallback_ticker in _BOND_FALLBACKS:
        if fallback_ticker != ticker:
            try:
                data = _fetch_data([fallback_ticker], period, interval, start, end)
                if not data.empty:
                    return _select_price_series(data, fallback_ticker)
            except Exception:
                continue

    return pd.DataFrame()
