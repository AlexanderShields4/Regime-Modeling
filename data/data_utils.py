# data_utils.py
import yfinance as yf
import pandas as pd

# ====== STOCK TICKERS ======
# Note: Removed tickers that didn't exist in 2000 to enable full historical analysis back to 2000
# Removed: GOOGL (2004), META (2012), TSLA (2010), NFLX (2002), V (2008), MA (2006), PYPL (2015), TOT (2025), BABA (2014)
STOCK_TICKERS = [
    # Tech
    "AAPL", "MSFT", "AMZN", "NVDA", "AMD", "INTC",
    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "AXP",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "LLY", "BMY", "AMGN",
    # Industrials
    "BA", "CAT", "GE", "MMM", "HON", "LMT", "UPS", "FDX",
    # Consumer
    "PG", "KO", "PEP", "NKE", "MCD", "DIS", "SBUX", "COST", "HD", "WMT",
    # Energy
    "XOM", "CVX", "BP", "COP", "SLB",
    # Global examples
    "TM", "NSRGY", "TSM", "RIO", "SAP"
]

# ====== INDEX TICKERS ======
INDEX_TICKERS = [
    "^GSPC",  # S&P 500
    "^IXIC",  # NASDAQ
    "^DJI",   # Dow Jones
    "^RUT",   # Russell 2000
    "^VIX",   # VIX
    "^FTSE",  # FTSE 100
    "^N225",  # Nikkei 225
    "^HSI",   # Hang Seng
]

# ====== NATURAL RESOURCES TICKERS ======
# Note: Removed tickers that didn't exist in 2000 to enable full historical analysis
# Removed: PSX (2012), MPC (2011), WEAT (2011), SOYB (2011), CORN (2010),
#          UNG (2007), SLV (2006), USO (2006), CF (2005), GLD (2004), AGI (2003), VALE (2002), BG (2001)
NATURAL_RESOURCES_TICKERS = [
    # Energy
    "XOM", "CVX", "BP", "COP", "SLB", "EOG", "VLO",
    # Metals & Mining
    "FCX", "NEM", "RIO", "BHP", "AA", "CLF",
    # Agriculture
    "ADM", "TSN", "CAG", "MOS", "IP", "WY"
]

# ====== BOND ETF TICKERS ======
BOND_ETF_TICKERS = [
    "TLT",   # iShares 20+ Year Treasury Bond ETF (primary)
    "IEF",   # iShares 7-10 Year Treasury Bond ETF (intermediate)
    "AGG",   # iShares Core U.S. Aggregate Bond ETF (broad market)
]

# ====== FETCH FUNCTIONS ======
def _fetch_data(tickers: list[str], period: str = None, interval: str = "1d",
                start: str = "2000-01-01", end: str = None) -> pd.DataFrame:
    """Helper to download and format data from Yahoo Finance."""
    # Use period if provided (for backward compatibility), otherwise use start/end
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

    # Handle empty data
    if data.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns (multiple tickers) vs simple columns (single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers: columns are (Ticker, OHLCV)
        close_data = {}
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                close_data[ticker] = data[ticker]["Close"]
        df = pd.DataFrame(close_data)
    else:
        # Single ticker: columns are just OHLCV
        if len(tickers) == 1 and "Close" in data.columns:
            df = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            df = pd.DataFrame()

    df.index.name = "Date"
    return df

def get_individual_stocks(tickers=None, period=None, interval="1d", start="2000-01-01", end=None):
    """Fetch historical data for multiple individual stocks."""
    tickers = tickers or STOCK_TICKERS
    return _fetch_data(tickers, period, interval, start, end)

def get_indices(tickers=None, period=None, interval="1d", start="2000-01-01", end=None):
    """Fetch historical data for multiple market indices."""
    tickers = tickers or INDEX_TICKERS
    return _fetch_data(tickers, period, interval, start, end)

def get_natural_resources(tickers=None, period=None, interval="1d", start="2000-01-01", end=None):
    """Fetch historical data for natural resources and commodities."""
    tickers = tickers or NATURAL_RESOURCES_TICKERS
    return _fetch_data(tickers, period, interval, start, end)

def get_volume_data(tickers=None, period=None, interval="1d", start="2000-01-01", end=None):
    """Fetch volume data for multiple tickers."""
    tickers = tickers or STOCK_TICKERS

    # Use period if provided (for backward compatibility), otherwise use start/end
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

    # Handle empty data
    if data.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns (multiple tickers) vs simple columns (single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers: columns are (Ticker, OHLCV)
        volume_data = {}
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                volume_data[ticker] = data[ticker]["Volume"]
        df = pd.DataFrame(volume_data)
    else:
        # Single ticker: columns are just OHLCV
        if len(tickers) == 1 and "Volume" in data.columns:
            df = pd.DataFrame({tickers[0]: data["Volume"]})
        else:
            df = pd.DataFrame()

    df.index.name = "Date"
    return df


def get_bond_data(ticker="TLT", period=None, interval="1d", start="2000-01-01", end=None):
    """
    Fetch bond ETF data (e.g., TLT) using adjusted close prices.
    Falls back to alternative bond ETFs if unavailable.
    """
    # Helper list of fallback bond ETFs
    BOND_ETF_TICKERS = ["IEF", "AGG", "BND"]
    def _select_price_series(df, ticker_symbol):
        """Return a sensible price Series/DataFrame for the requested ticker.

        Handles several shapes returned by `yf.download` and our `_fetch_data` helper:
        - MultiIndex OHLCV (ticker, field)
        - Single-ticker OHLCV
        - Already-reduced DataFrame with ticker columns
        """
        # If empty, return empty DataFrame
        if df.empty:
            return pd.DataFrame()

        # If DataFrame has MultiIndex columns (ticker, field)
        if isinstance(df.columns, pd.MultiIndex):
            # Prefer 'Adj Close' then 'Close'
            fields = df.columns.get_level_values(1)
            if 'Adj Close' in fields:
                try:
                    return df.xs('Adj Close', axis=1, level=1)[ticker_symbol]
                except Exception:
                    # Fallback: return the first 'Adj Close' series
                    adj = df.xs('Adj Close', axis=1, level=1)
                    return adj.iloc[:, 0]
            elif 'Close' in fields:
                try:
                    return df.xs('Close', axis=1, level=1)[ticker_symbol]
                except Exception:
                    close = df.xs('Close', axis=1, level=1)
                    return close.iloc[:, 0]

        # If DataFrame has simple columns
        # If the ticker is a column, return that column (already reduced by _fetch_data)
        if ticker_symbol in df.columns:
            return df[ticker_symbol]

        # If common OHLCV column names exist, prefer 'Adj Close' then 'Close'
        if 'Adj Close' in df.columns:
            return df['Adj Close']
        if 'Close' in df.columns:
            return df['Close']

        # As a last resort, return the first column (squeezed)
        try:
            return df.iloc[:, 0]
        except Exception:
            return pd.DataFrame()

    # Try the main ticker
    try:
        data = _fetch_data([ticker], period, interval, start, end)
        if not data.empty:
            return _select_price_series(data, ticker)
    except Exception:
        pass

    # Try fallbacks if primary failed
    for fallback_ticker in BOND_ETF_TICKERS:
        if fallback_ticker != ticker:
            try:
                data = _fetch_data([fallback_ticker], period, interval, start, end)
                if not data.empty:
                    return _select_price_series(data, fallback_ticker)
            except Exception:
                continue

    # If all failed
    return pd.DataFrame()
