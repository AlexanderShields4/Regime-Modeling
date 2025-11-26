# data_utils.py
import yfinance as yf
import pandas as pd

# ====== STOCK TICKERS ======
STOCK_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "NFLX",
    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "AXP", "V", "MA", "PYPL",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "LLY", "BMY", "AMGN",
    # Industrials
    "BA", "CAT", "GE", "MMM", "HON", "LMT", "UPS", "FDX",
    # Consumer
    "PG", "KO", "PEP", "NKE", "MCD", "DIS", "SBUX", "COST", "HD", "WMT",
    # Energy
    "XOM", "CVX", "BP", "TOT", "COP", "SLB",
    # Global examples
    "TM", "NSRGY", "BABA", "TSM", "RIO", "SAP"
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
NATURAL_RESOURCES_TICKERS = [
    # Energy
    "XOM", "CVX", "BP", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    # Metals & Mining
    "FCX", "NEM", "RIO", "BHP", "VALE", "AA", "CLF", "AGI",
    # Agriculture
    "ADM", "BG", "TSN", "CAG", "MOS", "CF", "IP", "WY",
    # Commodities (ETFs)
    "GLD", "SLV", "USO", "UNG", "CORN", "WEAT", "SOYB"
]

# ====== BOND ETF TICKERS ======
BOND_ETF_TICKERS = [
    "TLT",   # iShares 20+ Year Treasury Bond ETF (primary)
    "IEF",   # iShares 7-10 Year Treasury Bond ETF (intermediate)
    "AGG",   # iShares Core U.S. Aggregate Bond ETF (broad market)
]

# ====== FETCH FUNCTIONS ====== 
def _fetch_data(tickers: list[str], period:str="15y", interval: str="1d") -> pd.DataFrame:
    """Helper to download and format data from Yahoo Finance."""
    data = yf.download(
        tickers=tickers,
        period=period,
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

def get_individual_stocks(tickers=None, period="15y", interval="1d"):
    """Fetch historical data for multiple individual stocks."""
    tickers = tickers or STOCK_TICKERS
    return _fetch_data(tickers, period, interval)

def get_indices(tickers=None, period="15y", interval="1d"):
    """Fetch historical data for multiple market indices."""
    tickers = tickers or INDEX_TICKERS
    return _fetch_data(tickers, period, interval)

def get_natural_resources(tickers=None, period="15y", interval="1d"):
    """Fetch historical data for natural resources and commodities."""
    tickers = tickers or NATURAL_RESOURCES_TICKERS
    return _fetch_data(tickers, period, interval)

def get_volume_data(tickers=None, period="15y", interval="1d"):
    """Fetch volume data for multiple tickers."""
    tickers = tickers or STOCK_TICKERS
    data = yf.download(
        tickers=tickers,
        period=period,
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


def get_bond_data(ticker="TLT", period="15y", interval="1d"):
    # Fetch bond ETF data (primarily TLT for long-duration treasuries)
    # Falls back to IEF or AGG if TLT unavailable
    try:
        # Try primary bond ETF (TLT)
        data = _fetch_data([ticker], period, interval)
        if not data.empty:
            return data
    except Exception as e:
        pass

    # Try fallbacks if primary failed
    for fallback_ticker in BOND_ETF_TICKERS:
        if fallback_ticker != ticker:
            try:
                data = _fetch_data([fallback_ticker], period, interval)
                if not data.empty:
                    return data
            except Exception as e:
                continue

    # If all failed, return empty DataFrame
    return pd.DataFrame()
