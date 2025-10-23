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

# ====== FETCH FUNCTIONS ====== 
def _fetch_data(tickers, period="1y", interval="1d"):
    """Helper to download and format data from Yahoo Finance."""
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )
    df = pd.concat({ticker: data[ticker]["Close"] for ticker in tickers if ticker in data}, axis=1)
    df.index.name = "Date"
    return df

def get_individual_stocks(tickers=None, period="1y", interval="1d"):
    """Fetch historical data for multiple individual stocks."""
    tickers = tickers or STOCK_TICKERS
    return _fetch_data(tickers, period, interval)

def get_indices(tickers=None, period="1y", interval="1d"):
    """Fetch historical data for multiple market indices."""
    tickers = tickers or INDEX_TICKERS
    return _fetch_data(tickers, period, interval)

def get_natural_resources(tickers=None, period="1y", interval="1d"):
    """Fetch historical data for natural resources and commodities."""
    tickers = tickers or NATURAL_RESOURCES_TICKERS
    return _fetch_data(tickers, period, interval)

def get_volume_data(tickers=None, period="1y", interval="1d"):
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
    df = pd.concat({ticker: data[ticker]["Volume"] for ticker in tickers if ticker in data}, axis=1)
    df.index.name = "Date"
    return df

def calculate_moving_averages(df, window=20):
    """Calculate moving averages for a given DataFrame."""
    return df.rolling(window=window).mean()
