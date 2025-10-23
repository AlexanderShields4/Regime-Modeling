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
