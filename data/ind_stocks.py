# data/ind_stocks.py
from .data_utils import get_individual_stocks

def load_individual_stocks(tickers=None, period="1y", interval="1d"):
    """Wrapper around data_utils function for consistency."""
    return get_individual_stocks(tickers=tickers, period=period, interval=interval)

