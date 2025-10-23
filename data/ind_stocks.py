# data/ind_stocks.py
import data_utils as ut
#from  import get_individual_stocks  # relative import

def load_individual_stocks(tickers=None, period="1y", interval="1d"):
    """Wrapper around data_utils function for consistency."""
    return ut.get_individual_stocks(tickers=tickers, period=period, interval=interval)

