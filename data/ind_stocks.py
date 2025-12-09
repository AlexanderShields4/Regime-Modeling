from .data_utils import get_individual_stocks

def load_individual_stocks(tickers=None, period=None, interval="1d", start="2000-01-01", end=None):
    return get_individual_stocks(tickers=tickers, period=period, interval=interval, start=start, end=end)

