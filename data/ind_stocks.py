# Change the import to use an absolute path
from data_utils import get_individual_stocks

def load_individual_stocks(period="1y", interval="1d"):
    df = get_individual_stocks(period=period, interval=interval)
    return df
