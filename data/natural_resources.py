from .data_utils import get_natural_resources

def load_resources(period=None, interval="1d", start="2000-01-01", end=None):
    df = get_natural_resources(period=period, interval=interval, start=start, end=end)
    return df
