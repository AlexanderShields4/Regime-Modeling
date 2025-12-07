from .data_utils import get_indices

def load_indices(period=None, interval="1d", start="2000-01-01", end=None):
    df = get_indices(period=period, interval=interval, start=start, end=end)
    return df
