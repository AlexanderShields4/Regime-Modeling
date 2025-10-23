from ..data_utils import get_indices

def load_indices(period="1y", interval="1d"):
    df = get_indices(period=period, interval=interval)
    return df
