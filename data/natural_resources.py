from ..data_utils import get_natural_resources

def load_resources(period="1y", interval="1d"):
    df = get_natural_resources(period=period, interval=interval)
    return df
