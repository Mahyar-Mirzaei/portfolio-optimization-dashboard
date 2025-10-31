import pandas as pd
from typing import List

def read_prices_from_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """
    CSV should have a Date column and columns for each ticker with adjusted close prices.
    Date format: YYYY-MM-DD. Returns DataFrame indexed by Date sorted ascending.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df

# Optional: simple yfinance fetch (only if user wants)
def fetch_prices_yfinance(tickers: List[str], start: str, end: str, adjusted: bool=True) -> pd.DataFrame:
    import yfinance as yf
    data = {}
    for t in tickers:
        s = yf.download(t, start=start, end=end, progress=False)
        col = 'Adj Close' if adjusted else 'Close'
        data[t] = s[col]
    df = pd.DataFrame(data)
    return df
