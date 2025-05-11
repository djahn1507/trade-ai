# import yfinance as yf
# import pandas as pd

# def fetch_stock_data(ticker: str, start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
#     data = yf.download(ticker, start=start, end=end)
#     data.dropna(inplace=True)
#     return data

import yfinance as yf

def fetch_stock_data(ticker: str, start_date: str, end_date: str):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.dropna(inplace=True)
    return df
