import pandas as pd
import numpy as np

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)

    # Feature engineering
    df['Stock_Price'] = df['Close']
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=5).std()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Price_Range'] = df['High'] - df['Low']

    df = df.fillna(method='bfill').fillna(method='ffill')
    df['time_idx'] = np.arange(len(df))
    df['stock_id'] = "ALPN"

    return df
