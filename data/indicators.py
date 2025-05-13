import pandas as pd
import ta  # pip install ta
from config import ticker
from sklearn.preprocessing import StandardScaler

import pandas as pd
import ta  # pip install ta
from sklearn.preprocessing import StandardScaler


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    # print('ori', df)
    
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Extract the price data columns correctly
    # When dealing with a multi-index DataFrame, we need to get the actual 1D series
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    
    # Calculating technical indicators with the proper 1D series
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['macd'] = ta.trend.MACD(close).macd()
    df['ma_50'] = close.rolling(window=50).mean()
    df['ma_200'] = close.rolling(window=200).mean()
    
    bb = ta.volatility.BollingerBands(close)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    # New indicators like Stochastic RSI and CCI
    df['stoch_rsi'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df['cci'] = ta.trend.CCIIndicator(high, low, close).cci()

    # Remove NaN values
    df = df.dropna()

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    return df_scaled
