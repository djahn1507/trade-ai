import pandas as pd
import ta  # pip install ta
from config import ticker


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Nur die Spalte mit Preisdaten extrahieren
    df = pd.DataFrame({
        'Close': df['Close'].squeeze()
    }, index=df.index)

    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    df['ma_200'] = df['Close'].rolling(window=200).mean()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df.dropna()
