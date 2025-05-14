import pandas as pd
import ta  # pip install ta
from config import ticker
from sklearn.preprocessing import StandardScaler

import pandas as pd
import ta  # pip install ta
from sklearn.preprocessing import StandardScaler


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Extract the price data columns correctly
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze() if 'Volume' in df.columns else None
    
    # Basis-Indikatoren
    df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['macd'] = ta.trend.MACD(close).macd()
    df['ma_50'] = close.rolling(window=50).mean()
    df['ma_200'] = close.rolling(window=200).mean()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_pct'] = (close - df['bb_low']) / (df['bb_high'] - df['bb_low'])  # Relative Position in BB
    
    # Trendstärke
    df['adx'] = ta.trend.ADXIndicator(high, low, close).adx()
    
    # Volatilität
    df['atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df['atr_pct'] = df['atr'] / close  # ATR als Prozent vom Preis
    
    # Oszillatoren
    df['stoch'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df['stoch_rsi'] = ta.momentum.StochasticOscillator(
        df['rsi'], df['rsi'], df['rsi'], window=14).stoch()
    df['cci'] = ta.trend.CCIIndicator(high, low, close).cci()
    
    # Volumen-basierte Indikatoren (falls vorhanden)
    if volume is not None:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df['volume_sma'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma']
    
    # Trendrichtung
    df['sma_trend'] = ((df['ma_50'] > df['ma_200']).astype(int) * 2) - 1  # 1 für Aufwärtstrend, -1 für Abwärtstrend
    
    # Remove NaN values
    df = df.dropna()

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    return df_scaled
