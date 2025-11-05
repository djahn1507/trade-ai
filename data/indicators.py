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
    df['bb_pct'] = (close - df['bb_low']) / (df['bb_high'] -
                                             df['bb_low'])  # Relative Position in BB

    # Trendstärke
    df['adx'] = ta.trend.ADXIndicator(high, low, close).adx()

    # Volatilität
    df['atr'] = ta.volatility.AverageTrueRange(
        high, low, close).average_true_range()
    df['atr_pct'] = df['atr'] / close  # ATR als Prozent vom Preis

    # Oszillatoren
    df['stoch'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df['stoch_rsi'] = ta.momentum.StochasticOscillator(
        df['rsi'], df['rsi'], df['rsi'], window=14).stoch()
    df['cci'] = ta.trend.CCIIndicator(high, low, close).cci()

    # Volumen-basierte Indikatoren (falls vorhanden)
    if volume is not None:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close, volume).on_balance_volume()
        df['volume_sma'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma']

    # Trendrichtung
    # 1 für Aufwärtstrend, -1 für Abwärtstrend
    df['sma_trend'] = ((df['ma_50'] > df['ma_200']).astype(int) * 2) - 1

    # Marktphasen-Erkennung
    df['market_regime'] = 0  # 0 = neutral, 1 = bullish, -1 = bearish

    # Bullish wenn kurzer MA > langer MA und ADX > 20
    bullish_condition = (df['ma_50'] > df['ma_200']) & (df['adx'] > 20)
    df.loc[bullish_condition, 'market_regime'] = 1

    # Bearish wenn kurzer MA < langer MA und ADX > 20
    bearish_condition = (df['ma_50'] < df['ma_200']) & (df['adx'] > 20)
    df.loc[bearish_condition, 'market_regime'] = -1

    # Volatilitätsregime
    df['volatility_regime'] = 0  # 0 = normal, 1 = hoch, -1 = niedrig
    vol_threshold_high = df['atr_pct'].rolling(window=50).mean() * 1.5
    vol_threshold_low = df['atr_pct'].rolling(window=50).mean() * 0.5

    df.loc[df['atr_pct'] > vol_threshold_high, 'volatility_regime'] = 1
    df.loc[df['atr_pct'] < vol_threshold_low, 'volatility_regime'] = -1

    # Remove NaN values
    df = df.dropna()

    # Columns that must remain in ihrem ursprünglichen Wertebereich für Backtests
    preserve_columns = [
        col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'atr']
        if col in df.columns
    ]

    # Scale only the remaining feature columns
    feature_columns = [col for col in df.columns if col not in preserve_columns]

    if feature_columns:
        scaler = StandardScaler()
        df.loc[:, feature_columns] = scaler.fit_transform(df[feature_columns])

    return df
