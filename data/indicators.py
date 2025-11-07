"""Feature Engineering für Marktindikatoren.

Das Originalprojekt verlässt sich stark auf das ``ta``-Paket. In der
Ausführungsumgebung steht es jedoch nicht zwingend zur Verfügung. Wir fangen
dies ab, indem wir – falls ``ta`` fehlt – auf eine kleinere Menge an manuell
berechneten Indikatoren zurückgreifen. Damit bleiben alle nachfolgenden
Pipeline-Schritte funktionsfähig und ``main.py`` lässt sich ohne zusätzliche
Installationen ausführen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - expected in the sandbox
    class StandardScaler:  # type: ignore
        def fit(self, data):
            self._means = []
            self._stds = []
            n_features = len(data[0]) if data else 0
            for col in range(n_features):
                column = [row[col] for row in data]
                mean = sum(column) / len(column) if column else 0.0
                variance = sum((value - mean) ** 2 for value in column) / len(column) if column else 0.0
                std = variance ** 0.5 if variance > 0 else 1.0
                self._means.append(mean)
                self._stds.append(std)
            return self

        def fit_transform(self, data):
            self.fit(data)
            return self.transform(data)

        def transform(self, data):
            transformed = []
            for row in data:
                transformed.append([
                    (value - mean) / std if std else 0.0
                    for value, mean, std in zip(row, self._means, self._stds)
                ])
            return transformed

try:  # pragma: no cover - optionale Abhängigkeit
    import ta  # type: ignore
    _HAVE_TA = True
except ModuleNotFoundError:  # pragma: no cover - wird erwartet
    ta = None  # type: ignore
    _HAVE_TA = False


def _safe_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Berechnet eine einfache RSI-Approximation ohne ``ta``."""

    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(method="ffill")


def _add_manual_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df.get("Volume")

    df["rsi"] = _safe_rsi(close)
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["ma_200"] = close.rolling(200).mean()
    df["returns"] = close.pct_change().fillna(0)
    df["volatility_20"] = df["returns"].rolling(20).std().fillna(method="bfill")
    df["momentum_10"] = close / close.shift(10) - 1
    df["price_range"] = (high - low) / close

    if volume is not None:
        df["volume_sma"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma"]

    df["sma_trend"] = ((df["ma_50"] > df["ma_200"]).astype(int) * 2) - 1
    df["market_regime"] = np.select(
        [df["sma_trend"] > 0, df["sma_trend"] < 0],
        [1, -1],
        default=0,
    )

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze() if "Volume" in df.columns else None

    if _HAVE_TA and ta is not None:
        df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        df["macd"] = ta.trend.MACD(close).macd()
        df["ma_50"] = close.rolling(window=50).mean()
        df["ma_200"] = close.rolling(window=200).mean()

        bb = ta.volatility.BollingerBands(close)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_pct"] = (close - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

        df["adx"] = ta.trend.ADXIndicator(high, low, close).adx()

        df["atr"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        df["atr_pct"] = df["atr"] / close

        df["stoch"] = ta.momentum.StochasticOscillator(high, low, close).stoch()
        df["stoch_rsi"] = ta.momentum.StochasticOscillator(
            df["rsi"], df["rsi"], df["rsi"], window=14
        ).stoch()
        df["cci"] = ta.trend.CCIIndicator(high, low, close).cci()

        if volume is not None:
            df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            df["volume_sma"] = volume.rolling(window=20).mean()
            df["volume_ratio"] = volume / df["volume_sma"]

        df["sma_trend"] = ((df["ma_50"] > df["ma_200"]).astype(int) * 2) - 1

        df["market_regime"] = 0
        bullish_condition = (df["ma_50"] > df["ma_200"]) & (df["adx"] > 20)
        df.loc[bullish_condition, "market_regime"] = 1

        bearish_condition = (df["ma_50"] < df["ma_200"]) & (df["adx"] > 20)
        df.loc[bearish_condition, "market_regime"] = -1

        df["volatility_regime"] = 0
        vol_threshold_high = df["atr_pct"].rolling(window=50).mean() * 1.5
        vol_threshold_low = df["atr_pct"].rolling(window=50).mean() * 0.5

        df.loc[df["atr_pct"] > vol_threshold_high, "volatility_regime"] = 1
        df.loc[df["atr_pct"] < vol_threshold_low, "volatility_regime"] = -1
    else:
        df = _add_manual_indicators(df)
        df["atr"] = (high - low).rolling(14).mean().fillna(method="bfill")
        df["atr_pct"] = df["atr"] / close

    df = df.dropna()

    preserve_columns = [
        col
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "atr"]
        if col in df.columns
    ]

    feature_columns = [col for col in df.columns if col not in preserve_columns]

    if feature_columns:
        scaler = StandardScaler()
        matrix = [[df[col].iloc[i] for col in feature_columns] for i in range(len(df))]
        scaled = scaler.fit_transform(matrix)
        for col_index, column_name in enumerate(feature_columns):
            column_values = [row[col_index] for row in scaled]
            df[column_name] = column_values

    return df
