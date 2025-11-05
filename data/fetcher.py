"""Hilfsfunktionen zum Laden von Kursdaten.

Das Projekt nutzt standardmäßig ``yfinance``. In der bereitgestellten
Ausführungsumgebung ist das Paket jedoch nicht vorinstalliert, was beim
Ausführen von ``main.py`` unmittelbar zu einem ``ModuleNotFoundError`` führt.
Damit der Trainings- und Backtesting-Workflow trotzdem lauffähig bleibt,
verwenden wir – falls ``yfinance`` fehlt oder keine Daten geliefert werden –
synthetische Kursdaten, die einer zufälligen Aufwärtsbewegung mit Rauschen
folgen. Auf diese Weise lässt sich der komplette End-to-End-Prozess
durchspielen, ohne dass externe Abhängigkeiten installiert werden müssen.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional Abhängigkeit
    import yfinance as yf  # type: ignore
    _HAVE_YFINANCE = True
except ModuleNotFoundError:  # pragma: no cover - wird in der Umgebung erwartet
    yf = None  # type: ignore
    _HAVE_YFINANCE = False


def _generate_synthetic_data(
    ticker: str,
    start_date: str,
    end_date: str,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Erzeugt synthetische OHLCV-Daten für den angegebenen Zeitraum."""

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Ungültiger Zeitraum für synthetische Daten")

    index = pd.date_range(start=start, end=end, freq="B")
    if not len(index):
        raise ValueError("Der erzeugte Datumsindex ist leer")

    rng = np.random.default_rng(seed)
    drift = 0.0005
    volatility = 0.02
    steps = rng.normal(loc=drift, scale=volatility, size=len(index))
    prices = 100 * np.exp(np.cumsum(steps))

    opens = prices * (1 + rng.normal(0, 0.004, size=len(index)))
    closes = prices
    highs = np.maximum(opens, closes) * (1 + rng.normal(0.001, 0.002, len(index)))
    lows = np.minimum(opens, closes) * (1 - rng.normal(0.001, 0.002, len(index)))
    lows = np.minimum(lows, np.minimum(opens, closes))
    highs = np.maximum(highs, np.maximum(opens, closes))

    volumes = rng.integers(1e5, 5e5, size=len(index))

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Adj Close": closes,
            "Volume": volumes,
        },
        index=index,
    )
    df.index.name = "Date"

    print(
        f"⚠️  Verwende synthetische Kursdaten für {ticker}, da 'yfinance' nicht "
        "verfügbar ist oder keine Daten geliefert hat."
    )

    return df


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Lädt Kursdaten via ``yfinance`` oder erstellt eine synthetische Alternative."""

    data: Optional[pd.DataFrame] = None

    if _HAVE_YFINANCE and yf is not None:
        with contextlib.suppress(Exception):
            data = yf.download(ticker, start=start_date, end=end_date)
            if isinstance(data, pd.DataFrame) and not data.empty:
                data.dropna(inplace=True)
                if not data.empty:
                    return data

    # Wenn wir hier ankommen, liefern wir synthetische Daten
    return _generate_synthetic_data(ticker, start_date, end_date)
