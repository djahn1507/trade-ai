from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfungen relevant
    import numpy as np  # noqa: WPS433 - nur für Typing
    import pandas as pd  # noqa: WPS433 - nur für Typing


@lru_cache(maxsize=1)
def _get_numpy_module():
    """Gibt das zur Laufzeit importierte ``numpy``-Modul zurück."""

    return import_module("numpy")


@lru_cache(maxsize=1)
def _get_pandas_module():
    """Gibt das zur Laufzeit importierte ``pandas``-Modul zurück."""

    return import_module("pandas")

# def create_lstm_dataset_classification(df: pd.DataFrame, sequence_length: int = 30) -> tuple:
#     """
#     Erstellt ein klassifikationsbasiertes Dataset für LSTM:
#     - Inputs: Sequenzen der letzten N Tage (Features)
#     - Outputs: Binäre Zielvariable (1 = Kurs steigt, 0 = Kurs fällt)
#     """
#     df = df.copy()
#     df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

#     X, y = [], []
#     for i in range(len(df) - sequence_length - 1):
#         window = df.iloc[i:i+sequence_length]
#         label = df['target'].iloc[i+sequence_length]
#         features = window.drop(columns=['target']).values
#         X.append(features)
#         y.append(label)

#     return np.array(X), np.array(y)


def create_labels(df: pd.DataFrame, lookahead: int = 3, threshold: float = 0.015) -> pd.Series:
    """
    Ziel ist 1, wenn der Preis in den nächsten `lookahead` Tagen um mindestens `threshold` steigt.
    """
    pd = _get_pandas_module()

    future_return = (df['Close'].shift(-lookahead) - df['Close']) / df['Close']
    return (future_return > threshold).astype(int)


def add_features_from_sequence(X, y=None):
    """Berechnet zusätzliche Sequenz-Features wie den Trend jeder Feature-Spalte.

    Für jede Zeitreihen-Spalte wird eine lineare Regression über den Sequenzindex
    durchgeführt und die resultierende Steigung als neues Feature angehängt.
    Die Implementierung ist vollständig vektorisiert, sodass jede Feature-Spalte
    nur einmal verarbeitet wird und unnötige Kopien vermieden werden.
    """

    np = _get_numpy_module()

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must be a 3D array with shape (samples, sequence_length, features)")

    n_samples, seq_len, n_features = X.shape

    if seq_len <= 1:
        # Für Sequenzen der Länge 0 oder 1 ist die Steigung definitionsgemäß 0.
        slope_features = np.zeros((n_samples, seq_len, n_features), dtype=X.dtype)
        return np.concatenate((X, slope_features), axis=2)

    # Zeitindex vorbereiten (0, 1, ..., seq_len-1) und zentrieren.
    x = np.arange(seq_len, dtype=X.dtype)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)

    # Mittelwerte der einzelnen Feature-Spalten je Sequenz.
    y_mean = X.mean(axis=1, keepdims=True)
    y_centered = X - y_mean

    # Numerator der Steigungsformel: Summe((x - x_mean) * (y - y_mean)).
    numerator = np.sum(x_centered.reshape(1, seq_len, 1) * y_centered, axis=1)
    slopes = numerator / denom

    # Die Steigungen als konstante Feature-Spalten an jede Sequenz anhängen.
    slope_features = np.broadcast_to(slopes[:, None, :], (n_samples, seq_len, n_features))

    return np.concatenate((X, slope_features), axis=2)


def create_lstm_dataset_classification(df, sequence_length, lookahead=5, threshold=0.02):
    pd = _get_pandas_module()
    np = _get_numpy_module()

    df = df.copy()
    df['target'] = create_labels(df, lookahead, threshold)

    X, y = [], []
    for i in range(len(df) - sequence_length - lookahead):
        window = df.iloc[i:i + sequence_length]
        label = df['target'].iloc[i + sequence_length]
        X.append(window.drop(columns=['target']).values)
        y.append(label)

    X = add_features_from_sequence(np.array(X))

    return np.array(X), np.array(y)
