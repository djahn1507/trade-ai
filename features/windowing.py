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


def create_labels(
    df: pd.DataFrame,
    lookahead: int = 5,
    threshold: float = 0.01,
) -> pd.Series:
    """Erzeugt ein binäres Label auf Basis der maximalen Rendite im Lookahead-Fenster."""

    pd = _get_pandas_module()

    close = df["Close"].astype(float)
    shifted_closes = [close.shift(-offset) for offset in range(1, max(lookahead, 1) + 1)]
    future_max = pd.concat(shifted_closes, axis=1).max(axis=1)
    future_return = (future_max - close) / close

    return (future_return > threshold).astype(int)


def add_features_from_sequence(X, y=None):
    """Berechnet zusätzliche Sequenz-Features wie Trend, Mittelwert und Volatilität."""

    np = _get_numpy_module()

    X_arr = np.asarray(X)
    if X_arr.ndim != 3:
        raise ValueError("X must be a 3D array with shape (samples, sequence_length, features)")

    data = X_arr.tolist()
    n_samples, seq_len, n_features = X_arr.shape

    slopes = [[0.0 for _ in range(n_features)] for _ in range(n_samples)]
    means = [[0.0 for _ in range(n_features)] for _ in range(n_samples)]
    stds = [[0.0 for _ in range(n_features)] for _ in range(n_samples)]
    momentum = [[0.0 for _ in range(n_features)] for _ in range(n_samples)]
    last_vs_mean = [[0.0 for _ in range(n_features)] for _ in range(n_samples)]

    if seq_len > 1:
        x_positions = list(range(seq_len))
        x_mean = sum(x_positions) / seq_len
        denom = sum((pos - x_mean) ** 2 for pos in x_positions)
    else:
        x_positions = [0]
        x_mean = 0.0
        denom = 0.0

    for sample_idx, sample in enumerate(data):
        for feature_idx in range(n_features):
            series = [step[feature_idx] for step in sample]
            mean_value = sum(series) / len(series) if series else 0.0
            means[sample_idx][feature_idx] = mean_value

            if len(series) > 1 and denom:
                numerator = sum((pos - x_mean) * (value - mean_value) for pos, value in zip(x_positions, series))
                slopes[sample_idx][feature_idx] = numerator / denom
                variance = sum((value - mean_value) ** 2 for value in series) / len(series)
                stds[sample_idx][feature_idx] = variance ** 0.5
                momentum[sample_idx][feature_idx] = series[-1] - series[0]
                last_vs_mean[sample_idx][feature_idx] = series[-1] - mean_value
            else:
                slopes[sample_idx][feature_idx] = 0.0
                stds[sample_idx][feature_idx] = 0.0
                momentum[sample_idx][feature_idx] = 0.0
                last_vs_mean[sample_idx][feature_idx] = series[-1] - mean_value if series else 0.0

    def _expand(feature_matrix):
        expanded = []
        for sample_idx in range(n_samples):
            repeated = []
            for _ in range(seq_len):
                repeated.append(feature_matrix[sample_idx])
            expanded.append(repeated)
        return expanded

    slope_features = _expand(slopes)
    mean_features = _expand(means)
    std_features = _expand(stds)
    momentum_features = _expand(momentum)
    last_vs_mean_features = _expand(last_vs_mean)

    enhanced = []
    for sample_idx in range(n_samples):
        sample_rows = []
        for step_idx in range(seq_len):
            row = []
            row.extend(data[sample_idx][step_idx])
            row.extend(slope_features[sample_idx][step_idx])
            row.extend(mean_features[sample_idx][step_idx])
            row.extend(std_features[sample_idx][step_idx])
            row.extend(momentum_features[sample_idx][step_idx])
            row.extend(last_vs_mean_features[sample_idx][step_idx])
            sample_rows.append(row)
        enhanced.append(sample_rows)

    return np.asarray(enhanced)


def create_lstm_dataset_classification(df, sequence_length, lookahead=5, threshold=0.01):
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
