import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    future_return = (df['Close'].shift(-lookahead) - df['Close']) / df['Close']
    return (future_return > threshold).astype(int)


def add_features_from_sequence(X, y=None):
    """
    Extrahiert zusätzliche Features aus der Sequenz selbst
    wie Trend-Richtung, Volatilität und Momentum
    """
    n_samples, seq_len, n_features = X.shape
    X_enhanced = []

    for i in range(n_samples):
        seq = X[i]

        # Close-Preis-Spalte finden (Annahme: standardisierte Features)
        # Wir nutzen RSI-Spalte als Index 0

        # Trend-Richtung: Steigung der Regression der letzten N Punkte
        x = np.arange(seq_len).reshape(-1, 1)
        for j in range(n_features):
            y_feature = seq[:, j].reshape(-1, 1)
            slope = np.polyfit(x.flatten(), y_feature.flatten(), 1)[0]

            # Füge die Steigung als neues Feature hinzu
            seq = np.column_stack((seq, np.full(seq_len, slope)))

        X_enhanced.append(seq)

    return np.array(X_enhanced)


def create_lstm_dataset_classification(df, sequence_length, lookahead=5, threshold=0.02):
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
