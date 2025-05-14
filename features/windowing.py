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


def create_lstm_dataset_classification(df, sequence_length, lookahead=5, threshold=0.02):
    df = df.copy()
    df['target'] = create_labels(df, lookahead, threshold)

    X, y = [], []
    for i in range(len(df) - sequence_length - lookahead):
        window = df.iloc[i:i + sequence_length]
        label = df['target'].iloc[i + sequence_length]
        X.append(window.drop(columns=['target']).values)
        y.append(label)

    return np.array(X), np.array(y)
