import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm_model(
    input_shape: tuple,
    units: int = 64,
    dropout_rate: float = 0.2,
    output_activation: str = 'sigmoid'
) -> tf.keras.Model:
    """
    Baut ein klassifikationsbasiertes LSTM-Modell mit Platzhaltern für:
    - Preis & technische Features (LSTM geeignet)
    - Fundamentale & Sentiment-Features (später kombinierbar)
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation=output_activation))  # Klassifikation (sigmoid)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
