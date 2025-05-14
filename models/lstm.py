import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm_model(
    input_shape: tuple,
    units: int = 128,
    dropout_rate: float = 0.4,
    output_activation: str = 'sigmoid'
) -> tf.keras.Model:
    """
    Baut ein verbessertes LSTM-Modell mit zusätzlichen Schichten 
    und größeren Einheiten für bessere Mustererkennung
    """
    model = Sequential()
    
    # Input-Layer
    model.add(Input(shape=input_shape))
    
    # Erste LSTM-Schicht
    model.add(LSTM(units, return_sequences=True, recurrent_dropout=0.2))
    model.add(Dropout(dropout_rate))
    
    # Zweite LSTM-Schicht
    model.add(LSTM(units//2, return_sequences=True))
    model.add(Dropout(dropout_rate/2))
    
    # Dritte LSTM-Schicht (letzte recurrent)
    model.add(LSTM(units//4))
    model.add(Dropout(dropout_rate/2))
    
    # Dense-Layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation=output_activation))

    # Optimizer mit angepasster Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()

    return model