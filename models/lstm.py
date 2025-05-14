import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# In models/lstm.py nach den Imports hinzufügen:
from tensorflow.keras.layers import Attention, Concatenate, Reshape, Permute, Dense, multiply, Lambda, Flatten

# Neue Modellarchitektur mit Attention-Mechanismus:


def build_lstm_model(
    input_shape: tuple,
    units: int = 128,
    dropout_rate: float = 0.4,
    output_activation: str = 'sigmoid'
) -> tf.keras.Model:
    """
    Baut ein erweitertes LSTM-Modell mit Attention-Mechanismus
    für bessere Fokussierung auf wichtige Zeitpunkte
    """
    inputs = Input(shape=input_shape)

    # Erste LSTM-Schicht mit Sequenzausgabe
    lstm1 = LSTM(units, return_sequences=True, recurrent_dropout=0.2)(inputs)
    lstm1 = Dropout(dropout_rate)(lstm1)

    # Zweite LSTM-Schicht mit Sequenzausgabe
    lstm2 = LSTM(units//2, return_sequences=True)(lstm1)
    lstm2 = Dropout(dropout_rate/2)(lstm2)

    # Attention-Mechanismus
    attention = Dense(1, activation='tanh')(lstm2)
    attention = Flatten()(attention)
    attention = tf.keras.activations.softmax(attention)
    attention = Reshape((attention.shape[1], 1))(attention)

    # Gewichtete Summe
    merge_model = multiply([lstm2, attention])
    merge_model = tf.reduce_sum(merge_model, axis=1)

    # Dense-Layers
    dense1 = Dense(64, activation='relu')(merge_model)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1, activation=output_activation)(dense2)

    # Modell erstellen
    model = tf.keras.Model(inputs=inputs, outputs=output)

    # Optimizer mit angepasster Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
