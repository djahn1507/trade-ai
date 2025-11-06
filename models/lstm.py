import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.layers import Flatten, Lambda, multiply
from tensorflow.keras import backend as K

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

    # Attention-Mechanismus mit Keras-kompatiblen Operationen
    attention = Dense(1, activation='tanh')(lstm2)
    attention = Flatten()(attention)
    attention = Lambda(lambda x: K.softmax(x))(attention)
    attention = Lambda(lambda x: K.expand_dims(x, axis=-1))(attention)

    # Wenden Sie die Attention auf die LSTM-Ausgabe an
    weighted = multiply([lstm2, attention])
    
    # Verwenden Sie Lambda für die Summe
    merge_model = Lambda(lambda x: K.sum(x, axis=1))(weighted)

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
