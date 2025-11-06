import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.layers import Lambda, Multiply, Softmax
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
    attention_scores = Dense(1, activation='tanh')(lstm2)
    squeezed_scores = Lambda(lambda x: K.squeeze(x, axis=-1), name="attention_squeeze")(attention_scores)
    attention_weights = Softmax(name="attention_softmax")(squeezed_scores)
    expanded_weights = Lambda(lambda x: K.expand_dims(x, axis=-1), name="attention_expand")(attention_weights)

    # Wenden Sie die Attention auf die LSTM-Ausgabe an
    weighted = Multiply(name="attention_weighting")([lstm2, expanded_weights])
    
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
