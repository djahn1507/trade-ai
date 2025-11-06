# from sklearn.model_selection import train_test_split
# from models.lstm import build_lstm_model
# from features.windowing import create_lstm_dataset_classification
# from data.fetcher import fetch_stock_data
# from data.indicators import add_technical_indicators
# import tensorflow as tf
# import os

# def train_model(ticker="AAPL", sequence_length=30, save_path="models/lstm_model.h5"):
#     df = fetch_stock_data(ticker)
#     df = add_technical_indicators(df)
#     X, y = create_lstm_dataset_classification(df, sequence_length)

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

#     model = build_lstm_model(input_shape=X.shape[1:])
#     history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
#                         epochs=10, batch_size=32)

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     model.save(save_path)
#     print(f"✅ Modell gespeichert: {save_path}")

#     return model, history

from config import *
from data.fetcher import fetch_stock_data
from data.indicators import add_technical_indicators
from features.windowing import create_lstm_dataset_classification
from models.lstm import build_lstm_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os


def train_model():
    # 1. Daten abrufen
    df = fetch_stock_data(ticker, train_start, test_end)
    df = add_technical_indicators(df)

    # 2. Train/Test split anhand Datum
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]

    # 3. Sequenzen erzeugen
    X_train, y_train = create_lstm_dataset_classification(
        train_df, sequence_length)
    X_test, y_test = create_lstm_dataset_classification(
        test_df, sequence_length)

    # 4. Modell bauen & trainieren
    model = build_lstm_model(input_shape=X_train.shape[1:])

    # Callbacks für besseres Training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3)
    ]

    # Längeres Training mit Callbacks
    model.fit(X_train, y_train,
              epochs=20,  # Mehr Epochen mit Early Stopping
              batch_size=32,
              validation_split=0.2,  # Größerer Validierungssatz
              shuffle=False,  # Zeitreihen: Reihenfolge beibehalten
              callbacks=callbacks)

    # 5. Modell speichern
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.h5")

    return model, X_test, y_test, test_df
