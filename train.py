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

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit

from config import (
    batch_size_candidates,
    label_lookahead,
    label_threshold,
    sequence_length,
    sequence_length_candidates,
    test_end,
    test_start,
    ticker,
    train_end,
    train_start,
)
from data.fetcher import fetch_stock_data
from data.indicators import add_technical_indicators
from features.windowing import create_lstm_dataset_classification
from models.lstm import build_lstm_model


def _prepare_datasets(
    train_df,
    test_df,
    seq_length: int,
    lookahead: int,
    threshold: float,
):
    X_train, y_train = create_lstm_dataset_classification(
        train_df,
        seq_length,
        lookahead=lookahead,
        threshold=threshold,
    )
    X_test, y_test = create_lstm_dataset_classification(
        test_df,
        seq_length,
        lookahead=lookahead,
        threshold=threshold,
    )

    return X_train, y_train, X_test, y_test


def _determine_splits(n_samples: int) -> int:
    if n_samples < 120:
        return 0
    max_splits = min(5, max(2, n_samples // 240))
    return max_splits


def _build_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        ),
    ]


def _evaluate_configuration(
    X_train,
    y_train,
    batch_size: int,
) -> float:
    splits = _determine_splits(len(X_train))

    if splits < 2:
        split_index = math.floor(len(X_train) * 0.8)
        if split_index <= 0 or split_index >= len(X_train):
            return float("inf")

        model = build_lstm_model(input_shape=X_train.shape[1:])
        history = model.fit(
            X_train[:split_index],
            y_train[:split_index],
            epochs=15,
            batch_size=batch_size,
            validation_data=(X_train[split_index:], y_train[split_index:]),
            shuffle=False,
            callbacks=_build_callbacks(),
            verbose=0,
        )
        return float(min(history.history.get("val_loss", [float("inf")])))

    losses = []
    splitter = TimeSeriesSplit(n_splits=splits)
    for train_idx, val_idx in splitter.split(X_train):
        model = build_lstm_model(input_shape=X_train.shape[1:])
        history = model.fit(
            X_train[train_idx],
            y_train[train_idx],
            epochs=12,
            batch_size=batch_size,
            validation_data=(X_train[val_idx], y_train[val_idx]),
            shuffle=False,
            callbacks=_build_callbacks(),
            verbose=0,
        )
        losses.append(float(min(history.history.get("val_loss", [float("inf")]))) )

    return float(np.mean(losses)) if losses else float("inf")


def train_model():
    # 1. Daten abrufen
    df = fetch_stock_data(ticker, train_start, test_end)
    df = add_technical_indicators(df)

    # 2. Train/Test split anhand Datum
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]

    candidate_lengths: Iterable[int] = sequence_length_candidates or [sequence_length]
    candidate_batches: Iterable[int] = batch_size_candidates or [32]

    best_score = float("inf")
    best_config: Dict[str, int] = {}
    cached_train_sets: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for seq_len in candidate_lengths:
        X_train, y_train, _, _ = _prepare_datasets(
            train_df,
            test_df,
            seq_len,
            label_lookahead,
            label_threshold,
        )
        if len(X_train) == 0:
            continue

        cached_train_sets[seq_len] = (X_train, y_train)

        for batch_size in candidate_batches:
            score = _evaluate_configuration(X_train, y_train, batch_size)
            if score < best_score:
                best_score = score
                best_config = {"sequence_length": seq_len, "batch_size": batch_size}

    if not best_config:
        seq_len = sequence_length
        X_train, y_train, X_test, y_test = _prepare_datasets(
            train_df,
            test_df,
            seq_len,
            label_lookahead,
            label_threshold,
        )
        model = build_lstm_model(input_shape=X_train.shape[1:])
        model.fit(
            X_train,
            y_train,
            epochs=15,
            batch_size=32,
            validation_split=0.2,
            shuffle=False,
            callbacks=_build_callbacks(),
        )
        os.makedirs("models", exist_ok=True)
        model.save("models/lstm_model.h5")
        return model, X_test, y_test, test_df

    seq_len = best_config["sequence_length"]
    batch_size = best_config["batch_size"]
    X_train, y_train = cached_train_sets[seq_len]
    _, _, X_test, y_test = _prepare_datasets(
        train_df,
        test_df,
        seq_len,
        label_lookahead,
        label_threshold,
    )

    model = build_lstm_model(input_shape=X_train.shape[1:])

    history = model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=False,
        callbacks=_build_callbacks(),
    )

    if history.history.get("val_loss"):
        print(
            "Bestes Setup: Sequenzlänge = {seq}, Batch-Größe = {batch}, "
            "Val-Loss = {loss:.4f}".format(
                seq=seq_len,
                batch=batch_size,
                loss=min(history.history["val_loss"]),
            )
        )

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.h5")

    return model, X_test, y_test, test_df
