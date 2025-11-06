import pytest

# TensorFlow ist optional; falls es nicht installiert ist, wird dieser Test übersprungen.
tf = pytest.importorskip("tensorflow")

from models.lstm import build_lstm_model


def test_build_lstm_model_compiles_and_produces_output():
    model = build_lstm_model((5, 3))

    dummy_batch = tf.random.uniform((2, 5, 3))
    output = model(dummy_batch)

    assert output.shape == (2, 1)
