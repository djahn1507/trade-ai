import numpy as np
import pytest

from features.windowing import add_features_from_sequence


def reference_slopes(X):
    n_samples, seq_len, n_features = X.shape
    x = np.arange(seq_len)
    slopes = np.empty((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            slopes[i, j] = np.polyfit(x, X[i, :, j], 1)[0]
    return slopes


def test_add_features_from_sequence_matches_polyfit():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(3, 5, 4))

    enhanced = add_features_from_sequence(X)

    assert enhanced.shape == (3, 5, 8)

    added = enhanced[:, :, 4:]
    slopes = reference_slopes(X)

    np.testing.assert_allclose(added, slopes[:, None, :])


def test_add_features_from_sequence_handles_short_sequences():
    X = np.ones((2, 1, 3))
    enhanced = add_features_from_sequence(X)
    np.testing.assert_array_equal(enhanced[:, :, 3:], np.zeros((2, 1, 3)))


def test_add_features_from_sequence_requires_3d_input():
    with pytest.raises(ValueError):
        add_features_from_sequence(np.ones((3, 4)))
