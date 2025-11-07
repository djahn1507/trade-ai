import pytest

np = pytest.importorskip("numpy")

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

    n_features = X.shape[2]
    assert enhanced.shape == (3, 5, n_features * 6)

    slopes = reference_slopes(X)
    slope_block = enhanced[:, :, n_features : n_features * 2]
    np.testing.assert_allclose(slope_block, slopes[:, None, :])

    mean_block = enhanced[:, :, n_features * 2 : n_features * 3]
    np.testing.assert_allclose(mean_block, X.mean(axis=1)[:, None, :])

    momentum_block = enhanced[:, :, n_features * 4 : n_features * 5]
    expected_momentum = (X[:, -1, :] - X[:, 0, :])[:, None, :]
    np.testing.assert_allclose(
        momentum_block, np.broadcast_to(expected_momentum, momentum_block.shape)
    )


def test_add_features_from_sequence_handles_short_sequences():
    X = np.ones((2, 1, 3))
    enhanced = add_features_from_sequence(X)

    n_features = X.shape[2]
    slopes = enhanced[:, :, n_features : n_features * 2]
    np.testing.assert_array_equal(slopes, np.zeros_like(slopes))

    std_block = enhanced[:, :, n_features * 3 : n_features * 4]
    np.testing.assert_array_equal(std_block, np.zeros_like(std_block))

    mean_block = enhanced[:, :, n_features * 2 : n_features * 3]
    np.testing.assert_array_equal(mean_block, np.ones_like(mean_block))


def test_add_features_from_sequence_requires_3d_input():
    with pytest.raises(ValueError):
        add_features_from_sequence(np.ones((3, 4)))
