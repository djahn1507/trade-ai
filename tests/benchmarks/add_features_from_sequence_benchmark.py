"""Quick benchmark for the vectorised ``add_features_from_sequence`` implementation."""

from __future__ import annotations

import timeit
from dataclasses import dataclass

import numpy as np

from features.windowing import add_features_from_sequence


@dataclass
class BenchmarkResult:
    name: str
    duration: float

    def __str__(self) -> str:  # pragma: no cover - convenience string format
        return f"{self.name}: {self.duration:.4f}s"


def legacy_add_features_from_sequence(X: np.ndarray) -> np.ndarray:
    """Reference implementation using ``np.polyfit`` in nested loops."""

    n_samples, seq_len, n_features = X.shape
    X_enhanced = []

    for i in range(n_samples):
        seq = X[i]
        x = np.arange(seq_len).reshape(-1, 1)
        for j in range(n_features):
            y_feature = seq[:, j].reshape(-1, 1)
            slope = np.polyfit(x.flatten(), y_feature.flatten(), 1)[0]
            seq = np.column_stack((seq, np.full(seq_len, slope)))
        X_enhanced.append(seq)

    return np.array(X_enhanced)


def run_once(fn, data) -> BenchmarkResult:
    duration = timeit.timeit(lambda: fn(data), number=1)
    return BenchmarkResult(fn.__name__, duration)


def main() -> None:
    rng = np.random.default_rng(123)
    samples, seq_len, features = 512, 64, 10
    data = rng.normal(size=(samples, seq_len, features))

    legacy = run_once(legacy_add_features_from_sequence, data)
    vectorised = run_once(add_features_from_sequence, data)

    print("Benchmark add_features_from_sequence ({} samples, seq_len={}, features={})".format(
        samples, seq_len, features
    ))
    print(legacy)
    print(vectorised)

    speedup = legacy.duration / vectorised.duration if vectorised.duration else float("inf")
    print(f"Speed-up: {speedup:.2f}x")


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    main()
