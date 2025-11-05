import math
import pytest

pd = pytest.importorskip("pandas")

from backtest.simulator import simulate_backtest


class DummyModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def predict(self, _):
        return self.outputs


def test_simulate_backtest_computes_benchmark_with_real_prices():
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 102.0, 105.0, 95.0, 110.0],
        "atr": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    model = DummyModel([0.4, 0.8, 0.2, 0.2, 0.2])
    X_test = [[0.0] for _ in range(5)]
    y_test = [0, 1, 0, 0, 0]

    results = simulate_backtest(model, X_test, y_test, df, threshold=0.5)

    benchmark = results["Benchmark"]
    assert benchmark["Buy & Hold Rendite (%)"] == 10.0

    portfolio = results["Portfolio"]
    assert math.isclose(portfolio["Rendite (%)"], (105.0 / 102.0 - 1) * 100, abs_tol=1e-2)
    assert math.isclose(benchmark["Überperformance (%)"],
                        portfolio["Rendite (%)"] - 10.0,
                        abs_tol=1e-2)

    dist = results["Vorhersage-Verteilung"]
    assert dist["Max"] == 0.8
    assert dist["Min"] == 0.2

