import math
import pytest

pd = pytest.importorskip("pandas")

from backtest.portfolio import kapital_backtest


def test_kapital_backtest_generates_single_profitable_trade():
    data = {
        "Close": [100.0, 100.0, 102.0, 105.0, 95.0, 110.0],
        "atr": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    df = pd.DataFrame(data)

    predictions = [0.4, 0.8, 0.2, 0.2, 0.2]

    result = kapital_backtest(df, predictions, threshold=0.5, initial_cash=10_000)

    assert result["Anzahl Trades"] == 1
    assert result["Gewonnene Trades"] == 1

    expected_final = round(10_000 * (105.0 / 102.0), 2)
    assert result["Endkapital"] == expected_final
    assert math.isclose(result["Rendite (%)"], (105.0 / 102.0 - 1) * 100, abs_tol=1e-2)

