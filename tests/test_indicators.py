import math
import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from data.indicators import add_technical_indicators


def test_add_technical_indicators_preserves_price_columns():
    dates = pd.date_range("2020-01-01", periods=250, freq="D")
    base = np.linspace(100, 150, 250)
    df = pd.DataFrame({
        "Open": base + 0.5,
        "High": base + 1.0,
        "Low": base - 1.0,
        "Close": base,
        "Adj Close": base,
        "Volume": np.linspace(1_000_000, 1_500_000, 250),
    }, index=dates)

    result = add_technical_indicators(df)

    # Die Indikatoren entfernen NaN-Werte am Anfang der Serie.
    aligned_original = df.loc[result.index, "Close"]

    pd.testing.assert_series_equal(result["Close"], aligned_original)

    # Der Durchschnittswert des Close-Preises muss dem Original entsprechen und
    # darf nicht auf 0 skaliert sein.
    assert math.isclose(result["Close"].mean(), aligned_original.mean())

    # Verifizere, dass ein skaliertes Feature (z.B. RSI) um 0 zentriert ist.
    assert math.isclose(result["rsi"].mean(), 0, abs_tol=1e-8)

