import numpy as np
import pandas as pd

def kapital_backtest(df: pd.DataFrame, predictions: np.ndarray, threshold=0.5, initial_cash=10_000):
    cash = initial_cash
    position = 0
    equity_curve = []

    assert 'Close' in df.columns, "'Close'-Spalte fehlt im DataFrame"
    assert len(predictions) + 1 <= len(df), "DataFrame ist zu kurz oder predictions zu lang!"

    for i in range(1, len(predictions) + 1):
        price_today = df['Close'].iloc[i]
        signal = predictions[i - 1] > threshold

        if signal and position == 0:
            position = cash / price_today
            cash = 0
        elif not signal and position > 0:
            cash = position * price_today
            position = 0

        portfolio_value = cash + position * price_today
        equity_curve.append(portfolio_value)

    final_value = equity_curve[-1]
    return {
        "Startkapital": initial_cash,
        "Endkapital": round(final_value, 2),
        "Rendite (%)": round((final_value - initial_cash) / initial_cash * 100, 2),
        "Max Equity": round(max(equity_curve), 2),
        "Min Equity": round(min(equity_curve), 2),
        "Equity-Verlauf": equity_curve
    }
