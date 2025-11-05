import statistics
import pandas as pd

from backtest.metrics import evaluate_classification
from backtest.portfolio import kapital_backtest


def simulate_backtest(model, X_test, y_test, df_test, threshold=0.6,
                      stop_loss_pct=0.05, take_profit_pct=0.10) -> dict:
    """Führt einen vollständigen Backtest inklusive Portfolio-Simulation durch."""
    y_pred_raw = model.predict(X_test)

    def _flatten(values):
        if isinstance(values, (list, tuple)):
            flattened = []
            for item in values:
                flattened.extend(_flatten(item))
            return flattened
        if hasattr(values, "tolist"):
            return _flatten(values.tolist())
        return [float(values)]

    y_pred = [float(v) for v in _flatten(y_pred_raw)]

    if isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test_array = [float(v) for v in _flatten(y_test.values)]
    else:
        y_test_array = [float(v) for v in _flatten(y_test)]

    klassifikation = evaluate_classification(y_test_array, y_pred, threshold)

    if not isinstance(df_test, pd.DataFrame):
        raise TypeError("df_test muss ein pandas DataFrame sein")

    pred_dist = {
        "Min": float(min(y_pred)) if y_pred else 0.0,
        "Max": float(max(y_pred)) if y_pred else 0.0,
        "Mittelwert": float(sum(y_pred) / len(y_pred)) if y_pred else 0.0,
        "Median": float(statistics.median(y_pred)) if y_pred else 0.0,
        "Standardabweichung": float(statistics.pstdev(y_pred)) if len(y_pred) > 1 else 0.0,
    }

    portfolio = kapital_backtest(
        df_test, y_pred, threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    trade_details = portfolio.get("Trade-Details", [])

    if trade_details:
        positive = [t["Rendite_pct"] for t in trade_details if t["Rendite_pct"] > 0]
        negative = [t["Rendite_pct"] for t in trade_details if t["Rendite_pct"] < 0]
        avg_win = sum(positive) / len(positive) if positive else 0
        avg_loss = sum(negative) / len(negative) if negative else 0
        trade_analysis = {
            "Durchschn. Gewinn (%)": round(avg_win, 2),
            "Durchschn. Verlust (%)": round(avg_loss, 2),
            "Gewinn/Verlust-Verhältnis": round(
                abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        }
    else:
        trade_analysis = {
            "Durchschn. Gewinn (%)": 0,
            "Durchschn. Verlust (%)": 0,
            "Gewinn/Verlust-Verhältnis": 0,
        }

    benchmark = {"Buy & Hold Rendite (%)": 0, "Überperformance (%)": 0}

    equity_curve = portfolio.get("Equity-Verlauf", [])
    if len(df_test) > 1 and equity_curve:
        try:
            valid_start_index = min(1, len(df_test) - 1)
            start_price = float(df_test['Close'].iloc[valid_start_index])
            end_price = float(df_test['Close'].iloc[-1])
            initial_cash = portfolio["Startkapital"]
            shares = initial_cash / start_price if start_price != 0 else 0

            buy_hold = [initial_cash] * len(equity_curve)
            for i in range(1, len(equity_curve)):
                idx = min(i + 1, len(df_test) - 1)
                current_price = float(df_test['Close'].iloc[idx])
                buy_hold[i] = shares * current_price

            buy_hold_return = ((buy_hold[-1] - buy_hold[0]) / buy_hold[0] * 100)
            benchmark = {
                "Buy & Hold Rendite (%)": round(buy_hold_return, 2),
                "Überperformance (%)": round(
                    portfolio["Rendite (%)"] - buy_hold_return, 2),
            }
        except (IndexError, ValueError, ZeroDivisionError) as exc:
            print(f"Warning: Could not calculate buy & hold benchmark: {exc}")

    ergebnisse = {
        "Metriken": klassifikation,
        "Portfolio": {
            k: v for k, v in portfolio.items()
            if k not in ["Equity-Verlauf", "Trade-Details"]
        },
        "Trade-Analyse": trade_analysis,
        "Benchmark": benchmark,
        "Vorhersage-Verteilung": pred_dist,
        "Chart": None,
    }

    return ergebnisse
