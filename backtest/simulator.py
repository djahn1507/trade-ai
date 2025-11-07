import statistics
import pandas as pd

from backtest.metrics import evaluate_classification
from backtest.portfolio import kapital_backtest


def simulate_backtest(
    model,
    X_test,
    y_test,
    df_test,
    threshold=0.6,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    probability_buffer: float = 0.0,
    slippage_pct: float = 0.0005,
    trading_fee_pct: float = 0.0005,
    cooldown_bars: int = 1,
) -> dict:
    """Führt einen vollständigen Backtest inklusive Portfolio-Simulation durch."""
    y_pred_raw = model.predict(X_test)

    def _coerce_to_float(value):
        """Convert pandas/numpy scalars without triggering FutureWarnings."""

        if hasattr(value, "item"):
            try:
                return float(value.item())
            except (TypeError, ValueError):
                pass

        if hasattr(value, "iloc"):
            try:
                first_value = value.iloc[0]
            except (IndexError, TypeError, AttributeError):
                first_value = value
            else:
                return _coerce_to_float(first_value)

        return float(value)

    def _flatten(values):
        if isinstance(values, (list, tuple)):
            flattened = []
            for item in values:
                flattened.extend(_flatten(item))
            return flattened
        if hasattr(values, "tolist"):
            return _flatten(values.tolist())
        return [_coerce_to_float(values)]

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
        df_test,
        y_pred,
        threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        probability_buffer=probability_buffer,
        slippage_pct=slippage_pct,
        trading_fee_pct=trading_fee_pct,
        cooldown_bars=cooldown_bars,
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

    def _berechne_benchmark() -> dict:
        fallback = {"Buy & Hold Rendite (%)": 0, "Überperformance (%)": 0}

        if not isinstance(df_test, pd.DataFrame):
            return fallback

        if "Close" not in df_test.columns:
            return fallback

        equity_curve_local = portfolio.get("Equity-Verlauf", [])
        if len(df_test) < 2 or not equity_curve_local:
            return fallback

        try:
            close_series = df_test["Close"].astype(float)
        except (TypeError, ValueError, AttributeError):
            # ``astype`` existiert ggf. nicht bei stubs in den Tests
            try:
                close_series = df_test["Close"].apply(float)
            except AttributeError:
                return fallback

        try:
            valid_start_index = min(1, len(close_series) - 1)
            start_price = float(close_series.iloc[valid_start_index])
        except (IndexError, TypeError, ValueError, AttributeError):
            return fallback

        if start_price == 0:
            return fallback

        initial_cash = portfolio.get("Startkapital", 0)
        if not initial_cash:
            return fallback

        shares = initial_cash / start_price
        buy_hold = [initial_cash] * len(equity_curve_local)

        for i in range(1, len(equity_curve_local)):
            idx = min(i + 1, len(close_series) - 1)
            try:
                current_price = float(close_series.iloc[idx])
            except (IndexError, TypeError, ValueError, AttributeError):
                return fallback
            buy_hold[i] = shares * current_price

        if not buy_hold or buy_hold[0] == 0:
            return fallback

        buy_hold_return = (buy_hold[-1] - buy_hold[0]) / buy_hold[0] * 100
        ueberperformance = portfolio.get("Rendite (%)", 0) - buy_hold_return

        return {
            "Buy & Hold Rendite (%)": round(buy_hold_return, 2),
            "Überperformance (%)": round(ueberperformance, 2),
        }

    benchmark = _berechne_benchmark()

    total_signals = len(y_pred)
    triggered_signals = sum(1 for prob in y_pred if prob >= threshold)
    confident_signals = sum(1 for prob in y_pred if prob >= threshold + probability_buffer)
    avg_distance = (
        sum(max(0.0, prob - threshold) for prob in y_pred) / triggered_signals
        if triggered_signals
        else 0.0
    )

    ergebnisse = {
        "Metriken": klassifikation,
        "Portfolio": {
            k: v for k, v in portfolio.items()
            if k not in ["Equity-Verlauf", "Trade-Details"]
        },
        "Trade-Analyse": trade_analysis,
        "Benchmark": benchmark,
        "Vorhersage-Verteilung": pred_dist,
        "Signal-Analyse": {
            "Gesamt-Signale": total_signals,
            "Signale über Threshold": triggered_signals,
            "Signale über Threshold + Buffer": confident_signals,
            "Durchschn. Abstand zum Threshold": round(avg_distance, 4),
            "Verwendeter Buffer": probability_buffer,
        },
        "Chart": None,
    }

    return ergebnisse
