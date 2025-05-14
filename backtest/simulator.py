from backtest.metrics import evaluate_classification
from backtest.portfolio import kapital_backtest
import numpy as np
import pandas as pd
from datetime import datetime


def simulate_backtest(model, X_test, y_test, df_test, threshold=0.6,
                      stop_loss_pct=0.05, take_profit_pct=0.10) -> dict:
    """
    Führt einen vollständigen Backtest durch: Klassifikationsmetriken und Portfolio-Simulation
    mit erweiterten Risikomanagement-Funktionen

    Args:
        model: Trainiertes ML-Modell
        X_test: Feature-Matrix für Testdaten
        y_test: Tatsächliche Labels
        df_test: DataFrame mit Testdaten (inkl. Close-Preise)
        threshold: Schwellenwert für Handelssignale
        stop_loss_pct: Prozentsatz für Stop-Loss
        take_profit_pct: Prozentsatz für Take-Profit

    Returns:
        dict: Vollständige Backtest-Ergebnisse
    """
    # Vorhersagen
    y_pred_raw = model.predict(X_test)

    # Sicherstellen, dass y_pred ein NumPy-Array ist und richtig formatiert
    y_pred = np.asarray(y_pred_raw).flatten()

    # Sicherstellen, dass y_test auch ein NumPy-Array ist
    y_test_array = np.asarray(y_test).flatten() if isinstance(
        y_test, (pd.Series, pd.DataFrame)) else y_test

    # Metriken berechnen
    klassifikation = evaluate_classification(y_test_array, y_pred, threshold)

    # DataFrame für Portfolio-Backtest vorbereiten
    if not isinstance(df_test, pd.DataFrame):
        raise TypeError("df_test muss ein pandas DataFrame sein")

    # Vorhersageverteilung analysieren
    pred_dist = {
        "Min": float(np.min(y_pred)),
        "Max": float(np.max(y_pred)),
        "Mittelwert": float(np.mean(y_pred)),
        "Median": float(np.median(y_pred)),
        "Standardabweichung": float(np.std(y_pred))
    }

    # Portfolio simulieren mit Risikomanagement
    portfolio = kapital_backtest(df_test, y_pred, threshold,
                                 stop_loss_pct=stop_loss_pct,
                                 take_profit_pct=take_profit_pct)

    # Trade-Analyse
    trade_details = portfolio.get("Trade-Details", [])

    # Gewinn/Verlust pro Trade analysieren
    if trade_details:
        avg_win = np.mean([t["Rendite_pct"] for t in trade_details if t["Rendite_pct"] > 0]) if any(
            t["Rendite_pct"] > 0 for t in trade_details) else 0
        avg_loss = np.mean([t["Rendite_pct"] for t in trade_details if t["Rendite_pct"] < 0]) if any(
            t["Rendite_pct"] < 0 for t in trade_details) else 0
        trade_analysis = {
            "Durchschn. Gewinn (%)": round(avg_win, 2),
            "Durchschn. Verlust (%)": round(avg_loss, 2),
            "Gewinn/Verlust-Verhältnis": round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2)
        }
    else:
        trade_analysis = {
            "Durchschn. Gewinn (%)": 0,
            "Durchschn. Verlust (%)": 0,
            "Gewinn/Verlust-Verhältnis": 0
        }

    # Berechne Buy & Hold Benchmark ohne Grafik zu erstellen
    benchmark = {"Buy & Hold Rendite (%)": 0, "Überperformance (%)": 0}

    if len(df_test) > 1 and len(portfolio["Equity-Verlauf"]) > 0:
        try:
            # Ensure we have valid indices to work with
            valid_start_index = min(1, len(df_test) - 1)
            start_price = df_test['Close'].iloc[valid_start_index]
            end_price = df_test['Close'].iloc[-1]
            initial_cash = portfolio["Startkapital"]
            shares = initial_cash / start_price

            # Create buy_hold array with proper length
            buy_hold = np.full(
                len(portfolio["Equity-Verlauf"]), initial_cash)

            # Fill buy_hold array with proper values
            for i in range(1, len(portfolio["Equity-Verlauf"])):
                # Ensure index is within bounds
                idx = min(i + 1, len(df_test) - 1)
                current_price = df_test['Close'].iloc[idx]
                buy_hold[i] = shares * current_price

            # Calculate buy & hold return
            buy_hold_return = (
                buy_hold[-1] - buy_hold[0]) / buy_hold[0] * 100
            benchmark = {
                "Buy & Hold Rendite (%)": round(buy_hold_return, 2),
                "Überperformance (%)": round(portfolio["Rendite (%)"] - buy_hold_return, 2)
            }
        except (IndexError, ValueError) as e:
            print(
                f"Warning: Could not calculate buy & hold benchmark: {e}")

    # Kein Chartname, da wir keine Grafik erstellen
    chart_filename = None

    # Zusammenfassen für Logging
    ergebnisse = {
        "Metriken": klassifikation,
        "Portfolio": {k: v for k, v in portfolio.items() if k not in ["Equity-Verlauf", "Trade-Details"]},
        "Trade-Analyse": trade_analysis,
        "Benchmark": benchmark,
        "Vorhersage-Verteilung": pred_dist,
        "Chart": chart_filename
    }

    return ergebnisse