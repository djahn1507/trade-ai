from train import train_model
from backtest.simulator import simulate_backtest
from utils.logging import speichere_logfile
from config import ticker
import numpy as np
import pandas as pd


def finde_optimalen_threshold(model, X_test, y_test, test_df, thresholds=None):
    """
    Findet den optimalen Threshold für die besten Backtest-Ergebnisse
    """
    if thresholds is None:
        # Feiner abgestufte Suche im besseren Bereich
        thresholds = np.concatenate([
            np.arange(0.35, 0.55, 0.015),  # Feinere Abstufung im vielversprechenden Bereich
            np.arange(0.55, 0.75, 0.05)    # Gröbere Abstufung darüber
        ])

    results = []

    print(f"Optimiere Threshold-Wert...")

    for thresh in thresholds:
        ergebnisse = simulate_backtest(
            model,
            X_test,
            y_test,
            test_df,
            threshold=thresh,
            stop_loss_pct=0.05,  # Standard-Stop-Loss
            take_profit_pct=0.10  # Standard-Take-Profit
        )

        rendite = ergebnisse["Portfolio"]["Rendite (%)"]
        win_rate = ergebnisse["Portfolio"].get("Win-Rate", 0)

        results.append({
            "Threshold": thresh,
            "Rendite (%)": rendite,
            "Win-Rate": win_rate,
            "Sharpe": ergebnisse["Portfolio"].get("Sharpe Ratio", 0)
        })

        print(
            f"  - Threshold {thresh:.2f}: Rendite = {rendite:.2f}%, Win-Rate = {win_rate}%")

    # Nach Rendite sortieren
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df["Rendite (%)"].idxmax()]

    # Speichern der Threshold-Optimierungsergebnisse als JSON
    threshold_results = {
        "Ticker": ticker,
        "Zeit": pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "Optimierung": "Threshold",
        "Ergebnisse": results_df.to_dict(orient="records"),
        "Bester_Threshold": float(best_result["Threshold"]),
        "Beste_Rendite": float(best_result["Rendite (%)"]),
        "Beste_Win_Rate": float(best_result["Win-Rate"])
    }

    # Mit der bestehenden Logging-Funktion speichern
    speichere_logfile(ticker, {"Threshold_Optimierung": threshold_results})

    print(
        f"\n✅ Optimaler Threshold: {best_result['Threshold']:.2f} mit Rendite: {best_result['Rendite (%)']}%")

    return best_result["Threshold"], best_result["Rendite (%)"]


def optimiere_risikomanagement(model, X_test, y_test, test_df, threshold, 
                               stop_losses=None, take_profits=None):
    """
    Optimiert die Risikomanagement-Parameter für maximale Rendite
    """
    if stop_losses is None:
        stop_losses = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]  # Mehr Fokus auf engere Stops
    
    if take_profits is None:
        take_profits = [0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09, 0.11]  # Mehr Optionen für Take-Profit

    results = []

    print(f"\nOptimiere Risikomanagement-Parameter...")

    for sl in stop_losses:
        for tp in take_profits:
            ergebnisse = simulate_backtest(
                model,
                X_test,
                y_test,
                test_df,
                threshold=threshold,
                stop_loss_pct=sl,
                take_profit_pct=tp
            )

            rendite = ergebnisse["Portfolio"]["Rendite (%)"]
            win_rate = ergebnisse["Portfolio"].get("Win-Rate", 0)

            results.append({
                "Stop-Loss": sl,
                "Take-Profit": tp,
                "Rendite (%)": rendite,
                "Win-Rate": win_rate,
                "Sharpe": ergebnisse["Portfolio"].get("Sharpe Ratio", 0)
            })

            print(
                f"  - SL: {sl:.2f}, TP: {tp:.2f} - Rendite: {rendite:.2f}%, Win-Rate: {win_rate}%")

    # Nach Rendite sortieren
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df["Rendite (%)"].idxmax()]

    # Speichern der Risikomanagement-Optimierungsergebnisse als JSON
    risk_results = {
        "Ticker": ticker,
        "Zeit": pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "Optimierung": "Risikomanagement",
        "Ergebnisse": results_df.to_dict(orient="records"),
        "Bester_StopLoss": float(best_result["Stop-Loss"]),
        "Bester_TakeProfit": float(best_result["Take-Profit"]),
        "Beste_Rendite": float(best_result["Rendite (%)"]),
        "Beste_Win_Rate": float(best_result["Win-Rate"])
    }

    # Mit der bestehenden Logging-Funktion speichern
    speichere_logfile(ticker, {"Risikomanagement_Optimierung": risk_results})

    print(f"\n✅ Optimale Parameter: Stop-Loss = {best_result['Stop-Loss']:.2f}, " +
          f"Take-Profit = {best_result['Take-Profit']:.2f} mit Rendite: {best_result['Rendite (%)']}%")

    return (best_result["Stop-Loss"], best_result["Take-Profit"]), best_result["Rendite (%)"]


if __name__ == "__main__":
    # 1. Training starten
    print("\n=== Trading AI System ===")
    print(f"Training und Backtesting für {ticker}...")
    model, X_test, y_test, test_df = train_model()

    # 2. Parameter optimieren
    print("\n=== Optimierung der Parameter ===")
    best_threshold, _ = finde_optimalen_threshold(
        model, X_test, y_test, test_df)
    best_risk_params, _ = optimiere_risikomanagement(
        model, X_test, y_test, test_df, best_threshold)

    # 3. Finalen Backtest mit optimierten Parametern durchführen
    print("\n=== Finaler Backtest mit optimierten Parametern ===")
    ergebnisse = simulate_backtest(
        model,
        X_test,
        y_test,
        test_df,
        threshold=best_threshold,
        stop_loss_pct=best_risk_params[0],
        take_profit_pct=best_risk_params[1]
    )

    # 4. Ergebnisse speichern
    speichere_logfile(ticker, ergebnisse)

    # 5. Zusammenfassung anzeigen
    print("\n=== Zusammenfassung ===")
    print(f"Symbol: {ticker}")
    print(f"Optimaler Threshold: {best_threshold:.2f}")
    print(f"Optimaler Stop-Loss: {best_risk_params[0]:.2f}")
    print(f"Optimaler Take-Profit: {best_risk_params[1]:.2f}")
    print(f"Rendite: {ergebnisse['Portfolio']['Rendite (%)']}%")
    print(f"Win-Rate: {ergebnisse['Portfolio'].get('Win-Rate', 0)}%")
    print(f"Sharpe Ratio: {ergebnisse['Portfolio'].get('Sharpe Ratio', 0)}")
    print(
        f"Benchmark (Buy & Hold): {ergebnisse['Benchmark']['Buy & Hold Rendite (%)']}%")
    print(
        f"Max Drawdown: {ergebnisse['Portfolio'].get('Max Drawdown (%)', 0)}%")
    print(f"Anzahl Trades: {ergebnisse['Portfolio'].get('Anzahl Trades', 0)}")

    print(f"\n✅ Analyse abgeschlossen! Alle Ergebnisse wurden als JSON gespeichert.")
