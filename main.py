from train import train_model
from backtest.simulator import simulate_backtest
from utils.logging import speichere_logfile
from config import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def finde_optimalen_threshold(model, X_test, y_test, test_df, thresholds=None):
    """
    Findet den optimalen Threshold für die besten Backtest-Ergebnisse
    
    Args:
        model: Trainiertes ML-Modell
        X_test, y_test: Test-Daten
        test_df: DataFrame für Backtest
        thresholds: Liste der zu testenden Threshold-Werte
        
    Returns:
        tuple: (Optimaler Threshold, Beste Rendite)
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.95, 0.05)  # Teste von 0.5 bis 0.9 in 0.05-Schritten
    
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
        
        print(f"  - Threshold {thresh:.2f}: Rendite = {rendite:.2f}%, Win-Rate = {win_rate}%")
    
    # Nach Rendite sortieren
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df["Rendite (%)"].idxmax()]
    
    # Plotte die Ergebnisse
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["Threshold"], results_df["Rendite (%)"], marker='o')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.title(f"Rendite nach Threshold-Wert für {ticker}")
    plt.xlabel("Threshold")
    plt.ylabel("Rendite (%)")
    plt.savefig(f"threshold_optimization_{ticker}.png")
    plt.close()
    
    print(f"\n✅ Optimaler Threshold: {best_result['Threshold']:.2f} mit Rendite: {best_result['Rendite (%)']}%")
    
    return best_result["Threshold"], best_result["Rendite (%)"]

def optimiere_risikomanagement(model, X_test, y_test, test_df, threshold, 
                               stop_losses=None, take_profits=None):
    """
    Optimiert die Risikomanagement-Parameter für maximale Rendite
    
    Args:
        model, X_test, y_test, test_df: Wie bei anderen Funktionen
        threshold: Zu verwendender Threshold-Wert
        stop_losses: Liste möglicher Stop-Loss-Werte
        take_profits: Liste möglicher Take-Profit-Werte
        
    Returns:
        tuple: (Beste Parameter, Beste Rendite)
    """
    if stop_losses is None:
        stop_losses = [0.02, 0.03, 0.05, 0.07, 0.10]
    
    if take_profits is None:
        take_profits = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    
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
            
            print(f"  - SL: {sl:.2f}, TP: {tp:.2f} - Rendite: {rendite:.2f}%, Win-Rate: {win_rate}%")
    
    # Nach Rendite sortieren
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df["Rendite (%)"].idxmax()]
    
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
    best_threshold, _ = finde_optimalen_threshold(model, X_test, y_test, test_df)
    best_risk_params, _ = optimiere_risikomanagement(model, X_test, y_test, test_df, best_threshold)
    
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
    print(f"Benchmark (Buy & Hold): {ergebnisse['Benchmark']['Buy & Hold Rendite (%)']}%")
    print(f"Max Drawdown: {ergebnisse['Portfolio'].get('Max Drawdown (%)', 0)}%")
    print(f"Anzahl Trades: {ergebnisse['Portfolio'].get('Anzahl Trades', 0)}")
    
    print(f"\n✅ Analyse abgeschlossen! Equity-Chart gespeichert: {ergebnisse.get('Chart')}")