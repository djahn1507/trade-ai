import numpy as np
import pandas as pd

def kapital_backtest(df: pd.DataFrame, predictions: np.ndarray, threshold=0.5, 
                   initial_cash=10_000, stop_loss_pct=0.05, take_profit_pct=0.10):
    """
    Simuliert eine einfache Portfolio-Strategie basierend auf Vorhersage-Signalen
    mit Risikomanagement (Stop-Loss und Take-Profit).
    
    Args:
        df: DataFrame mit mindestens einer 'Close'-Spalte für die Preisdaten
        predictions: NumPy-Array mit Vorhersagewerten (Wahrscheinlichkeiten)
        threshold: Schwellenwert für Kaufsignale
        initial_cash: Anfangskapital
        stop_loss_pct: Prozentsatz für Stop-Loss (z.B. 0.05 = 5%)
        take_profit_pct: Prozentsatz für Take-Profit (z.B. 0.10 = 10%)
        
    Returns:
        dict: Performance-Metriken des simulierten Portfolios
    """
    cash = initial_cash
    position = 0
    equity_curve = []
    trades = []
    
    # Trading-Statistiken
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    assert 'Close' in df.columns, "'Close'-Spalte fehlt im DataFrame"
    assert len(predictions) + 1 <= len(df), "DataFrame ist zu kurz oder predictions zu lang!"

    # Konvertiere predictions in ein einfaches NumPy-Array falls nötig
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    
    predictions = np.asarray(predictions).flatten()
    
    # Für Stop-Loss und Take-Profit
    entry_price = 0
    stop_price = 0
    target_price = 0
    
    for i in range(1, len(predictions) + 1):
        price_today = float(df['Close'].iloc[i])
        
        # Explizit einen booleschen Wert erzeugen
        pred_value = float(predictions[i - 1])
        signal_bool = pred_value > threshold
        
        # Überprüfe Stop-Loss oder Take-Profit, wenn Position aktiv
        if position > 0:
            # Stop-Loss prüfen
            if price_today <= stop_price:
                cash = position * price_today
                # Trade-Statistiken erfassen
                trade_result = (price_today / entry_price - 1) * 100
                trades.append({
                    "Art": "Verkauf (Stop-Loss)",
                    "Einstiegspreis": entry_price,
                    "Ausstiegspreis": price_today,
                    "Rendite_pct": trade_result
                })
                
                losing_trades += 1
                total_trades += 1
                
                # Position schließen
                position = 0
                entry_price = 0
                
            # Take-Profit prüfen
            elif price_today >= target_price:
                cash = position * price_today
                # Trade-Statistiken erfassen
                trade_result = (price_today / entry_price - 1) * 100
                trades.append({
                    "Art": "Verkauf (Take-Profit)",
                    "Einstiegspreis": entry_price,
                    "Ausstiegspreis": price_today,
                    "Rendite_pct": trade_result
                })
                
                winning_trades += 1
                total_trades += 1
                
                # Position schließen
                position = 0
                entry_price = 0
        
        # Normale Handelssignale verarbeiten
        if signal_bool and position == 0:
            # Kaufen
            position = cash / price_today
            cash = 0
            
            # Setze Entry, Stop-Loss und Take-Profit
            entry_price = price_today
            stop_price = price_today * (1 - stop_loss_pct)
            target_price = price_today * (1 + take_profit_pct)
            
        elif not signal_bool and position > 0 and entry_price > 0:
            # Verkaufen (normales Signal)
            cash = position * price_today
            
            # Trade-Statistiken erfassen
            trade_result = (price_today / entry_price - 1) * 100
            trades.append({
                "Art": "Verkauf (Signal)",
                "Einstiegspreis": entry_price,
                "Ausstiegspreis": price_today,
                "Rendite_pct": trade_result
            })
            
            if price_today > entry_price:
                winning_trades += 1
            else:
                losing_trades += 1
                
            total_trades += 1
            
            # Position schließen
            position = 0
            entry_price = 0

        portfolio_value = cash + position * price_today
        equity_curve.append(portfolio_value)

    # Letzte Position ggf. auflösen (End of Backtest)
    if position > 0 and i == len(predictions):
        cash = position * price_today
        
        # Trade-Statistiken für letzten Trade
        if entry_price > 0:
            trade_result = (price_today / entry_price - 1) * 100
            trades.append({
                "Art": "Verkauf (Backtest-Ende)",
                "Einstiegspreis": entry_price,
                "Ausstiegspreis": price_today,
                "Rendite_pct": trade_result
            })
            
            if price_today > entry_price:
                winning_trades += 1
            else:
                losing_trades += 1
                
            total_trades += 1
            
        position = 0

    # Berechne Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100  # in Prozent
    max_drawdown = np.max(drawdown)
    
    # Erstellung der Ergebnisse
    final_value = equity_curve[-1]
    gesamtrendite = (final_value - initial_cash) / initial_cash * 100
    
    # Berechne Sharpe Ratio (vereinfacht, ohne risikofreien Zinssatz)
    if len(equity_curve) > 1:
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Win-Rate berechnen
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    return {
        "Startkapital": initial_cash,
        "Endkapital": round(final_value, 2),
        "Rendite (%)": round(gesamtrendite, 2),
        "Max Equity": round(max(equity_curve), 2),
        "Min Equity": round(min(equity_curve), 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Anzahl Trades": total_trades,
        "Gewonnene Trades": winning_trades,
        "Verlorene Trades": losing_trades,
        "Win-Rate": round(win_rate * 100, 2),
        "Trade-Details": trades,
        "Equity-Verlauf": equity_curve
    }