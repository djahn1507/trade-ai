from backtest.metrics import evaluate_classification
from backtest.portfolio import kapital_backtest
import numpy as np
import pandas as pd

def simulate_backtest(model, X_test, y_test, df_test, threshold=0.6) -> dict:
    """
    Führt einen vollständigen Backtest durch: Klassifikationsmetriken und Portfolio-Simulation
    
    Args:
        model: Trainiertes ML-Modell
        X_test: Feature-Matrix für Testdaten
        y_test: Tatsächliche Labels
        df_test: DataFrame mit Testdaten (inkl. Close-Preise)
        threshold: Schwellenwert für Handelssignale
        
    Returns:
        dict: Vollständige Backtest-Ergebnisse
    """
    # Vorhersagen
    y_pred_raw = model.predict(X_test)
    
    # Sicherstellen, dass y_pred ein NumPy-Array ist und richtig formatiert
    y_pred = np.asarray(y_pred_raw).flatten()
    
    # Sicherstellen, dass y_test auch ein NumPy-Array ist
    y_test_array = np.asarray(y_test).flatten() if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test
    
    # Metriken berechnen
    klassifikation = evaluate_classification(y_test_array, y_pred, threshold)

    # DataFrame für Portfolio-Backtest vorbereiten
    if not isinstance(df_test, pd.DataFrame):
        raise TypeError("df_test muss ein pandas DataFrame sein")
    
    # Portfolio simulieren
    portfolio = kapital_backtest(df_test, y_pred, threshold)

    # Zusammenfassen für Logging
    ergebnisse = {
        "Metriken": klassifikation,
        "Portfolio": {k: v for k, v in portfolio.items() if k != "Equity-Verlauf"}
    }

    return ergebnisse