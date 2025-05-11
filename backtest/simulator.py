from backtest.metrics import evaluate_classification
from backtest.portfolio import kapital_backtest
import numpy as np

def simulate_backtest(model, X_test, y_test, df_test, threshold=0.6) -> dict:
    # Vorhersagen
    y_pred = model.predict(X_test).flatten()

    # Metriken berechnen
    klassifikation = evaluate_classification(y_test, y_pred, threshold)

    # Portfolio simulieren
    portfolio = kapital_backtest(df_test, y_pred, threshold)

    # Zusammenfassen für Logging
    ergebnisse = {
        "Metriken": klassifikation,
        "Portfolio": {k: v for k, v in portfolio.items() if k != "Equity-Verlauf"}
    }

    return ergebnisse
