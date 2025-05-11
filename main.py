from train import train_model
from backtest.simulator import simulate_backtest
from utils.logging import speichere_logfile
from config import ticker

if __name__ == "__main__":
    # 1. Training starten
    model, X_test, y_test, test_df = train_model()

    # 2. Backtest durchführen
    ergebnisse = simulate_backtest(model, X_test, y_test, test_df, threshold=0.6)

    # 3. Ergebnisse speichern
    speichere_logfile(ticker, ergebnisse)
