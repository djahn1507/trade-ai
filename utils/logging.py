import json
import os
from datetime import datetime


def logge_metriken_json(metriken: dict, ticker: str, speicherpfad="logs/"):
    os.makedirs(speicherpfad, exist_ok=True)
    zeitstempel = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dateiname = f"{ticker}_{zeitstempel}.json"

    pfad = os.path.join(speicherpfad, dateiname)
    with open(pfad, "w") as f:
        json.dump(metriken, f, indent=2, ensure_ascii=False)

    print(f"✅ Metriken gespeichert unter: {pfad}")


def speichere_logfile(ticker: str, ergebnisse: dict, pfad: str = "logs/"):
    os.makedirs(pfad, exist_ok=True)
    zeit = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dateiname = f"{ticker}_{zeit}.json"
    full_path = os.path.join(pfad, dateiname)

    daten = {
        "Ticker": ticker,
        "Zeit": zeit,
        "Backtest-Ergebnisse": ergebnisse
    }

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(daten, f, indent=2, ensure_ascii=False)

    print(f"✅ Log gespeichert: {full_path}")
