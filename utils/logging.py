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
    """
    Speichert die erweiterten Backtest-Ergebnisse im JSON-Format.
    
    Args:
        ticker: Handelssymbol
        ergebnisse: Ergebnisdaten aus dem Backtest
        pfad: Speicherpfad
    """
    os.makedirs(pfad, exist_ok=True)
    zeit = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dateiname = f"{ticker}_{zeit}.json"
    full_path = os.path.join(pfad, dateiname)
    
    # Entferne Details, die nicht gut für JSON geeignet sind
    ergebnisse_clean = ergebnisse.copy()
    if "Portfolio" in ergebnisse_clean and "Trade-Details" in ergebnisse_clean["Portfolio"]:
        del ergebnisse_clean["Portfolio"]["Trade-Details"]
    
    # Entferne Equity-Verlauf (zu groß und unnötig für JSON)
    if "Portfolio" in ergebnisse_clean and "Equity-Verlauf" in ergebnisse_clean["Portfolio"]:
        del ergebnisse_clean["Portfolio"]["Equity-Verlauf"]
    
    # NumPy-Arrays in Listen umwandeln, da sie nicht JSON-serialisierbar sind
    def clean_for_json(data):
        if isinstance(data, dict):
            return {k: clean_for_json(v) for k, v in data.items()}
        elif hasattr(data, 'tolist'):  # NumPy arrays and similar
            return data.tolist()
        elif isinstance(data, list):
            return [clean_for_json(i) for i in data]
        else:
            return data
    
    ergebnisse_clean = clean_for_json(ergebnisse_clean)
    
    daten = {
        "Ticker": ticker,
        "Zeit": zeit,
        "Backtest-Ergebnisse": ergebnisse_clean
    }

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(daten, f, indent=2, ensure_ascii=False)

    print(f"✅ Log gespeichert: {full_path}")