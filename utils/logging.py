"""Hilfsfunktionen für die JSON-Protokollierung von Backtest-Ergebnissen."""

from __future__ import annotations

import json
import os
from datetime import datetime


def logge_metriken_json(metriken: dict, ticker: str, speicherpfad: str = "logs/") -> None:
    """Speichert reine Metriken im JSON-Format."""

    os.makedirs(speicherpfad, exist_ok=True)
    zeitstempel = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dateiname = f"{ticker}_{zeitstempel}.json"

    pfad = os.path.join(speicherpfad, dateiname)
    with open(pfad, "w", encoding="utf-8") as handle:
        json.dump(metriken, handle, indent=2, ensure_ascii=False)

    print(f"✅ Metriken gespeichert unter: {pfad}")


def speichere_logfile(ticker: str, ergebnisse: dict, pfad: str = "logs/") -> None:
    """Speichert die erweiterten Backtest-Ergebnisse im JSON-Format."""

    os.makedirs(pfad, exist_ok=True)
    zeit = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dateiname = f"{ticker}_{zeit}.json"
    full_path = os.path.join(pfad, dateiname)

    ergebnisse_clean = ergebnisse.copy()
    if "Portfolio" in ergebnisse_clean and "Trade-Details" in ergebnisse_clean["Portfolio"]:
        del ergebnisse_clean["Portfolio"]["Trade-Details"]

    if "Portfolio" in ergebnisse_clean and "Equity-Verlauf" in ergebnisse_clean["Portfolio"]:
        del ergebnisse_clean["Portfolio"]["Equity-Verlauf"]

    def clean_for_json(data):  # type: ignore[override]
        if isinstance(data, dict):
            return {k: clean_for_json(v) for k, v in data.items()}
        if hasattr(data, "tolist"):
            return data.tolist()
        if isinstance(data, list):
            return [clean_for_json(item) for item in data]
        return data

    ergebnisse_clean = clean_for_json(ergebnisse_clean)

    daten = {
        "Ticker": ticker,
        "Zeit": zeit,
        "Backtest-Ergebnisse": ergebnisse_clean,
    }

    with open(full_path, "w", encoding="utf-8") as handle:
        json.dump(daten, handle, indent=2, ensure_ascii=False)

    print(f"✅ Log gespeichert: {full_path}")
