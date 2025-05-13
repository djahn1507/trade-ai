import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    buy_signals = np.sum(y_pred == 1)
    correct_buys = np.sum((y_pred == 1) & (y_true == 1))
    winrate = correct_buys / buy_signals if buy_signals > 0 else 0.0

    return {
        "Genauigkeit": round(accuracy, 4),
        "Präzision": round(precision, 4),
        "Sensitivität (Recall)": round(recall, 4),
        "F1-Wert": round(f1, 4),
        "Kauf-Signale": int(buy_signals),
        "Korrekte Käufe": int(correct_buys),
        "Trefferquote": round(winrate, 4)
    }