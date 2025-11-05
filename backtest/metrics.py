from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _to_list(values):
    if hasattr(values, "tolist"):
        return list(values.tolist())
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def evaluate_classification(y_true, y_pred_probs, threshold=0.5):
    y_true_list = [int(v) for v in _to_list(y_true)]
    prob_list = [float(v) for v in _to_list(y_pred_probs)]
    y_pred = [1 if prob >= threshold else 0 for prob in prob_list]

    accuracy = accuracy_score(y_true_list, y_pred)
    precision = precision_score(y_true_list, y_pred, zero_division=0)
    recall = recall_score(y_true_list, y_pred, zero_division=0)
    f1 = f1_score(y_true_list, y_pred, zero_division=0)

    buy_signals = sum(y_pred)
    correct_buys = sum(1 for yp, yt in zip(y_pred, y_true_list) if yp == 1 and yt == 1)
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