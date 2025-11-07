try:  # pragma: no cover - optional dependency
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - expected in sandbox
    def accuracy_score(y_true, y_pred):
        matches = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return matches / len(y_true) if y_true else 0.0

    def precision_score(y_true, y_pred, zero_division=0):
        true_positive = sum(1 for yt, yp in zip(y_true, y_pred) if yp == 1 and yt == 1)
        predicted_positive = sum(1 for yp in y_pred if yp == 1)
        if predicted_positive == 0:
            return 0.0 if zero_division == 0 else zero_division
        return true_positive / predicted_positive

    def recall_score(y_true, y_pred, zero_division=0):
        true_positive = sum(1 for yt, yp in zip(y_true, y_pred) if yp == 1 and yt == 1)
        actual_positive = sum(1 for yt in y_true if yt == 1)
        if actual_positive == 0:
            return 0.0 if zero_division == 0 else zero_division
        return true_positive / actual_positive

    def f1_score(y_true, y_pred, zero_division=0):
        precision = precision_score(y_true, y_pred, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, zero_division=zero_division)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


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