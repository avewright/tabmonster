from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def compute_metrics(problem_type: str, y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    if problem_type == "regression":
        preds = logits.reshape(-1)
        rmse = mean_squared_error(y_true, preds, squared=False)
        return {"rmse": float(rmse)}

    if logits.ndim == 1 or logits.shape[1] == 1:
        scores = logits.reshape(-1)
        preds = (scores > 0).astype(int)
        metrics = {"accuracy": float(accuracy_score(y_true, preds))}
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            pass
        return metrics

    preds = np.argmax(logits, axis=1)
    return {"accuracy": float(accuracy_score(y_true, preds))}

