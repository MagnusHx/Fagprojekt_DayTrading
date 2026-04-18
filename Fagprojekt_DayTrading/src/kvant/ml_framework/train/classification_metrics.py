from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> Dict[str, Any]:
    """Compute aggregate, per-class, and class-distribution metrics."""
    if len(y_true) == 0:
        return {"accuracy": 0.0}

    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)]))
    labels = np.asarray(labels, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }

    n_total = max(int(len(y_true)), 1)

    for idx, label in enumerate(labels):
        y_true_count = int(np.sum(y_true.astype(np.int64, copy=False) == label))
        y_pred_count = int(np.sum(y_pred.astype(np.int64, copy=False) == label))
        out[f"precision_class_{label}"] = float(precision[idx])
        out[f"recall_class_{label}"] = float(recall[idx])
        out[f"f1_class_{label}"] = float(f1[idx])
        out[f"support_class_{label}"] = int(support[idx])
        out[f"y_true_count_class_{label}"] = y_true_count
        out[f"y_pred_count_class_{label}"] = y_pred_count
        out[f"y_true_pct_class_{label}"] = float(y_true_count / n_total)
        out[f"y_pred_pct_class_{label}"] = float(y_pred_count / n_total)

    return out
