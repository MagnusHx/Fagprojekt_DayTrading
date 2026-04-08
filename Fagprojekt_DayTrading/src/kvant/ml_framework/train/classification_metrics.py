from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from kvant.labels import LABEL_DOWN, LABEL_EXIT, LABEL_UP


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute aggregate, per-class, and class-distribution metrics."""
    if len(y_true) == 0:
        return {"accuracy": 0.0}

    labels = np.asarray([LABEL_DOWN, LABEL_EXIT, LABEL_UP], dtype=np.int64)
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

    y_true_counts = np.bincount(y_true.astype(np.int64, copy=False), minlength=len(labels))
    y_pred_counts = np.bincount(y_pred.astype(np.int64, copy=False), minlength=len(labels))
    n_total = max(int(len(y_true)), 1)

    for idx, label in enumerate(labels):
        out[f"precision_class_{label}"] = float(precision[idx])
        out[f"recall_class_{label}"] = float(recall[idx])
        out[f"f1_class_{label}"] = float(f1[idx])
        out[f"support_class_{label}"] = int(support[idx])
        out[f"y_true_count_class_{label}"] = int(y_true_counts[idx])
        out[f"y_pred_count_class_{label}"] = int(y_pred_counts[idx])
        out[f"y_true_pct_class_{label}"] = float(y_true_counts[idx] / n_total)
        out[f"y_pred_pct_class_{label}"] = float(y_pred_counts[idx] / n_total)

    return out
