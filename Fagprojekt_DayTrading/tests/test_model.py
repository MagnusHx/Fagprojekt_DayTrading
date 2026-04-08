import numpy as np

from kvant.ml_framework.train.classification_metrics import classification_metrics


def test_classification_metrics_uses_all_three_classes() -> None:
    """Classification metrics should account for all three labels."""
    y_true = np.asarray([0, 1, 2, 1, 0, 2], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int64)

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["support_class_0"] == 2
    assert metrics["support_class_1"] == 2
    assert metrics["support_class_2"] == 2
    assert metrics["y_pred_count_class_0"] == 2
    assert metrics["y_pred_count_class_1"] == 2
    assert metrics["y_pred_count_class_2"] == 2
