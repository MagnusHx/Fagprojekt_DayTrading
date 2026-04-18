import numpy as np
import torch

from kvant.ml_framework.models import Conv1DClassifier, ResNetLSTMClassifier, create_model
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


def test_classification_metrics_supports_binary_labels() -> None:
    """Classification metrics should work for binary directional datasets too."""
    y_true = np.asarray([0, 1, 1, 0], dtype=np.int64)
    y_pred = np.asarray([0, 1, 0, 0], dtype=np.int64)

    metrics = classification_metrics(y_true, y_pred, labels=(0, 1))

    assert metrics["support_class_0"] == 2
    assert metrics["support_class_1"] == 2
    assert metrics["y_pred_count_class_0"] == 3
    assert metrics["y_pred_count_class_1"] == 1


def test_resnet_lstm_forward_shape_matches_requested_classes() -> None:
    """ResNet-LSTM should produce logits with shape (batch, n_classes)."""
    model = ResNetLSTMClassifier(
        n_features=12,
        n_classes=2,
        conv_channels=32,
        num_blocks=2,
        kernel_size=3,
        lstm_hidden_size=16,
        lstm_layers=1,
        dropout=0.1,
    )
    x = torch.randn(4, 12, 20)
    out = model(x)
    assert out.shape == (4, 2)


def test_create_model_dispatches_known_model_names() -> None:
    """The model factory should return the requested model type."""
    conv = create_model(model_name="conv1d", n_features=8, n_classes=3)
    resnet_lstm = create_model(
        model_name="resnet_lstm",
        n_features=8,
        n_classes=2,
        conv_channels=16,
        num_blocks=1,
        kernel_size=3,
        lstm_hidden_size=8,
        lstm_layers=1,
        dropout=0.1,
    )

    assert isinstance(conv, Conv1DClassifier)
    assert isinstance(resnet_lstm, ResNetLSTMClassifier)
