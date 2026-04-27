from __future__ import annotations

from .conv1d import Conv1DClassifier
from .resnet_lstm import ResNetLSTMClassifier


def create_model(
    *,
    model_name: str,
    n_features: int,
    n_classes: int,
    conv_channels: int = 64,
    num_blocks: int = 2,
    kernel_size: int = 5,
    lstm_hidden_size: int = 64,
    lstm_layers: int = 1,
    dropout: float = 0.3,
):
    if model_name == "conv1d":
        return Conv1DClassifier(n_features=n_features, n_classes=n_classes, dropout=dropout)
    if model_name == "resnet_lstm":
        return ResNetLSTMClassifier(
            n_features=n_features,
            n_classes=n_classes,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name={model_name!r}")


__all__ = ["Conv1DClassifier", "ResNetLSTMClassifier", "create_model"]
