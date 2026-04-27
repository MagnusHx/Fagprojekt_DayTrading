from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ResNetLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        *,
        conv_channels: int = 64,
        num_blocks: int = 2,
        kernel_size: int = 5,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length cleanly")
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1")

        self.stem = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock1D(
                    conv_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, n_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.res_blocks(out)
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, channels)
        _, (hidden, _) = self.lstm(out)
        return hidden[-1]

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(x))
