from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3, *, dropout: float = 0.3):
        super().__init__()
        dropout_rate = float(dropout)
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(x))
