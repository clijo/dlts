from __future__ import annotations

import torch
from torch import nn


class BaselineCNNClassifier(nn.Module):
    """Shallow Conv1D baseline for sequence classification."""

    def __init__(
        self, input_dim: int, num_classes: int, width: int = 128, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(width),
            nn.Conv1d(width, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(width),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.transpose(1, 2))
        return self.classifier(h)

    def freeze_backbone(self) -> None:
        pass

    def unfreeze_last_n_encoder_layers(self, _n_layers: int) -> None:
        pass
