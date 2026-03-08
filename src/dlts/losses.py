from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def inverse_frequency_class_weights(
    y: torch.Tensor, num_classes: int, eps: float = 1e-6
) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float()
    inv = 1.0 / torch.clamp(counts, min=eps)
    weights = inv / inv.mean()
    return weights
