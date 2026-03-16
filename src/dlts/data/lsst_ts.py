"""
LSST/PLAsTiCC data pipeline.

Preprocessing steps applied in order:
  1. NaN imputation     — forward-fill -> backward-fill -> zero-fill per channel
  2. Label encoding     — string class IDs -> contiguous integers [0, K-1]
  3. Z-normalisation    — global per-channel, statistics fit on training data only
  4. Augmentation       — applied on-the-fly at training time only (see LSSTDataset)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tslearn.datasets import UCR_UEA_datasets


@dataclass(frozen=True)
class TSMetadata:
    n_dimensions: int
    series_length: int
    class_labels: list[int]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _fill_nan(X: np.ndarray) -> np.ndarray:
    """
    Impute NaN values along the time axis per sample per channel.
    Strategy: forward-fill -> backward-fill -> zero-fill (fallback for
    channels that are entirely NaN).
    """
    X = X.copy()
    for i in range(X.shape[0]):
        for c in range(X.shape[2]):
            col = X[i, :, c]
            if not np.isnan(col).any():
                continue
            # Forward-fill
            idx = np.where(~np.isnan(col), np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col = col[idx]
            # Backward-fill
            idx = np.where(~np.isnan(col), np.arange(len(col)), len(col) - 1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            col = col[idx]
            # Zero-fill (entirely-NaN channels)
            X[i, :, c] = np.nan_to_num(col, nan=0.0)
    return X


def _normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Global per-channel z-normalisation.
    Mean and std are computed over (N, T) of the training set and applied
    to both splits — test statistics are never used.
    """
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    mean  = X_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, C)
    scale = np.maximum(X_train.std(axis=(0, 1), keepdims=True), eps)
    return (X_train - mean) / scale, (X_test - mean) / scale


def _encode_labels(
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Map raw string/int class IDs to contiguous integers [0, K-1]."""
    labels      = sorted({int(v) for v in y_train_raw})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_train = np.array([label_to_idx[int(v)] for v in y_train_raw], dtype=np.int64)
    y_test  = np.array([label_to_idx[int(v)] for v in y_test_raw],  dtype=np.int64)
    return y_train, y_test, labels


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_lsst(
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TSMetadata]:
    """Download (or load from cache) and preprocess the LSST dataset."""
    ds = UCR_UEA_datasets()
    X_train, y_train_raw, X_test, y_test_raw = ds.load_dataset("LSST")

    X_train = _fill_nan(X_train)
    X_test  = _fill_nan(X_test)

    y_train, y_test, labels = _encode_labels(y_train_raw, y_test_raw)

    if normalize:
        X_train, X_test = _normalize(X_train, X_test)
    else:
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)

    metadata = TSMetadata(
        n_dimensions=int(X_train.shape[-1]),
        series_length=int(X_train.shape[1]),
        class_labels=labels,
    )
    return X_train, y_train, X_test, y_test, metadata


# ── Dataset ───────────────────────────────────────────────────────────────────

class LSSTDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    PyTorch Dataset for LSST multivariate time series classification.

    Augmentation (training only)
    ----------------------------
    Three transforms are applied independently in __getitem__ when augment=True:

    1. Jitter          — additive Gaussian noise N(0, sigma²) applied to all
                         timesteps.  Simulates photometric measurement noise.

    2. Amplitude scale — each channel is multiplied by an independent scalar
                         drawn uniformly from scale_range.  Teaches amplitude
                         invariance without altering temporal structure.

    3. Channel dropout — each of the C passbands is independently zeroed out
                         with probability channel_drop_prob.  Directly mirrors
                         the missing-passband structure of LSST at test time
                         (up to ~30% NaN rate per passband in the raw data).

    Time-shifting / rolling is deliberately excluded: torch.roll wraps the
    series end back to the start, producing physically invalid light curves.

    Parameters
    ----------
    X               : float32 array (N, T, C)
    y               : int64 array (N,)
    device          : optional device to preload tensors onto
    augment         : enable on-the-fly augmentation (use True for train only)
    jitter_sigma    : std of additive Gaussian noise
    scale_range     : (low, high) uniform interval for amplitude scaling
    channel_drop_prob : per-channel zeroing probability
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: torch.device | None = None,
        augment: bool = False,
        jitter_sigma: float = 0.05,
        scale_range: tuple[float, float] = (0.9, 1.1),
        channel_drop_prob: float = 0.15,
    ) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

        self.augment          = augment
        self.jitter_sigma     = jitter_sigma
        self.scale_range      = scale_range
        self.channel_drop_prob = channel_drop_prob

    def __len__(self) -> int:
        return self.y.shape[0]

    def _augment(self, x: Tensor) -> Tensor:
        """Apply augmentations to a single sample x: (T, C)."""
        # 1. Jitter
        if torch.rand(1, device=x.device).item() < 0.5:
            x = x + torch.randn_like(x) * self.jitter_sigma

        # 2. Amplitude scaling (independent per channel)
        if torch.rand(1, device=x.device).item() < 0.5:
            lo, hi = self.scale_range
            scale = lo + (hi - lo) * torch.rand(x.shape[-1], device=x.device)
            x = x * scale.unsqueeze(0)

        # 3. Channel dropout (independent Bernoulli per channel)
        if self.channel_drop_prob > 0.0:
            keep = torch.rand(x.shape[-1], device=x.device) >= self.channel_drop_prob
            x = x * keep.float().unsqueeze(0)

        return x

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.X[idx]
        if self.augment:
            x = self._augment(x.clone())
        return x, self.y[idx]
