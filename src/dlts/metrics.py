from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


def classification_metrics(
    y_true: np.ndarray, probs: np.ndarray, class_weights: np.ndarray | None = None
) -> dict[str, float]:
    """Compute classification metrics for LSST-style imbalanced data."""
    pred = probs.argmax(axis=1)

    sample_weight = None
    if class_weights is not None:
        sample_weight = class_weights[y_true]

    logloss = float(log_loss(y_true, probs, sample_weight=sample_weight))

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": logloss,
    }
