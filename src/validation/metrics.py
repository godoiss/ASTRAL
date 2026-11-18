"""Scientific binary-classification metrics shared by all evaluated models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the pre-registered metric set from labels and positive scores."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.shape != scores.shape:
        raise ValueError("Targets and scores must have the same flattened shape")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must contain only zero and one")
    if not np.all(np.isfinite(scores)) or not np.all((scores >= 0.0) & (scores <= 1.0)):
        raise ValueError("Scores must be finite and contained in [0, 1]")

    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "pr_auc": float(average_precision_score(targets, scores)),
        "binary_cross_entropy": float(log_loss(targets, scores, labels=[0, 1])),
        "confusion_matrix": matrix.astype(int).tolist(),
        "records": int(targets.size),
        "positive_prevalence": float(np.mean(targets)),
        "threshold": float(threshold),
    }


def compute_curve_data(
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, list[float]]:
    """Return JSON-ready ROC and precision-recall curve coordinates."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, scores)
    precision, recall, _ = precision_recall_curve(targets, scores)
    return {
        "roc_false_positive_rate": false_positive_rate.tolist(),
        "roc_true_positive_rate": true_positive_rate.tolist(),
        "pr_precision": precision.tolist(),
        "pr_recall": recall.tolist(),
    }

