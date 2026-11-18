"""Mini-batch training and evaluation for the NumPy ANN-01 engine."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.core.losses import binary_cross_entropy
from src.core.mlp import MLP
from src.data.preprocessing import DataPartition
from src.training.config import TrainingConfig


LOGGER = logging.getLogger(__name__)
EpochCallback = Callable[
    [dict[str, int | float], list[dict[str, int | float]]],
    None,
]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Binary-classification metrics for one data partition."""

    loss: float
    accuracy: float
    records: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-ready metrics dictionary."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "records": self.records,
        }


def binary_accuracy(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Return the fraction of correct binary decisions."""
    predictions = (probabilities >= threshold).astype(np.float64)
    return float(np.mean(predictions == targets))


class Trainer:
    """Train an MLP with reproducible shuffled mini-batch gradient descent."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.generator = np.random.default_rng(config.seed)

    def evaluate(self, model: MLP, partition: DataPartition) -> EvaluationMetrics:
        """Evaluate loss and accuracy without changing observable caches."""
        if partition.targets is None:
            raise ValueError("Evaluation requires labelled targets")
        probabilities = model.predict_proba(partition.features)
        return EvaluationMetrics(
            loss=binary_cross_entropy(partition.targets, probabilities),
            accuracy=binary_accuracy(
                partition.targets,
                probabilities,
                threshold=self.config.classification_threshold,
            ),
            records=partition.size,
        )

    def fit(
        self,
        model: MLP,
        train: DataPartition,
        validation: DataPartition,
        epoch_callback: EpochCallback | None = None,
    ) -> list[dict[str, int | float]]:
        """Train the model and record real loss/accuracy after every epoch.

        ``epoch_callback`` receives immutable copies of the current observation
        and accumulated history. It lets local interfaces expose genuine
        training progress without moving optimization logic outside the engine.
        """
        if train.targets is None or validation.targets is None:
            raise ValueError("Training and validation partitions require targets")
        if train.size == 0 or validation.size == 0:
            raise ValueError("Training and validation partitions cannot be empty")

        history: list[dict[str, int | float]] = []
        for epoch in range(1, self.config.epochs + 1):
            shuffled_indices = self.generator.permutation(train.size)
            for start in range(0, train.size, self.config.batch_size):
                batch_indices = shuffled_indices[start : start + self.config.batch_size]
                batch_features = train.features[batch_indices]
                batch_targets = train.targets[batch_indices]
                model.forward(batch_features, retain_cache=True)
                model.backward(batch_targets)
                model.apply_gradients(self.config.learning_rate)

            train_metrics = self.evaluate(model, train)
            validation_metrics = self.evaluate(model, validation)
            epoch_record: dict[str, int | float] = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
            }
            history.append(epoch_record)
            if epoch_callback is not None:
                epoch_callback(dict(epoch_record), [dict(item) for item in history])
            if (
                epoch == 1
                or epoch == self.config.epochs
                or epoch % self.config.log_every == 0
            ):
                LOGGER.info(
                    "ANN-01 epoch completed",
                    extra={"event": "epoch_completed", "details": epoch_record},
                )

        model.training_history = history
        return history


def plot_training_history(
    history: list[dict[str, int | float]],
    path: Path,
) -> Path:
    """Plot real loss and accuracy trajectories from the training history."""
    if not history:
        raise ValueError("Training history cannot be empty")
    project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(project_root / ".cache" / "matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train_loss"]) for record in history]
    validation_loss = [float(record["validation_loss"]) for record in history]
    train_accuracy = [float(record["train_accuracy"]) for record in history]
    validation_accuracy = [
        float(record["validation_accuracy"]) for record in history
    ]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(13, 5))
    loss_axis.plot(epochs, train_loss, label="Train")
    loss_axis.plot(epochs, validation_loss, label="Validation")
    loss_axis.set_title("ANN-01 binary cross-entropy")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()

    accuracy_axis.plot(epochs, train_accuracy, label="Train")
    accuracy_axis.plot(epochs, validation_accuracy, label="Validation")
    accuracy_axis.set_title("ANN-01 accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.legend()

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
