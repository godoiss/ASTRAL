"""Controlled NumPy neural experiments for Stage 04."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.mlp import MLP
from src.data.preprocessing import DataPartition
from src.training.trainer import Trainer
from src.validation.config import ExperimentSpec
from src.validation.metrics import compute_classification_metrics
from src.validation.tracking import ExperimentTracker


@dataclass(frozen=True)
class RuntimeTrainingConfig:
    """Minimal trainer-compatible configuration for architecture experiments."""

    learning_rate: float
    epochs: int
    batch_size: int
    seed: int
    classification_threshold: float
    log_every: int


@dataclass
class NeuralExperimentResult:
    """In-memory model plus its fully persisted experiment record."""

    spec: ExperimentSpec
    model: MLP
    record: dict[str, Any]
    train_scores: np.ndarray
    validation_scores: np.ndarray


def _parameter_payload(model: MLP) -> dict[str, Any]:
    return {
        "weights": [weights.tolist() for weights in model.weights],
        "biases": [biases.reshape(-1).tolist() for biases in model.biases],
        "weight_shapes": [list(weights.shape) for weights in model.weights],
        "bias_shapes": [list(biases.shape) for biases in model.biases],
    }


def run_neural_experiment(
    spec: ExperimentSpec,
    train: DataPartition,
    validation: DataPartition,
    threshold: float,
    tracker: ExperimentTracker,
    dataset_version_id: str,
) -> NeuralExperimentResult:
    """Train and evaluate one model using training and validation only."""
    if train.targets is None or validation.targets is None:
        raise ValueError("Controlled experiments require labelled partitions")
    runtime_config = RuntimeTrainingConfig(
        learning_rate=spec.learning_rate,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        seed=spec.seed,
        classification_threshold=threshold,
        log_every=spec.epochs,
    )
    model = MLP(
        layer_sizes=spec.layer_sizes,
        seed=spec.seed,
        engine_name="ASTRAL-Stage04",
    )
    start = time.perf_counter()
    Trainer(runtime_config).fit(model, train, validation)  # type: ignore[arg-type]
    training_seconds = time.perf_counter() - start

    train_scores = model.predict_proba(train.features).reshape(-1)
    validation_scores = model.predict_proba(validation.features).reshape(-1)
    train_metrics = compute_classification_metrics(
        train.targets,
        train_scores,
        threshold=threshold,
    )
    validation_metrics = compute_classification_metrics(
        validation.targets,
        validation_scores,
        threshold=threshold,
    )
    record: dict[str, Any] = {
        **spec.to_dict(),
        "dataset_version_id": dataset_version_id,
        "parameter_count": model.parameter_count,
        "training_seconds": training_seconds,
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
        },
        "generalization": {
            "accuracy_gap_train_minus_validation": (
                train_metrics["accuracy"] - validation_metrics["accuracy"]
            ),
            "bce_gap_validation_minus_train": (
                validation_metrics["binary_cross_entropy"]
                - train_metrics["binary_cross_entropy"]
            ),
        },
        "history": model.training_history,
        "parameters": _parameter_payload(model),
        "graphs": [],
        "test_metrics": None,
        "test_accessed": False,
    }
    tracker.record(record)
    return NeuralExperimentResult(
        spec=spec,
        model=model,
        record=record,
        train_scores=train_scores,
        validation_scores=validation_scores,
    )


def select_by_validation_pr_auc(
    results: list[NeuralExperimentResult],
) -> NeuralExperimentResult:
    """Select using validation only, with deterministic pre-registered ties."""
    if not results:
        raise ValueError("At least one experiment is required for selection")
    return max(
        results,
        key=lambda result: (
            result.record["metrics"]["validation"]["pr_auc"],
            result.record["metrics"]["validation"]["roc_auc"],
            result.record["metrics"]["validation"]["f1"],
            -result.record["metrics"]["validation"]["binary_cross_entropy"],
        ),
    )


def summarize_seed_stability(
    results: list[NeuralExperimentResult],
) -> dict[str, Any]:
    """Summarize validation variability across controlled initialization seeds."""
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
        "binary_cross_entropy",
    )
    summary: dict[str, Any] = {
        "seeds": [result.spec.seed for result in results],
        "runs": [result.spec.experiment_id for result in results],
        "metrics": {},
    }
    for metric_name in metric_names:
        values = np.asarray(
            [
                result.record["metrics"]["validation"][metric_name]
                for result in results
            ],
            dtype=np.float64,
        )
        summary["metrics"][metric_name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return summary

