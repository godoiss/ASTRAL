"""Tests for Stage 04 metrics, protocol, selection, and tracking."""

import json
from pathlib import Path

import numpy as np

from src.data.preprocessing import DataPartition
from src.validation.config import CampaignConfig, ExperimentSpec
from src.validation.experiment import (
    run_neural_experiment,
    select_by_validation_pr_auc,
)
from src.validation.metrics import compute_classification_metrics
from src.validation.tracking import ExperimentTracker


def _partition(features: np.ndarray, targets: np.ndarray) -> DataPartition:
    records = len(features)
    return DataPartition(
        features=features,
        targets=targets,
        group_ids=np.asarray([f"star-{index}" for index in range(records)]),
        object_ids=np.asarray([f"koi-{index}" for index in range(records)]),
        dispositions=np.where(
            targets.reshape(-1) == 1,
            "CONFIRMED",
            "FALSE POSITIVE",
        ),
    )


def test_complete_metric_set_matches_known_predictions() -> None:
    """Verify threshold, ranking, probabilistic, and matrix metrics together."""
    targets = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.9])

    metrics = compute_classification_metrics(targets, scores, threshold=0.5)

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.8
    assert metrics["roc_auc"] == 0.75
    assert 0.0 < metrics["pr_auc"] <= 1.0
    assert metrics["binary_cross_entropy"] > 0.0
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_pre_registered_campaign_varies_every_requested_axis() -> None:
    """Protect the controlled matrix and the unchanged ANN-01 reference."""
    config = CampaignConfig.load(Path("configs/scientific_validation.json"))
    categories = {item.category for item in config.experiments}
    original = config.experiments[0]

    assert len(config.experiments) == 11
    assert original.layer_sizes == (6, 8, 4, 1)
    assert original.learning_rate == 0.01
    assert original.epochs == 120
    assert original.batch_size == 64
    assert {
        "learning_rate",
        "epochs",
        "batch_size",
        "neurons",
        "hidden_layers",
    }.issubset(categories)
    assert len(config.stability_seeds) == 5


def test_experiment_record_contains_real_history_parameters_and_no_test(
    tmp_path,
) -> None:
    """Ensure a search run persists required evidence without test access."""
    generator = np.random.default_rng(13)
    features = generator.normal(size=(40, 6))
    targets = (features[:, :1] > 0).astype(np.float64)
    train = _partition(features[:30], targets[:30])
    validation = _partition(features[30:], targets[30:])
    tracker = ExperimentTracker(
        tmp_path,
        campaign_id="test-campaign",
        dataset={"version_id": "sha256:test"},
    )
    spec = ExperimentSpec(
        experiment_id="test-run",
        label="Test Run",
        category="test",
        layer_sizes=(6, 4, 1),
        learning_rate=0.01,
        epochs=3,
        batch_size=8,
        seed=17,
    )

    result = run_neural_experiment(
        spec,
        train,
        validation,
        threshold=0.5,
        tracker=tracker,
        dataset_version_id="sha256:test",
    )
    stored = json.loads((tmp_path / "runs" / "test-run.json").read_text())

    assert len(result.model.training_history) == 3
    assert stored["parameters"]["weights"]
    assert stored["parameters"]["biases"]
    assert stored["test_accessed"] is False
    assert stored["test_metrics"] is None


def test_selection_uses_validation_pr_auc() -> None:
    """Ensure selection ignores training metrics and follows the declared metric."""
    from types import SimpleNamespace

    lower = SimpleNamespace(
        record={
            "metrics": {
                "validation": {
                    "pr_auc": 0.70,
                    "roc_auc": 0.99,
                    "f1": 0.99,
                    "binary_cross_entropy": 0.1,
                }
            }
        }
    )
    higher = SimpleNamespace(
        record={
            "metrics": {
                "validation": {
                    "pr_auc": 0.71,
                    "roc_auc": 0.60,
                    "f1": 0.50,
                    "binary_cross_entropy": 0.8,
                }
            }
        }
    )

    assert select_by_validation_pr_auc([lower, higher]) is higher  # type: ignore[list-item]
