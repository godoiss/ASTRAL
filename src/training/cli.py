"""Command-line training and inference interface for ANN-01."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.mlp import MLP
from src.data.preprocessing import (
    LABEL_MAPPING,
    DataPartition,
    PreparedDataset,
    export_split_manifest,
    load_inference_partition,
    prepare_kepler_dataset,
)
from src.training.config import TrainingConfig
from src.training.logging_utils import configure_structured_logging
from src.training.persistence import load_model_bundle, save_model_bundle
from src.training.trainer import Trainer, plot_training_history


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "kepler_koi.csv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ann01.json"
DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "ann-01"
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ann01_split_manifest.csv"
)
DEFAULT_INFERENCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ann01_candidate_predictions.csv"
)
DEFAULT_HISTORY_PLOT = PROJECT_ROOT / "visualization" / "ann01_training_history.png"


def _partition_summary(partition: DataPartition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "records": partition.size,
        "groups": partition.group_count,
    }
    if partition.targets is not None:
        positives = int(np.sum(partition.targets == 1.0))
        payload["confirmed"] = positives
        payload["false_positive"] = partition.size - positives
    return payload


def _assert_no_group_leakage(dataset: PreparedDataset) -> None:
    train_groups = set(dataset.train.group_ids.tolist())
    validation_groups = set(dataset.validation.group_ids.tolist())
    test_groups = set(dataset.test.group_ids.tolist())
    if train_groups & validation_groups or train_groups & test_groups:
        raise RuntimeError("Host-star leakage detected from training partition")
    if validation_groups & test_groups:
        raise RuntimeError("Host-star leakage detected between validation and test")


def train_command(arguments: argparse.Namespace) -> int:
    """Prepare real KOI data, train ANN-01, and persist observable state."""
    config = TrainingConfig.load(arguments.config)
    dataset = prepare_kepler_dataset(
        arguments.data,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    _assert_no_group_leakage(dataset)
    summaries = {
        "train": _partition_summary(dataset.train),
        "validation": _partition_summary(dataset.validation),
        "test": _partition_summary(dataset.test),
        "candidates": _partition_summary(dataset.candidates),
    }
    LOGGER.info(
        "Leakage-safe dataset prepared",
        extra={"event": "dataset_prepared", "details": summaries},
    )

    model = MLP(
        layer_sizes=config.layer_sizes,
        seed=config.seed,
        engine_name=config.engine_name,
    )
    if model.parameter_count != 97:
        raise RuntimeError(f"ANN-01 must have 97 parameters, got {model.parameter_count}")
    trainer = Trainer(config)
    trainer.fit(model, dataset.train, dataset.validation)
    test_metrics = trainer.evaluate(model, dataset.test)
    metrics: dict[str, Any] = {
        "engine_name": config.engine_name,
        "parameter_count": model.parameter_count,
        "splits": summaries,
        "test": test_metrics.to_dict(),
        "final_epoch": model.training_history[-1],
    }

    snapshot_size = min(config.snapshot_sample_size, dataset.train.size)
    snapshot_features = dataset.train.features[:snapshot_size]
    snapshot_targets = dataset.train.targets[:snapshot_size]
    assert snapshot_targets is not None
    snapshot_probabilities = model.forward(snapshot_features, retain_cache=True)
    model.backward(snapshot_targets)
    snapshot_metadata = {
        "context": "post_training_real_train_sample",
        "object_ids": dataset.train.object_ids[:snapshot_size].tolist(),
        "group_ids": dataset.train.group_ids[:snapshot_size].tolist(),
        "targets": snapshot_targets.reshape(-1).tolist(),
        "probabilities": snapshot_probabilities.reshape(-1).tolist(),
        "test_metrics": test_metrics.to_dict(),
    }

    save_model_bundle(
        arguments.output,
        model,
        dataset.preprocessor,
        config,
        metrics,
    )
    model.export_snapshot(
        arguments.output / "network_snapshot.json",
        metadata=snapshot_metadata,
    )
    export_split_manifest(dataset, arguments.manifest)
    plot_training_history(model.training_history, arguments.history_plot)
    LOGGER.info(
        "ANN-01 training completed",
        extra={
            "event": "training_completed",
            "details": {
                "model_directory": str(arguments.output),
                "test_metrics": test_metrics.to_dict(),
            },
        },
    )
    return 0


def infer_command(arguments: argparse.Namespace) -> int:
    """Load a persisted ANN-01 model and emit individual real-data results."""
    bundle = load_model_bundle(arguments.model_directory)
    partition = load_inference_partition(
        arguments.data,
        bundle.preprocessor,
        candidates_only=not arguments.all_records,
    )
    probabilities = bundle.model.predict_proba(partition.features).reshape(-1)
    predictions = (
        probabilities >= bundle.config.classification_threshold
    ).astype(np.int64)
    inverse_labels = {int(value): key for key, value in LABEL_MAPPING.items()}
    result = pd.DataFrame(
        {
            "kepoi_name": partition.object_ids,
            "kepid": partition.group_ids,
            "original_disposition": partition.dispositions,
            "ann01_probability_confirmed": probabilities,
            "ann01_prediction": [inverse_labels[int(value)] for value in predictions],
        }
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(arguments.output, index=False)

    snapshot_size = min(bundle.config.snapshot_sample_size, partition.size)
    observable_probabilities = bundle.model.forward(
        partition.features[:snapshot_size],
        retain_cache=True,
    )
    bundle.model.export_snapshot(
        arguments.model_directory / "inference_snapshot.json",
        metadata={
            "context": "real_inference_sample",
            "object_ids": partition.object_ids[:snapshot_size].tolist(),
            "group_ids": partition.group_ids[:snapshot_size].tolist(),
            "original_dispositions": partition.dispositions[:snapshot_size].tolist(),
            "probabilities": observable_probabilities.reshape(-1).tolist(),
            "note": "Gradients are empty in an inference-only snapshot.",
        },
    )
    LOGGER.info(
        "ANN-01 inference completed",
        extra={
            "event": "inference_completed",
            "details": {
                "records": partition.size,
                "output": str(arguments.output),
                "candidates_only": not arguments.all_records,
            },
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ANN-01 command-line parser."""
    parser = argparse.ArgumentParser(description="ASTRAL ANN-01 neural engine")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train ANN-01")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    train_parser.add_argument(
        "--history-plot",
        type=Path,
        default=DEFAULT_HISTORY_PLOT,
    )
    train_parser.set_defaults(handler=train_command)

    infer_parser = subparsers.add_parser("infer", help="Run ANN-01 inference")
    infer_parser.add_argument(
        "--model-directory",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
    )
    infer_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    infer_parser.add_argument("--output", type=Path, default=DEFAULT_INFERENCE_PATH)
    infer_parser.add_argument(
        "--all-records",
        action="store_true",
        help="Infer every record instead of preserving a candidate-only output.",
    )
    infer_parser.set_defaults(handler=infer_command)
    return parser


def main() -> int:
    """Run a selected ANN-01 CLI command."""
    parser = build_parser()
    arguments = parser.parse_args()
    configure_structured_logging(arguments.log_level)
    try:
        return int(arguments.handler(arguments))
    except Exception:
        LOGGER.exception(
            "ANN-01 command failed",
            extra={"event": "command_failed"},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

