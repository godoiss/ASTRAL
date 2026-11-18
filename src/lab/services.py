"""Real-data services connecting Streamlit to the ASTRAL NumPy engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.mlp import MLP
from src.data.preprocessing import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    OBJECT_ID_COLUMN,
    REQUIRED_TRAINING_COLUMNS,
    TARGET_COLUMN,
    FeaturePreprocessor,
    PreparedDataset,
    export_split_manifest,
    prepare_kepler_dataset,
)
from src.training.config import TrainingConfig
from src.training.persistence import load_model_bundle, save_model_bundle
from src.training.trainer import EpochCallback, Trainer
from src.validation.metrics import compute_classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "kepler_koi.csv"
SCIENTIFIC_RESULTS_PATH = (
    PROJECT_ROOT / "experiments" / "stage04" / "scientific_validation_results.json"
)
SCIENTIFIC_MODEL_DIRECTORY = (
    PROJECT_ROOT / "experiments" / "stage04" / "selected_model"
)
LAB_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "developer-lab"
LAB_SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "developer_lab_split_manifest.csv"
)

FloatArray = NDArray[np.float64]


@dataclass
class LabModelState:
    """A loaded engine plus the exact preprocessing and provenance it needs."""

    model: MLP
    preprocessor: FeaturePreprocessor
    config: TrainingConfig
    metrics: dict[str, Any]
    source: str
    source_path: Path
    scientific_validation: bool


@dataclass(frozen=True)
class InferenceResult:
    """One real astronomical record's ANN-01 decision and neural state."""

    object_id: str
    group_id: str
    original_disposition: str
    score: float
    predicted_class: str
    threshold: float
    raw_features: dict[str, float | None]
    normalized_features: dict[str, float]
    activations: tuple[FloatArray, ...]
    pre_activations: tuple[FloatArray, ...]


def scientific_validation_results(
    path: Path = SCIENTIFIC_RESULTS_PATH,
) -> dict[str, Any] | None:
    """Return Stage 04 results when the validated artifact is available."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"dataset", "protocol", "final_test", "benchmark"}
    return payload if required.issubset(payload) else None


def _scientific_training_config(
    experiment: dict[str, Any],
    results: dict[str, Any],
) -> TrainingConfig:
    protocol = results["protocol"]
    return TrainingConfig(
        engine_name="ANN-01",
        layer_sizes=tuple(experiment["layer_sizes"]),
        learning_rate=float(experiment["learning_rate"]),
        epochs=int(experiment["epochs"]),
        batch_size=int(experiment["batch_size"]),
        seed=int(experiment["seed"]),
        train_ratio=float(protocol["train_ratio"]),
        validation_ratio=float(protocol["validation_ratio"]),
        test_ratio=float(protocol["test_ratio"]),
        classification_threshold=float(protocol["classification_threshold"]),
    )


def load_scientific_model(
    directory: Path = SCIENTIFIC_MODEL_DIRECTORY,
    results_path: Path = SCIENTIFIC_RESULTS_PATH,
) -> LabModelState | None:
    """Load the selected Stage 04 model without modifying its artifacts."""
    directory = Path(directory)
    results = scientific_validation_results(results_path)
    required = (
        directory / "model.npz",
        directory / "preprocessor.json",
        directory / "experiment_config.json",
        directory / "history.json",
        directory / "test_metrics.json",
    )
    if results is None or not all(path.is_file() for path in required):
        return None

    experiment = json.loads(required[2].read_text(encoding="utf-8"))
    config = _scientific_training_config(experiment, results)
    model = MLP.load(
        required[0],
        seed=config.seed,
        engine_name=config.engine_name,
    )
    model.training_history = json.loads(required[3].read_text(encoding="utf-8"))
    preprocessor = FeaturePreprocessor.load(required[1])
    metrics = json.loads(required[4].read_text(encoding="utf-8"))
    return LabModelState(
        model=model,
        preprocessor=preprocessor,
        config=config,
        metrics=metrics,
        source="Stage 04 selected scientific model",
        source_path=directory,
        scientific_validation=True,
    )


def load_latest_lab_model(
    directory: Path = LAB_MODEL_DIRECTORY,
) -> LabModelState | None:
    """Load the most recently persisted Developer Lab model, if present."""
    directory = Path(directory)
    required = (
        directory / "model.npz",
        directory / "preprocessor.json",
        directory / "config.json",
        directory / "history.json",
        directory / "metrics.json",
    )
    if not all(path.is_file() for path in required):
        return None
    bundle = load_model_bundle(directory)
    return LabModelState(
        model=bundle.model,
        preprocessor=bundle.preprocessor,
        config=bundle.config,
        metrics=bundle.metrics,
        source="Latest Developer Lab training",
        source_path=directory,
        scientific_validation=False,
    )


def load_dataset_frame(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the real KOI identifiers, labels, and ANN-01 input columns."""
    return pd.read_csv(Path(path), usecols=list(REQUIRED_TRAINING_COLUMNS))


def dataset_overview(
    frame: pd.DataFrame,
    path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    """Summarize the local NASA table without generating replacement data."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    dispositions = (
        frame[TARGET_COLUMN].value_counts(dropna=False).astype(int).to_dict()
    )
    missing = frame.loc[:, FEATURE_COLUMNS].isna().sum().astype(int).to_dict()
    return {
        "records": int(len(frame)),
        "stars": int(frame[GROUP_COLUMN].nunique()),
        "objects": int(frame[OBJECT_ID_COLUMN].nunique()),
        "dispositions": dispositions,
        "missing_values": missing,
        "sha256": digest,
        "version_id": f"sha256:{digest[:16]}",
        "path": str(Path(path).resolve()),
    }


def infer_record(
    state: LabModelState,
    record: pd.Series | pd.DataFrame,
) -> InferenceResult:
    """Run a retained forward pass for one genuine KOI record."""
    row_frame = record.to_frame().T if isinstance(record, pd.Series) else record.copy()
    if len(row_frame) != 1:
        raise ValueError("Individual inference requires exactly one record")
    transformed = state.preprocessor.transform(row_frame)
    score = float(state.model.forward(transformed, retain_cache=True)[0, 0])
    threshold = state.config.classification_threshold
    predicted_class = (
        "CONFIRMED-LIKE" if score >= threshold else "FALSE-POSITIVE-LIKE"
    )
    raw_features: dict[str, float | None] = {}
    for feature in FEATURE_COLUMNS:
        value = pd.to_numeric(row_frame.iloc[0][feature], errors="coerce")
        raw_features[feature] = None if pd.isna(value) else float(value)
    return InferenceResult(
        object_id=str(row_frame.iloc[0][OBJECT_ID_COLUMN]),
        group_id=str(row_frame.iloc[0][GROUP_COLUMN]),
        original_disposition=str(row_frame.iloc[0][TARGET_COLUMN]),
        score=score,
        predicted_class=predicted_class,
        threshold=threshold,
        raw_features=raw_features,
        normalized_features={
            feature: float(value)
            for feature, value in zip(FEATURE_COLUMNS, transformed[0], strict=True)
        },
        activations=tuple(values.copy() for values in state.model.activations),
        pre_activations=tuple(
            values.copy() for values in state.model.pre_activations
        ),
    )


def _partition_summary(dataset: PreparedDataset) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, partition in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test_reserved", dataset.test),
        ("candidates", dataset.candidates),
    ):
        summary[name] = {
            "records": partition.size,
            "stars": partition.group_count,
        }
    return summary


def train_developer_model(
    config: TrainingConfig,
    data_path: Path = DEFAULT_DATA_PATH,
    output_directory: Path = LAB_MODEL_DIRECTORY,
    manifest_path: Path = LAB_SPLIT_MANIFEST_PATH,
    epoch_callback: EpochCallback | None = None,
) -> LabModelState:
    """Train ANN-01 on real KOI train/validation data and persist the result.

    The reserved test partition is created for split integrity but deliberately
    not evaluated by this interactive workflow.
    """
    dataset = prepare_kepler_dataset(
        data_path,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    model = MLP(
        layer_sizes=config.layer_sizes,
        seed=config.seed,
        engine_name=config.engine_name,
    )
    trainer = Trainer(config)
    trainer.fit(
        model,
        dataset.train,
        dataset.validation,
        epoch_callback=epoch_callback,
    )
    validation_scores = model.predict_proba(dataset.validation.features).reshape(-1)
    assert dataset.validation.targets is not None
    validation_metrics = compute_classification_metrics(
        dataset.validation.targets.reshape(-1),
        validation_scores,
        threshold=config.classification_threshold,
    )
    metrics = {
        "workflow": "ASTRAL Developer Lab",
        "test_accessed": False,
        "validation": validation_metrics,
        "final_epoch": model.training_history[-1],
        "splits": _partition_summary(dataset),
    }
    save_model_bundle(
        output_directory,
        model,
        dataset.preprocessor,
        config,
        metrics,
    )
    export_split_manifest(dataset, manifest_path)

    snapshot_size = min(config.snapshot_sample_size, dataset.validation.size)
    snapshot_features = dataset.validation.features[:snapshot_size]
    snapshot_targets = dataset.validation.targets[:snapshot_size]
    model.forward(snapshot_features, retain_cache=True)
    model.backward(snapshot_targets)
    model.export_snapshot(
        Path(output_directory) / "network_snapshot.json",
        metadata={
            "context": "developer_lab_post_training_validation_sample",
            "object_ids": dataset.validation.object_ids[:snapshot_size].tolist(),
            "test_accessed": False,
        },
    )
    return LabModelState(
        model=model,
        preprocessor=dataset.preprocessor,
        config=config,
        metrics=metrics,
        source="Latest Developer Lab training",
        source_path=Path(output_directory),
        scientific_validation=False,
    )
