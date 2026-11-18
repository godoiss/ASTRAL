"""Persistence of ANN-01 weights, preprocessing, history, and metadata."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.mlp import MLP
from src.data.preprocessing import FeaturePreprocessor
from src.training.config import TrainingConfig


MODEL_FILENAME = "model.npz"
PREPROCESSOR_FILENAME = "preprocessor.json"
CONFIG_FILENAME = "config.json"
HISTORY_FILENAME = "history.json"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class ModelBundle:
    """A restored model and all state needed for deterministic inference."""

    model: MLP
    preprocessor: FeaturePreprocessor
    config: TrainingConfig
    metrics: dict[str, Any]


def _write_json_atomic(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".json.tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def save_model_bundle(
    directory: Path,
    model: MLP,
    preprocessor: FeaturePreprocessor,
    config: TrainingConfig,
    metrics: dict[str, Any],
) -> Path:
    """Persist everything required to reproduce ANN-01 inference."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / MODEL_FILENAME)
    preprocessor.save(directory / PREPROCESSOR_FILENAME)
    _write_json_atomic(directory / CONFIG_FILENAME, config.to_dict())
    _write_json_atomic(directory / HISTORY_FILENAME, model.training_history)
    _write_json_atomic(directory / METRICS_FILENAME, metrics)
    return directory


def load_model_bundle(directory: Path) -> ModelBundle:
    """Restore a complete ANN-01 inference bundle."""
    directory = Path(directory)
    config = TrainingConfig.load(directory / CONFIG_FILENAME)
    model = MLP.load(
        directory / MODEL_FILENAME,
        seed=config.seed,
        engine_name=config.engine_name,
    )
    model.training_history = json.loads(
        (directory / HISTORY_FILENAME).read_text(encoding="utf-8")
    )
    preprocessor = FeaturePreprocessor.load(directory / PREPROCESSOR_FILENAME)
    metrics = json.loads((directory / METRICS_FILENAME).read_text(encoding="utf-8"))
    return ModelBundle(
        model=model,
        preprocessor=preprocessor,
        config=config,
        metrics=metrics,
    )

