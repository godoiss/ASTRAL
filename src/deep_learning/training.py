"""Reproducible local training, validation, persistence, and inference for ANN-02."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.deep_learning.config import ANN02Config, TrainingConfig
from src.deep_learning.dataset import DEFAULT_DATASET_PATH, KeplerSignalDataset
from src.deep_learning.hardware import hardware_report, select_device
from src.deep_learning.model import DualBranchCNN
from src.deep_learning.observability import build_snapshot, export_snapshot
from src.observatory.catalog import load_observatory_catalog
from src.validation.metrics import compute_classification_metrics, compute_curve_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "ann02"
DEFAULT_RUNS_DIRECTORY = DEFAULT_MODEL_DIRECTORY / "runs"
DEFAULT_ACTIVE_MODEL_POINTER = DEFAULT_MODEL_DIRECTORY / "active_model.json"
DEFAULT_EXPERIMENT_DIRECTORY = PROJECT_ROOT / "experiments" / "stage06"
EpochCallback = Callable[[dict[str, float]], None]


@dataclass
class ANN02Bundle:
    """Loaded ANN-02 with its exact architecture, training metadata, and device."""

    model: DualBranchCNN
    model_config: ANN02Config
    training_config: TrainingConfig
    experiment: dict[str, Any]
    device: torch.device


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_torch(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".pt.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _loader(dataset: KeplerSignalDataset, config: TrainingConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _epoch(
    model: DualBranchCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    record_count = 0
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        global_view = batch["global_view"].to(device)
        local_view = batch["local_view"].to(device)
        labels = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(global_view, local_view)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_records = int(labels.numel())
        weighted_loss += float(loss.detach().cpu()) * batch_records
        record_count += batch_records
        targets.append(labels.detach().cpu().numpy())
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    if not record_count:
        raise ValueError("ANN-02 cannot evaluate an empty dataset split")
    return weighted_loss / record_count, np.concatenate(targets), np.concatenate(scores)


def validate_dataset_integrity(path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    """Verify shapes, labels, class coverage, and KIC separation before training."""
    with np.load(Path(path), allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    required = {
        "object_ids", "kic", "splits", "class_labels", "global_flux", "global_mask",
        "local_flux", "local_mask",
    }
    if not required.issubset(values):
        raise ValueError(f"ANN-02 dataset is missing fields: {sorted(required - values.keys())}")
    records = len(values["object_ids"])
    if len(np.unique(values["object_ids"].astype(str))) != records:
        raise ValueError("ANN-02 object identifiers must be unique")
    if values["global_flux"].shape != (records, 2001):
        raise ValueError("Global view must have shape (records, 2001)")
    if values["local_flux"].shape != (records, 201):
        raise ValueError("Local view must have shape (records, 201)")
    if not np.array_equal(np.isnan(values["global_flux"]), ~values["global_mask"]):
        raise ValueError("Global NaN values and availability mask disagree")
    if not np.array_equal(np.isnan(values["local_flux"]), ~values["local_mask"]):
        raise ValueError("Local NaN values and availability mask disagree")
    allowed_splits = {"train", "validation", "test", "candidate"}
    observed_splits = set(values["splits"].astype(str))
    if not observed_splits.issubset(allowed_splits):
        raise ValueError(f"Unsupported ANN-02 splits: {sorted(observed_splits - allowed_splits)}")
    labelled_mask = np.isin(values["splits"].astype(str), ["train", "validation", "test"])
    if not np.all(np.isin(values["class_labels"][labelled_mask], [0, 1])):
        raise ValueError("Train, validation, and test records must use binary labels 0 or 1")
    candidate_mask = values["splits"].astype(str) == "candidate"
    if not np.all(values["class_labels"][candidate_mask] == -1):
        raise ValueError("Candidate records must remain unlabelled with class_label=-1")

    assignments: dict[int, set[str]] = {}
    for kic, split, label in zip(values["kic"], values["splits"], values["class_labels"], strict=True):
        assignments.setdefault(int(kic), set()).add(str(split))
    conflicts = {kic: splits for kic, splits in assignments.items() if len(splits) > 1}
    if conflicts:
        raise ValueError(f"KIC leakage detected: {conflicts}")
    frozen_test_overlap: list[int] = []
    if Path(path).resolve() == DEFAULT_DATASET_PATH.resolve():
        catalog = load_observatory_catalog()
        frozen_test_stars = set(
            catalog.loc[catalog["split"].eq("test"), "kepid"].dropna().astype(int)
        )
        materialized_stars = set(values["kic"].astype(int).tolist())
        frozen_test_overlap = sorted(materialized_stars & frozen_test_stars)
        if frozen_test_overlap:
            raise ValueError(
                f"Frozen Stage 04 test-star leakage detected: {frozen_test_overlap}"
            )
    summary: dict[str, Any] = {
        "records": records,
        "stars": int(np.unique(values["kic"]).size),
        "splits": {},
        "test_records": 0,
        "frozen_test_star_overlap": frozen_test_overlap,
    }
    for split in ("train", "validation", "test", "candidate"):
        mask = values["splits"].astype(str) == split
        labels = values["class_labels"][mask]
        summary["splits"][split] = {
            "records": int(mask.sum()),
            "stars": int(np.unique(values["kic"][mask]).size),
            "class_0": int(np.sum(labels == 0)),
            "class_1": int(np.sum(labels == 1)),
        }
    summary["test_records"] = summary["splits"]["test"]["records"]
    for split in ("train", "validation"):
        if summary["splits"][split]["class_0"] == 0 or summary["splits"][split]["class_1"] == 0:
            raise ValueError(f"{split} split must contain both labelled classes")
    return summary


def train_ann02(
    model_config: ANN02Config,
    training_config: TrainingConfig,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_directory: Path | None = None,
    experiment_directory: Path = DEFAULT_EXPERIMENT_DIRECTORY,
    epoch_callback: EpochCallback | None = None,
) -> ANN02Bundle:
    """Optimize on train only, monitor validation, and persist a versioned best model."""
    started = time.perf_counter()
    experiment_id = datetime.now(UTC).strftime("ann02-%Y%m%dT%H%M%SZ")
    versioned_persistence = model_directory is None
    integrity = validate_dataset_integrity(dataset_path)
    if integrity["test_records"]:
        raise ValueError("Stage 06 pilot dataset must not materialize frozen test records")
    set_reproducibility(training_config.seed)
    device = select_device(training_config.device)
    train_dataset = KeplerSignalDataset(dataset_path, "train")
    validation_dataset = KeplerSignalDataset(dataset_path, "validation")
    model = DualBranchCNN(model_config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_targets, train_scores = _epoch(
            model, _loader(train_dataset, training_config, True), criterion, device, optimizer
        )
        validation_loss, validation_targets, validation_scores = _epoch(
            model, _loader(validation_dataset, training_config, False), criterion, device, None
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": float(accuracy_score(train_targets, train_scores >= training_config.threshold)),
            "validation_accuracy": float(
                accuracy_score(validation_targets, validation_scores >= training_config.threshold)
            ),
        }
        history.append(row)
        if epoch_callback:
            epoch_callback(dict(row))
        if validation_loss < best_loss - training_config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_loss, validation_targets, validation_scores = _epoch(
        model, _loader(validation_dataset, training_config, False), criterion, device, None
    )
    metrics = compute_classification_metrics(
        validation_targets, validation_scores, threshold=training_config.threshold
    )
    curves = compute_curve_data(validation_targets, validation_scores)
    duration = float(time.perf_counter() - started)
    if model_directory is None:
        model_directory = DEFAULT_RUNS_DIRECTORY / experiment_id
    model_directory = Path(model_directory)
    experiment = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "architecture": model.architecture(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "hardware": hardware_report(),
        "device_used": str(device),
        "duration_seconds": duration,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < training_config.epochs,
        "history": history,
        "validation_metrics": metrics,
        "validation_curves": curves,
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "sha256": _sha256(Path(dataset_path)),
            "integrity": integrity,
        },
        "persistence": {
            "model_directory": str(model_directory.resolve()),
            "versioned": versioned_persistence,
            "previous_model_preserved": versioned_persistence,
        },
        "scientific_guardrails": {
            "test_accessed": False,
            "test_used_for_hyperparameters": False,
            "candidates_used_for_training": False,
            "definitive_scientific_validation": False,
        },
    }
    _atomic_torch(model_directory / "model.pt", best_state)
    _atomic_json(model_directory / "model_config.json", model_config.to_dict())
    _atomic_json(model_directory / "training_config.json", training_config.to_dict())
    _atomic_json(model_directory / "experiment.json", experiment)
    _atomic_json(Path(experiment_directory) / f"{experiment_id}.json", experiment)
    _atomic_json(Path(experiment_directory) / "latest_experiment.json", experiment)

    sample = validation_dataset[0]
    model.train(False)
    model.zero_grad(set_to_none=True)
    global_view = sample["global_view"].unsqueeze(0).to(device)
    local_view = sample["local_view"].unsqueeze(0).to(device)
    label = sample["label"].unsqueeze(0).to(device)
    logits = model(global_view, local_view, retain_observability=True)
    loss = criterion(logits, label)
    loss.backward()
    snapshot = build_snapshot(model, str(sample["object_id"]), device, duration)
    export_snapshot(model_directory / "observability_snapshot.json", snapshot)
    if versioned_persistence:
        relative_directory = model_directory.resolve().relative_to(PROJECT_ROOT.resolve())
        _atomic_json(
            DEFAULT_ACTIVE_MODEL_POINTER,
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "model_directory": relative_directory.as_posix(),
                "activated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    return ANN02Bundle(model, model_config, training_config, experiment, device)


def active_ann02_model_directory() -> Path:
    """Resolve the active versioned model, falling back to the original pilot bundle."""
    if not DEFAULT_ACTIVE_MODEL_POINTER.is_file():
        return DEFAULT_MODEL_DIRECTORY
    try:
        payload = json.loads(DEFAULT_ACTIVE_MODEL_POINTER.read_text(encoding="utf-8"))
        directory = (PROJECT_ROOT / str(payload["model_directory"])).resolve()
        directory.relative_to(PROJECT_ROOT.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return DEFAULT_MODEL_DIRECTORY
    return directory if (directory / "model.pt").is_file() else DEFAULT_MODEL_DIRECTORY


def load_ann02(
    model_directory: Path | None = None,
    device_preference: str = "auto",
) -> ANN02Bundle | None:
    """Load a persisted ANN-02 bundle, or return None when it is unavailable."""
    directory = active_ann02_model_directory() if model_directory is None else Path(model_directory)
    required = [
        directory / "model.pt",
        directory / "model_config.json",
        directory / "training_config.json",
        directory / "experiment.json",
    ]
    if not all(path.is_file() for path in required):
        return None
    model_config = ANN02Config.from_dict(json.loads(required[1].read_text(encoding="utf-8")))
    training_config = TrainingConfig(**json.loads(required[2].read_text(encoding="utf-8")))
    experiment = json.loads(required[3].read_text(encoding="utf-8"))
    device = select_device(device_preference)
    model = DualBranchCNN(model_config).to(device)
    state = torch.load(required[0], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return ANN02Bundle(model, model_config, training_config, experiment, device)


def infer_ann02(bundle: ANN02Bundle, sample: dict[str, Any]) -> dict[str, Any]:
    """Run one real light-curve classification and return its retained feature maps."""
    started = time.perf_counter()
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(
            sample["global_view"].unsqueeze(0).to(bundle.device),
            sample["local_view"].unsqueeze(0).to(bundle.device),
            retain_observability=True,
        )
        score = float(torch.sigmoid(logits)[0].cpu())
    threshold = bundle.training_config.threshold
    return {
        "object_id": str(sample["object_id"]),
        "kic": int(sample["kic"]),
        "score": score,
        "predicted_class": "CONFIRMED-LIKE" if score >= threshold else "FALSE-POSITIVE-LIKE",
        "threshold": threshold,
        "device": str(bundle.device),
        "elapsed_seconds": float(time.perf_counter() - started),
        "feature_maps": {
            name: values[0].detach().cpu().numpy()
            for name, values in bundle.model.last_activations.items()
            if name.endswith("relu_2")
        },
    }
