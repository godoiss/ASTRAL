"""Controlled ANN-02 campaigns using training/validation only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.deep_learning.config import ANN02Config, TrainingConfig
from src.deep_learning.dataset import DEFAULT_DATASET_PATH
from src.deep_learning.training import activate_ann02_model, train_ann02, validate_dataset_integrity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE07_EXPERIMENTS = PROJECT_ROOT / "experiments" / "stage07"
DEFAULT_STAGE07_MODELS = PROJECT_ROOT / "models" / "ann02" / "stage07"


@dataclass(frozen=True)
class ANN02Trial:
    """One controlled architecture and optimizer configuration."""

    name: str
    model: ANN02Config
    training: TrainingConfig
    changed_factor: str


def default_trials() -> tuple[ANN02Trial, ...]:
    """Return a compact CPU-suitable campaign with one-factor comparisons."""
    common = dict(epochs=60, batch_size=16, patience=10, seed=2027, device="cpu")
    return (
        ANN02Trial("baseline", ANN02Config(), TrainingConfig(learning_rate=1e-3, **common), "reference"),
        ANN02Trial("low_learning_rate", ANN02Config(), TrainingConfig(learning_rate=3e-4, **common), "learning_rate=0.0003"),
        ANN02Trial(
            "wider_branches",
            ANN02Config(branch_channels=(12, 24), fused_features=24, dropout=0.20),
            TrainingConfig(learning_rate=1e-3, **common),
            "branch_channels=(12,24), fused_features=24",
        ),
        ANN02Trial(
            "stronger_regularization",
            ANN02Config(dropout=0.35),
            TrainingConfig(learning_rate=1e-3, weight_decay=5e-4, **common),
            "dropout=0.35, weight_decay=0.0005",
        ),
        ANN02Trial(
            "seed_stability",
            ANN02Config(),
            TrainingConfig(learning_rate=1e-3, **{**common, "seed": 97}),
            "random_seed=97",
        ),
    )


def run_ann02_campaign(
    trials: tuple[ANN02Trial, ...] | None = None,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    activate_best: bool = True,
) -> dict[str, Any]:
    """Train controlled trials, select by validation BCE, and never access test data."""
    integrity = validate_dataset_integrity(dataset_path)
    if integrity["test_records"]:
        raise ValueError("Stage 07 development campaigns cannot materialize test records")
    trials = trials or default_trials()
    campaign_id = datetime.now(UTC).strftime("stage07-%Y%m%dT%H%M%SZ")
    model_root = DEFAULT_STAGE07_MODELS / campaign_id
    run_records: list[dict[str, Any]] = []
    for trial in trials:
        model_directory = model_root / trial.name
        bundle = train_ann02(
            trial.model,
            trial.training,
            dataset_path=dataset_path,
            model_directory=model_directory,
            experiment_directory=DEFAULT_STAGE07_EXPERIMENTS / "runs",
        )
        metrics = bundle.experiment["validation_metrics"]
        run_records.append(
            {
                "name": trial.name,
                "changed_factor": trial.changed_factor,
                "experiment_id": bundle.experiment["experiment_id"],
                "model_directory": str(model_directory.resolve()),
                "architecture": bundle.model.architecture(),
                "training_config": trial.training.to_dict(),
                "best_epoch": bundle.experiment["best_epoch"],
                "epochs_completed": bundle.experiment["epochs_completed"],
                "duration_seconds": bundle.experiment["duration_seconds"],
                "validation_metrics": metrics,
            }
        )
    selected = min(run_records, key=lambda item: item["validation_metrics"]["binary_cross_entropy"])
    if activate_best:
        activate_ann02_model(Path(selected["model_directory"]))
    payload = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_metric": "validation binary cross-entropy",
        "selected_trial": selected["name"],
        "selected_experiment_id": selected["experiment_id"],
        "dataset_integrity": integrity,
        "runs": run_records,
        "guardrails": {
            "test_accessed": False,
            "candidates_used_for_training": False,
            "selection_used_train_and_validation_only": True,
            "definitive_scientific_validation": False,
        },
        "limitations": [
            "The validation partition is reused across a small hyperparameter campaign and can itself become a source of selection overfitting.",
            "The campaign is CPU-bounded and is not an exhaustive architecture search.",
            "The selected checkpoint requires future evaluation on the still-frozen test partition before any final performance claim.",
        ],
    }
    campaign_directory = DEFAULT_STAGE07_EXPERIMENTS / "campaigns"
    campaign_directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    (campaign_directory / f"{campaign_id}.json").write_text(text, encoding="utf-8")
    (DEFAULT_STAGE07_EXPERIMENTS / "latest_campaign.json").write_text(text, encoding="utf-8")
    return payload
