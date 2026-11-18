"""Configuration contracts for controlled scientific validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExperimentSpec:
    """One controlled neural experiment."""

    experiment_id: str
    label: str
    category: str
    layer_sizes: tuple[int, ...]
    learning_rate: float
    epochs: int
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if self.layer_sizes[0] != 6 or self.layer_sizes[-1] != 1:
            raise ValueError("Experiments require six inputs and one binary output")
        if len(self.layer_sizes) < 3:
            raise ValueError("Experiments require at least one hidden layer")
        if any(size <= 0 for size in self.layer_sizes):
            raise ValueError("Layer sizes must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("Learning rate must be positive")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Epochs and batch size must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready experiment definition."""
        return {
            "experiment_id": self.experiment_id,
            "label": self.label,
            "category": self.category,
            "layer_sizes": list(self.layer_sizes),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentSpec:
        """Build a validated experiment definition."""
        normalized = dict(payload)
        normalized["layer_sizes"] = tuple(normalized["layer_sizes"])
        return cls(**normalized)

    def with_seed(self, seed: int, experiment_id: str) -> ExperimentSpec:
        """Clone the hyperparameters for a controlled seed-stability run."""
        return ExperimentSpec(
            experiment_id=experiment_id,
            label=f"{self.label} / seed {seed}",
            category="seed_stability",
            layer_sizes=self.layer_sizes,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=seed,
        )


@dataclass(frozen=True)
class CampaignConfig:
    """Frozen protocol for one Stage 04 validation campaign."""

    campaign_id: str
    split_seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    classification_threshold: float
    selection_metric: str
    stability_seeds: tuple[int, ...]
    plot_dpi: int
    experiments: tuple[ExperimentSpec, ...]
    benchmarks: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if not np.isclose(
            self.train_ratio + self.validation_ratio + self.test_ratio,
            1.0,
        ):
            raise ValueError("Campaign split ratios must sum to one")
        if not 0.0 < self.classification_threshold < 1.0:
            raise ValueError("Classification threshold must be between zero and one")
        if self.selection_metric != "pr_auc":
            raise ValueError("Stage 04 protocol pre-registers PR-AUC for selection")
        if len(set(self.stability_seeds)) != len(self.stability_seeds):
            raise ValueError("Stability seeds must be unique")
        if not self.experiments:
            raise ValueError("At least one controlled experiment is required")
        identifiers = [item.experiment_id for item in self.experiments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Experiment identifiers must be unique")

    @classmethod
    def load(cls, path: Path) -> CampaignConfig:
        """Load and validate the pre-registered JSON protocol."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            campaign_id=payload["campaign_id"],
            split_seed=int(payload["split_seed"]),
            train_ratio=float(payload["train_ratio"]),
            validation_ratio=float(payload["validation_ratio"]),
            test_ratio=float(payload["test_ratio"]),
            classification_threshold=float(payload["classification_threshold"]),
            selection_metric=payload["selection_metric"],
            stability_seeds=tuple(int(seed) for seed in payload["stability_seeds"]),
            plot_dpi=int(payload["plot_dpi"]),
            experiments=tuple(
                ExperimentSpec.from_dict(item) for item in payload["experiments"]
            ),
            benchmarks=payload["benchmarks"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete protocol as a JSON-ready dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "split_seed": self.split_seed,
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
            "classification_threshold": self.classification_threshold,
            "selection_metric": self.selection_metric,
            "stability_seeds": list(self.stability_seeds),
            "plot_dpi": self.plot_dpi,
            "experiments": [item.to_dict() for item in self.experiments],
            "benchmarks": self.benchmarks,
        }

