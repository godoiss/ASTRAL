"""Hyperparameter configuration for ANN-01 training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    """Validated, serializable ANN-01 training configuration."""

    engine_name: str = "ANN-01"
    layer_sizes: tuple[int, ...] = (6, 8, 4, 1)
    learning_rate: float = 0.01
    epochs: int = 120
    batch_size: int = 64
    seed: int = 42
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    classification_threshold: float = 0.5
    log_every: int = 10
    snapshot_sample_size: int = 16

    def __post_init__(self) -> None:
        if tuple(self.layer_sizes) != (6, 8, 4, 1):
            raise ValueError("ANN-01 architecture must be exactly 6-8-4-1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not np.isclose(
            self.train_ratio + self.validation_ratio + self.test_ratio,
            1.0,
        ):
            raise ValueError("Dataset split ratios must sum to one")
        if any(
            ratio <= 0.0
            for ratio in (self.train_ratio, self.validation_ratio, self.test_ratio)
        ):
            raise ValueError("Dataset split ratios must be positive")
        if not 0.0 < self.classification_threshold < 1.0:
            raise ValueError("classification_threshold must be between zero and one")
        if self.log_every <= 0 or self.snapshot_sample_size <= 0:
            raise ValueError("log_every and snapshot_sample_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        payload = asdict(self)
        payload["layer_sizes"] = list(self.layer_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        """Build a validated config from a dictionary."""
        normalized = dict(payload)
        normalized["layer_sizes"] = tuple(normalized["layer_sizes"])
        return cls(**normalized)

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        """Load hyperparameters from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

