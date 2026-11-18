"""Validated configuration contracts for the ANN-02 workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetBuildConfig:
    """Bounded, resumable acquisition and quality criteria."""

    pilot_size: int = 8
    candidate_size: int = 1
    seed: int = 2026
    quarter: int | None = 3
    max_products_per_object: int = 1
    max_new_downloads: int = 5
    storage_limit_mb: float = 250.0
    per_object_timeout_seconds: int = 120
    max_period_days: float = 30.0
    min_global_coverage: float = 0.20
    min_local_coverage: float = 0.30
    min_observed_transits: int = 1

    def __post_init__(self) -> None:
        if not 8 <= self.pilot_size <= 200:
            raise ValueError("pilot_size must be between 8 and 200")
        if self.pilot_size % 4:
            raise ValueError("pilot_size must be divisible by four for balanced partitions")
        if not 0 <= self.candidate_size <= 20:
            raise ValueError("candidate_size must be between 0 and 20")
        if not 1 <= self.max_products_per_object <= 8:
            raise ValueError("max_products_per_object must be between 1 and 8")
        if not 0 <= self.max_new_downloads <= 200:
            raise ValueError("max_new_downloads must be between 0 and 200")
        if self.storage_limit_mb < 10:
            raise ValueError("storage_limit_mb must be at least 10")
        if not 30 <= self.per_object_timeout_seconds <= 900:
            raise ValueError("per_object_timeout_seconds must be between 30 and 900")
        if not 0.0 <= self.min_global_coverage <= 1.0:
            raise ValueError("min_global_coverage must be in [0, 1]")
        if not 0.0 <= self.min_local_coverage <= 1.0:
            raise ValueError("min_local_coverage must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ANN02Config:
    """Lightweight two-branch one-dimensional CNN architecture."""

    input_channels: int = 2
    branch_channels: tuple[int, int] = (8, 16)
    kernel_size: int = 5
    global_pool_size: int = 4
    local_pool_size: int = 2
    fused_features: int = 16
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if self.input_channels != 2:
            raise ValueError("ANN-02 requires flux and availability-mask channels")
        if len(self.branch_channels) != 2 or min(self.branch_channels) < 1:
            raise ValueError("branch_channels must contain two positive values")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["branch_channels"] = list(self.branch_channels)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ANN02Config":
        values = dict(payload)
        values["branch_channels"] = tuple(values.get("branch_channels", (8, 16)))
        return cls(**values)


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible ANN-02 optimization settings."""

    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 8
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 1e-4
    seed: int = 2026
    threshold: float = 0.5
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("learning_rate, epochs, and batch_size must be positive")
        if self.patience < 1 or self.min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_stage06_config(path: Path) -> tuple[DatasetBuildConfig, ANN02Config, TrainingConfig]:
    """Load and validate the three Stage 06 configuration sections."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        DatasetBuildConfig(**payload.get("dataset", {})),
        ANN02Config.from_dict(payload.get("model", {})),
        TrainingConfig(**payload.get("training", {})),
    )
