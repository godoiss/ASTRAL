"""Validated configuration for bounded MAST light-curve acquisition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObservatoryConfig:
    """Acquisition, processing, and ANN-02 representation parameters."""

    mission: str = "Kepler"
    author: str = "Kepler"
    cadence: str = "long"
    quarter: int | None = 3
    max_products: int = 1
    timeout_seconds: int = 60
    retries: int = 2
    quality_bitmask: int = 1130799
    flatten_window_length: int = 101
    flatten_polyorder: int = 2
    transit_mask_duration_multiplier: float = 1.5
    local_view_duration_multiplier: float = 3.0
    global_view_size: int = 2001
    local_view_size: int = 201

    def __post_init__(self) -> None:
        if self.mission not in {"Kepler", "TESS"}:
            raise ValueError("mission must be Kepler or TESS")
        if self.cadence not in {"long", "short", "fast"}:
            raise ValueError("cadence must be long, short, or fast")
        if self.max_products < 1 or self.max_products > 8:
            raise ValueError("max_products must remain between 1 and 8")
        if self.timeout_seconds < 5 or self.retries < 1:
            raise ValueError("timeout_seconds and retries must be positive")
        if self.flatten_window_length < 5 or self.flatten_window_length % 2 == 0:
            raise ValueError("flatten_window_length must be odd and at least 5")
        if self.flatten_polyorder < 1:
            raise ValueError("flatten_polyorder must be positive")
        if self.global_view_size < 101 or self.local_view_size < 51:
            raise ValueError("fixed views are too small for scientific inspection")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObservatoryConfig:
        """Construct and validate configuration from a mapping."""
        return cls(**payload)

    @classmethod
    def load(cls, path: Path) -> ObservatoryConfig:
        """Load configuration from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
