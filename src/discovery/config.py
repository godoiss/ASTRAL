"""Validated configuration for bounded astronomical discovery campaigns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectionConfig:
    """BLS grid, processing, and transparent screening thresholds."""

    minimum_period_days: float = 1.0
    maximum_period_days: float = 30.0
    durations_hours: tuple[float, ...] = (1.5, 2.5, 4.0, 6.0)
    period_samples: int = 4000
    bls_oversample: int = 10
    minimum_depth_snr: float = 6.0
    minimum_observed_events: int = 2
    minimum_event_coverage: float = 0.5
    minimum_points_per_event: int = 2
    detrend_window_days: float = 2.0
    quality_bitmask: int = 1130799
    phase_bins: int = 120
    alias_ratio_tolerance: float = 0.02
    odd_even_sigma_threshold: float = 3.0
    max_reported_aliases: int = 8

    def __post_init__(self) -> None:
        if self.minimum_period_days <= 0 or self.maximum_period_days <= self.minimum_period_days:
            raise ValueError("period bounds must be positive and increasing")
        if not self.durations_hours or any(value <= 0 for value in self.durations_hours):
            raise ValueError("durations_hours must contain positive values")
        if max(self.durations_hours) / 24.0 >= self.minimum_period_days:
            raise ValueError("all transit durations must be shorter than the minimum period")
        if not 500 <= self.period_samples <= 50000:
            raise ValueError("period_samples must remain between 500 and 50000")
        if self.bls_oversample < 1 or self.bls_oversample > 50:
            raise ValueError("bls_oversample must remain between 1 and 50")
        if self.minimum_depth_snr <= 0:
            raise ValueError("minimum_depth_snr must be positive")
        if not 0 < self.minimum_event_coverage <= 1:
            raise ValueError("minimum_event_coverage must be in (0, 1]")
        if self.phase_bins < 20:
            raise ValueError("phase_bins must be at least 20")


@dataclass(frozen=True)
class CampaignConfig:
    """Selection, acquisition, and resource limits for one campaign."""

    mission: str = "Kepler"
    strategy: str = "unlabeled_candidates_cache_first"
    sample_size: int = 10
    random_seed: int = 10103
    quarter: int | None = 3
    cadence: str = "long"
    max_products_per_star: int = 1
    max_new_downloads: int = 9
    storage_limit_mb: float = 350.0
    per_star_timeout_seconds: int = 90
    allow_network: bool = False

    def __post_init__(self) -> None:
        if self.mission != "Kepler":
            raise ValueError("Stage 10 campaigns currently support Kepler only")
        if self.strategy != "unlabeled_candidates_cache_first":
            raise ValueError("unsupported Stage 10 selection strategy")
        if not 1 <= self.sample_size <= 25:
            raise ValueError("sample_size must remain between 1 and 25")
        if not 0 <= self.max_new_downloads <= self.sample_size:
            raise ValueError("max_new_downloads must be between zero and sample_size")
        if self.max_products_per_star < 1 or self.max_products_per_star > 3:
            raise ValueError("max_products_per_star must remain between 1 and 3")
        if self.storage_limit_mb < 50:
            raise ValueError("storage_limit_mb must be at least 50 MB")
        if self.per_star_timeout_seconds < 15:
            raise ValueError("per_star_timeout_seconds must be at least 15 seconds")


@dataclass(frozen=True)
class DiscoveryConfig:
    """Complete serializable configuration for the Discovery Engine."""

    campaign: CampaignConfig = field(default_factory=CampaignConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DiscoveryConfig:
        """Build a validated configuration from a mapping."""
        return cls(
            campaign=CampaignConfig(**payload.get("campaign", {})),
            detection=DetectionConfig(
                **{
                    **payload.get("detection", {}),
                    "durations_hours": tuple(
                        payload.get("detection", {}).get(
                            "durations_hours", DetectionConfig().durations_hours
                        )
                    ),
                }
            ),
        )

    @classmethod
    def load(cls, path: Path) -> DiscoveryConfig:
        """Load a UTF-8 JSON configuration."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
