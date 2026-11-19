"""Typed scientific records used by the Discovery Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DiscoveryTarget:
    """A Kepler host star selected without supplying a transit ephemeris."""

    kic: int
    mission: str
    selection_strategy: str
    cache_available: bool
    selection_reference: str | None = None
    label: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible target metadata."""
        return asdict(self)


@dataclass
class RawLightCurve:
    """Original observed cadences for one host star."""

    kic: int
    time_bkjd: FloatArray
    flux: FloatArray
    flux_error: FloatArray
    quality: IntArray
    segment: IntArray
    provenance: dict[str, Any]
    source_path: Path | None = None


@dataclass
class ProcessedLightCurve:
    """Quality-filtered and independently detrended discovery series."""

    kic: int
    time_bkjd: FloatArray
    normalized_flux: FloatArray
    detrended_flux: FloatArray
    normalized_error: FloatArray
    segment: IntArray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AliasDiagnostic:
    """A competing BLS peak and its simple harmonic relation."""

    period_days: float
    depth_snr: float
    ratio_to_primary: float
    relation: str | None


@dataclass(frozen=True)
class CatalogAssociation:
    """Post-detection KOI/TCE association from the local catalog snapshot."""

    object_id: str
    disposition: str
    catalog_period_days: float | None
    period_ratio: float | None
    relation: str | None
    tce_planet_number: int | None
    tce_delivery: str | None


@dataclass
class DetectionResult:
    """Best periodic signal, diagnostics, and arrays required for inspection."""

    status: str
    period_days: float
    period_uncertainty_days: float
    transit_time_bkjd: float
    transit_time_uncertainty_days: float
    duration_hours: float
    duration_uncertainty_hours: float
    depth_fraction: float
    depth_ppm: float
    depth_error_fraction: float
    depth_snr: float
    ranking_score: float
    expected_events: int
    observed_events: int
    event_coverage: float
    phase_coverage: float
    event_depths: list[float]
    event_depth_mad: float | None
    odd_depth: float | None
    even_depth: float | None
    odd_even_sigma: float | None
    secondary_depth: float | None
    secondary_sigma: float | None
    possible_eclipsing_binary: bool
    aliases: list[AliasDiagnostic]
    limitations: list[str]
    periods: FloatArray = field(repr=False)
    powers: FloatArray = field(repr=False)
    phase: FloatArray = field(repr=False)
    phase_flux: FloatArray = field(repr=False)
    phase_bin_centers: FloatArray = field(repr=False)
    phase_bin_flux: FloatArray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return scalar diagnostics without large numerical arrays."""
        return {
            "status": self.status,
            "period_days": self.period_days,
            "period_uncertainty_days": self.period_uncertainty_days,
            "transit_time_bkjd": self.transit_time_bkjd,
            "transit_time_uncertainty_days": self.transit_time_uncertainty_days,
            "duration_hours": self.duration_hours,
            "duration_uncertainty_hours": self.duration_uncertainty_hours,
            "depth_fraction": self.depth_fraction,
            "depth_ppm": self.depth_ppm,
            "depth_error_fraction": self.depth_error_fraction,
            "depth_snr": self.depth_snr,
            "ranking_score": self.ranking_score,
            "expected_events": self.expected_events,
            "observed_events": self.observed_events,
            "event_coverage": self.event_coverage,
            "phase_coverage": self.phase_coverage,
            "event_depths": self.event_depths,
            "event_depth_mad": self.event_depth_mad,
            "odd_depth": self.odd_depth,
            "even_depth": self.even_depth,
            "odd_even_sigma": self.odd_even_sigma,
            "secondary_depth": self.secondary_depth,
            "secondary_sigma": self.secondary_sigma,
            "possible_eclipsing_binary": self.possible_eclipsing_binary,
            "aliases": [asdict(item) for item in self.aliases],
            "limitations": self.limitations,
        }
