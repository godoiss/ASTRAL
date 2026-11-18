"""Data contracts for catalog metadata and real light-curve artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class KoiMetadata:
    """Catalog identity, ephemeris, physical parameters, and split provenance."""

    object_id: str
    display_name: str
    kic: int
    kepler_name: str | None
    disposition: str
    period_days: float
    epoch_bkjd: float
    duration_hours: float
    depth_ppm: float | None
    planet_radius_earth: float | None
    equilibrium_temperature_k: float | None
    stellar_temperature_k: float | None
    stellar_radius_solar: float | None
    stellar_mass_solar: float | None
    stellar_logg: float | None
    semi_major_axis_au: float | None
    insolation_earth: float | None
    kepler_magnitude: float | None
    split: str
    class_label: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "object_id": self.object_id,
            "display_name": self.display_name,
            "kic": self.kic,
            "kepler_name": self.kepler_name,
            "disposition": self.disposition,
            "period_days": self.period_days,
            "epoch_bkjd": self.epoch_bkjd,
            "epoch_bjd": self.epoch_bkjd + 2454833.0,
            "duration_hours": self.duration_hours,
            "depth_ppm": self.depth_ppm,
            "planet_radius_earth": self.planet_radius_earth,
            "equilibrium_temperature_k": self.equilibrium_temperature_k,
            "stellar_temperature_k": self.stellar_temperature_k,
            "stellar_radius_solar": self.stellar_radius_solar,
            "stellar_mass_solar": self.stellar_mass_solar,
            "stellar_logg": self.stellar_logg,
            "semi_major_axis_au": self.semi_major_axis_au,
            "insolation_earth": self.insolation_earth,
            "kepler_magnitude": self.kepler_magnitude,
            "split": self.split,
            "class_label": self.class_label,
        }


@dataclass
class LightCurveArtifact:
    """Preserved raw cadences and derived signal views for one KOI."""

    metadata: KoiMetadata
    mission: str
    author: str
    cadence_seconds: float
    observation_start_bkjd: float
    observation_end_bkjd: float
    quarters: tuple[str, ...]
    raw_time_bkjd: FloatArray
    raw_flux: FloatArray
    raw_flux_error: FloatArray
    raw_quality: IntArray
    raw_segment: IntArray
    clean_time_bkjd: FloatArray
    normalized_flux: FloatArray
    detrended_flux: FloatArray
    phase: FloatArray
    local_phase_hours: FloatArray
    local_flux: FloatArray
    provenance: dict[str, Any]
    raw_path: Path | None = None
    processed_path: Path | None = None
    provenance_path: Path | None = None

    @property
    def raw_records(self) -> int:
        return int(self.raw_time_bkjd.size)

    @property
    def processed_records(self) -> int:
        return int(self.clean_time_bkjd.size)


@dataclass(frozen=True)
class NeuralLightCurveExample:
    """Fixed-size phase views with explicit observation masks for ANN-02."""

    object_id: str
    kic: int
    split: str
    class_label: int
    global_coordinates: FloatArray
    global_flux: FloatArray
    global_mask: BoolArray
    local_coordinates_hours: FloatArray
    local_flux: FloatArray
    local_mask: BoolArray

    def to_npz_payload(self) -> dict[str, Any]:
        """Return arrays and scalar identities without pickle-only objects."""
        return {
            "object_id": np.asarray(self.object_id),
            "kic": np.asarray(self.kic, dtype=np.int64),
            "split": np.asarray(self.split),
            "class_label": np.asarray(self.class_label, dtype=np.int64),
            "global_coordinates": self.global_coordinates,
            "global_flux": self.global_flux,
            "global_mask": self.global_mask,
            "local_coordinates_hours": self.local_coordinates_hours,
            "local_flux": self.local_flux,
            "local_mask": self.local_mask,
        }
