"""Read-only access to KOI metadata needed by the observatory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.observatory.models import KoiMetadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KOI_PATH = PROJECT_ROOT / "data" / "raw" / "kepler_koi.csv"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "experiments" / "stage04" / "split_manifest.csv"
DEFAULT_DERIVED_CATALOG = (
    PROJECT_ROOT / "data" / "observatory" / "catalog" / "koi_observatory_metadata.csv"
)

OBSERVATORY_COLUMNS = (
    "kepoi_name",
    "kepid",
    "kepler_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_teq",
    "koi_steff",
    "koi_srad",
    "koi_smass",
    "koi_slogg",
    "koi_sma",
    "koi_insol",
    "koi_kepmag",
)


class CatalogError(RuntimeError):
    """Raised when a KOI cannot safely supply an ephemeris."""


def display_koi_name(object_id: str) -> str:
    """Convert archive form ``K00753.01`` into human form ``KOI-753.01``."""
    value = str(object_id).strip().upper()
    if value.startswith("K") and "." in value:
        integer, suffix = value[1:].split(".", maxsplit=1)
        return f"KOI-{int(integer)}.{suffix}"
    return value


def load_observatory_catalog(
    koi_path: Path = DEFAULT_KOI_PATH,
    manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
) -> pd.DataFrame:
    """Load an observatory-specific view without changing the source CSV."""
    frame = pd.read_csv(koi_path, usecols=list(OBSERVATORY_COLUMNS))
    if Path(manifest_path).is_file():
        manifest = pd.read_csv(manifest_path, usecols=["kepoi_name", "split"])
        frame = frame.merge(manifest, on="kepoi_name", how="left", validate="one_to_one")
    else:
        frame["split"] = np.where(frame["koi_disposition"].eq("CANDIDATE"), "candidate", "unassigned")
    frame["split"] = frame["split"].fillna("unassigned")
    frame["display_name"] = frame["kepoi_name"].map(display_koi_name)
    return frame


def export_derived_catalog(
    output_path: Path = DEFAULT_DERIVED_CATALOG,
    koi_path: Path = DEFAULT_KOI_PATH,
    manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
) -> Path:
    """Persist a separate metadata view; never overwrite the validation input."""
    frame = load_observatory_catalog(koi_path, manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def _optional_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def metadata_from_row(row: pd.Series) -> KoiMetadata:
    """Validate and convert one catalog row to a typed metadata record."""
    required = ("kepid", "kepoi_name", "koi_period", "koi_time0bk", "koi_duration")
    if any(pd.isna(row[name]) for name in required):
        raise CatalogError("KIC, KOI, period, epoch, and duration are required")
    disposition = str(row["koi_disposition"])
    label = {"FALSE POSITIVE": 0, "CONFIRMED": 1}.get(disposition, -1)
    kepler_name = None if pd.isna(row["kepler_name"]) else str(row["kepler_name"])
    return KoiMetadata(
        object_id=str(row["kepoi_name"]),
        display_name=str(row.get("display_name", display_koi_name(str(row["kepoi_name"])))),
        kic=int(row["kepid"]),
        kepler_name=kepler_name,
        disposition=disposition,
        period_days=float(row["koi_period"]),
        epoch_bkjd=float(row["koi_time0bk"]),
        duration_hours=float(row["koi_duration"]),
        depth_ppm=_optional_float(row["koi_depth"]),
        planet_radius_earth=_optional_float(row["koi_prad"]),
        equilibrium_temperature_k=_optional_float(row["koi_teq"]),
        stellar_temperature_k=_optional_float(row["koi_steff"]),
        stellar_radius_solar=_optional_float(row["koi_srad"]),
        stellar_mass_solar=_optional_float(row["koi_smass"]),
        stellar_logg=_optional_float(row["koi_slogg"]),
        semi_major_axis_au=_optional_float(row["koi_sma"]),
        insolation_earth=_optional_float(row["koi_insol"]),
        kepler_magnitude=_optional_float(row["koi_kepmag"]),
        split=str(row.get("split", "unassigned")),
        class_label=label,
    )


def find_koi(
    identifier: str | int,
    catalog: pd.DataFrame | None = None,
) -> list[KoiMetadata]:
    """Find KOIs by archive name, display name, Kepler name, or KIC."""
    catalog = load_observatory_catalog() if catalog is None else catalog
    value = str(identifier).strip().upper()
    numeric = pd.to_numeric(value.replace("KIC", "").strip(), errors="coerce")
    mask = (
        catalog["kepoi_name"].astype(str).str.upper().eq(value)
        | catalog["display_name"].astype(str).str.upper().eq(value)
        | catalog["kepler_name"].fillna("").astype(str).str.upper().eq(value)
    )
    if not pd.isna(numeric):
        mask |= catalog["kepid"].eq(int(numeric))
    return [metadata_from_row(row) for _, row in catalog[mask].iterrows()]
