"""KIC-first access to real Kepler cadences with bounded optional MAST acquisition."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from src.discovery.models import RawLightCurve
from src.observatory.config import ObservatoryConfig
from src.observatory.mast import DEFAULT_FITS_CACHE, ProgressCallback, search_and_download
from src.observatory.processing import BKJD_OFFSET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_ROOT = PROJECT_ROOT / "data" / "discovery"
RAW_DIRECTORY = DISCOVERY_ROOT / "raw"
PROVENANCE_DIRECTORY = DISCOVERY_ROOT / "provenance"
OBSERVATORY_RAW = PROJECT_ROOT / "data" / "observatory" / "raw"
OBSERVATORY_PROVENANCE = PROJECT_ROOT / "data" / "observatory" / "provenance"
ANN02_MANIFEST = PROJECT_ROOT / "data" / "ann02" / "manifest.json"


class DiscoveryAcquisitionError(RuntimeError):
    """Raised when no authorized local or bounded remote observation is available."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".npz.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".json.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def discovery_raw_path(kic: int) -> Path:
    """Return the deterministic discovery-owned raw array path."""
    return RAW_DIRECTORY / f"kic_{int(kic):09d}.npz"


def discovery_provenance_path(kic: int) -> Path:
    """Return the deterministic discovery provenance path."""
    return PROVENANCE_DIRECTORY / f"kic_{int(kic):09d}.json"


def _observatory_source(kic: int) -> tuple[Path, Path] | None:
    if ANN02_MANIFEST.is_file():
        try:
            manifest = json.loads(ANN02_MANIFEST.read_text(encoding="utf-8"))
            for record in manifest.get("records", []):
                if int(record.get("kic", -1)) != int(kic):
                    continue
                object_id = str(record.get("object_id", ""))
                raw_path = OBSERVATORY_RAW / f"{object_id}.npz"
                provenance_path = OBSERVATORY_PROVENANCE / f"{object_id}.json"
                if raw_path.is_file() and provenance_path.is_file():
                    return raw_path, provenance_path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    for provenance_path in sorted(OBSERVATORY_PROVENANCE.glob("*.json")):
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            if int(payload.get("catalog", {}).get("kic", -1)) != int(kic):
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        raw_path = OBSERVATORY_RAW / f"{provenance_path.stem}.npz"
        if raw_path.is_file():
            return raw_path, provenance_path
    return None


def cached_kics() -> set[int]:
    """List stars with raw cadences already present in either local cache."""
    values: set[int] = set()
    for path in RAW_DIRECTORY.glob("kic_*.npz"):
        try:
            values.add(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    if ANN02_MANIFEST.is_file():
        try:
            manifest = json.loads(ANN02_MANIFEST.read_text(encoding="utf-8"))
            for record in manifest.get("records", []):
                object_id = str(record.get("object_id", ""))
                kic = int(record.get("kic", -1))
                if kic >= 0 and (OBSERVATORY_RAW / f"{object_id}.npz").is_file():
                    values.add(kic)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return values
    for provenance_path in OBSERVATORY_PROVENANCE.glob("*.json"):
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            kic = int(payload.get("catalog", {}).get("kic", -1))
            if kic >= 0 and (OBSERVATORY_RAW / f"{provenance_path.stem}.npz").is_file():
                values.add(kic)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return values


def _load_npz(kic: int, path: Path, provenance: dict[str, Any]) -> RawLightCurve:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    return RawLightCurve(
        kic=int(kic),
        time_bkjd=np.asarray(arrays["time_bkjd"], dtype=np.float64),
        flux=np.asarray(arrays["flux"], dtype=np.float64),
        flux_error=np.asarray(arrays["flux_error"], dtype=np.float64),
        quality=np.asarray(arrays["quality"], dtype=np.int64),
        segment=np.asarray(arrays["segment"], dtype=np.int64),
        provenance=provenance,
        source_path=path,
    )


def _time_bkjd(light_curve: Any) -> np.ndarray:
    time = light_curve.time
    if getattr(time, "format", None) == "bkjd":
        return np.asarray(time.value, dtype=np.float64)
    return np.asarray(time.tdb.jd - BKJD_OFFSET, dtype=np.float64)


def _values(values: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(getattr(values, "value", values), dtype=dtype)


def _collection_arrays(collection: Any) -> dict[str, np.ndarray]:
    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    segments: list[np.ndarray] = []
    for index, light_curve in enumerate(collection):
        time = _time_bkjd(light_curve)
        times.append(time)
        fluxes.append(_values(light_curve.flux))
        errors.append(_values(light_curve.flux_err))
        qualities.append(_values(light_curve.quality, np.int64))
        segments.append(np.full(time.size, index, dtype=np.int64))
    if not times:
        raise DiscoveryAcquisitionError("downloaded collection contains no light curves")
    return {
        "time_bkjd": np.concatenate(times),
        "flux": np.concatenate(fluxes),
        "flux_error": np.concatenate(errors),
        "quality": np.concatenate(qualities),
        "segment": np.concatenate(segments),
    }


def load_or_acquire_raw(
    kic: int,
    observatory_config: ObservatoryConfig,
    *,
    allow_network: bool,
    progress_callback: ProgressCallback | None = None,
) -> RawLightCurve:
    """Load raw cadences by KIC or perform one bounded MAST request.

    Existing Observatory arrays are safe inputs because only their original raw
    cadences are read; catalog-masked processed arrays are deliberately ignored.
    """
    own_raw = discovery_raw_path(kic)
    own_provenance = discovery_provenance_path(kic)
    if own_raw.is_file() and own_provenance.is_file():
        provenance = json.loads(own_provenance.read_text(encoding="utf-8"))
        return _load_npz(kic, own_raw, provenance)

    observatory = _observatory_source(kic)
    if observatory is not None:
        raw_path, provenance_path = observatory
        source = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance = {
            "schema_version": 1,
            "loaded_at_utc": datetime.now(UTC).isoformat(),
            "kic": int(kic),
            "source_kind": "observatory_raw_cache",
            "source_raw_sha256": _sha256(raw_path),
            "source_provenance_sha256": _sha256(provenance_path),
            "mission": source.get("observation", {}).get("mission", "Kepler"),
            "catalog_ephemeris_used_for_discovery_processing": False,
        }
        return _load_npz(kic, raw_path, provenance)

    if not allow_network:
        raise DiscoveryAcquisitionError(
            f"KIC {kic} has no local raw cache and network acquisition is disabled"
        )
    adapter = cast(Any, SimpleNamespace(kic=int(kic)))
    downloaded = search_and_download(
        adapter,
        observatory_config,
        cache_directory=DEFAULT_FITS_CACHE,
        progress_callback=progress_callback,
    )
    arrays = _collection_arrays(downloaded.collection)
    _atomic_npz(own_raw, arrays)
    provenance = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "kic": int(kic),
        "source_kind": "bounded_mast_download",
        "mission": observatory_config.mission,
        "query_identifier": f"KIC {int(kic)}",
        "products": downloaded.products,
        "selected_records": downloaded.selected_records,
        "search_records": downloaded.search_records,
        "download_elapsed_seconds": downloaded.elapsed_seconds,
        "raw_sha256": _sha256(own_raw),
        "catalog_ephemeris_used_for_discovery_processing": False,
        "software": {
            package: importlib.metadata.version(package)
            for package in ("lightkurve", "astropy", "astroquery", "numpy")
        },
    }
    _atomic_json(own_provenance, provenance)
    return _load_npz(kic, own_raw, provenance)
