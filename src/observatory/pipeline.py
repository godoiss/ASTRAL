"""End-to-end orchestration for cached observational light-curve artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.observatory.config import ObservatoryConfig
from src.observatory.dataset import prepare_neural_example
from src.observatory.mast import DEFAULT_FITS_CACHE, ProgressCallback, search_and_download
from src.observatory.models import KoiMetadata, LightCurveArtifact, NeuralLightCurveExample
from src.observatory.persistence import artifact_is_cached, load_artifact, save_artifact
from src.observatory.processing import process_light_curves


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KOI_SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "kepler_koi.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(callback: ProgressCallback | None, stage: str, value: float, message: str) -> None:
    if callback is not None:
        callback(stage, value, message)


def _matching_fits(cache_directory: Path, kic: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in cache_directory.rglob("*.fits"):
        if str(kic) not in path.name and str(kic) not in str(path.parent):
            continue
        matches.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return matches


def acquire_observation(
    metadata: KoiMetadata,
    config: ObservatoryConfig,
    force_refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[LightCurveArtifact, NeuralLightCurveExample]:
    """Use cached derivatives or acquire, process, and persist a real observation."""
    if artifact_is_cached(metadata.object_id) and not force_refresh:
        artifact = load_artifact(metadata)
        if artifact.provenance.get("configuration") == config.to_dict():
            _emit(progress_callback, "cache", 1.0, "Loaded persisted observatory artifact")
            return artifact, prepare_neural_example(artifact, config, persist=True)
        _emit(
            progress_callback,
            "cache",
            0.03,
            "Cached processing configuration differs; refreshing bounded artifact",
        )

    downloaded = search_and_download(
        metadata,
        config,
        cache_directory=DEFAULT_FITS_CACHE,
        progress_callback=progress_callback,
    )
    _emit(progress_callback, "processing", 0.66, "Filtering quality flags and normalizing flux")
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "archive": "Mikulski Archive for Space Telescopes (MAST)",
            "service": "https://mast.stsci.edu",
            "mission": config.mission,
            "query_identifier": f"KIC {metadata.kic}",
            "products": downloaded.products,
            "search_records": downloaded.search_records,
            "selected_records": downloaded.selected_records,
            "fits_cache": str(downloaded.cache_directory),
            "fits_files": _matching_fits(downloaded.cache_directory, metadata.kic),
            "download_elapsed_seconds": downloaded.elapsed_seconds,
        },
        "catalog_source": {
            "path": str(KOI_SOURCE_PATH.resolve()),
            "sha256": _sha256(KOI_SOURCE_PATH),
            "table": "NASA Exoplanet Archive cumulative KOI",
        },
        "software": {
            package: importlib.metadata.version(package)
            for package in ("lightkurve", "astropy", "astroquery", "numpy")
        },
        "configuration": config.to_dict(),
        "scientific_notes": [
            "koi_time0bk is BKJD = BJD - 2454833.0.",
            "Empty neural bins remain NaN and are identified by false masks.",
            "Candidate scores and light-curve views do not establish confirmation.",
        ],
    }
    artifact = process_light_curves(
        downloaded.collection,
        metadata,
        config,
        provenance,
    )
    _emit(progress_callback, "persistence", 0.88, "Persisting raw and processed arrays separately")
    artifact = save_artifact(artifact)
    example = prepare_neural_example(artifact, config, persist=True)
    _emit(progress_callback, "complete", 1.0, "Deep Space Observatory artifact ready")
    return artifact, example
