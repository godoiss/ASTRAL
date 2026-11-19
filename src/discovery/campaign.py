"""Reproducible orchestration and persistence for bounded discovery campaigns."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.discovery.acquisition import DISCOVERY_ROOT, load_or_acquire_raw
from src.discovery.catalog import (
    KOI_PATH,
    SPLIT_MANIFEST,
    crossmatch_detection,
    crossmatch_summary,
    select_exploration_targets,
)
from src.discovery.config import DiscoveryConfig
from src.discovery.detection import detect_periodic_signal
from src.discovery.models import DetectionResult, DiscoveryTarget, ProcessedLightCurve
from src.discovery.processing import process_discovery_light_curve
from src.observatory.config import ObservatoryConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = PROJECT_ROOT / "experiments" / "stage10" / "campaigns"
LATEST_CAMPAIGN = PROJECT_ROOT / "experiments" / "stage10" / "latest_campaign.json"
ProgressCallback = Callable[[str, int, int, str], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _campaign_id() -> str:
    return datetime.now(UTC).strftime("kepler-bls-%Y%m%dT%H%M%SZ")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _persist_target_arrays(
    directory: Path,
    processed: ProcessedLightCurve,
    detection: DetectionResult,
) -> dict[str, str]:
    processed_path = _atomic_npz(
        directory / "processed.npz",
        {
            "time_bkjd": processed.time_bkjd,
            "normalized_flux": processed.normalized_flux,
            "detrended_flux": processed.detrended_flux,
            "normalized_error": processed.normalized_error,
            "segment": processed.segment,
        },
    )
    detection_path = _atomic_npz(
        directory / "detection.npz",
        {
            "periods": detection.periods,
            "powers": detection.powers,
            "phase": detection.phase,
            "phase_flux": detection.phase_flux,
            "phase_bin_centers": detection.phase_bin_centers,
            "phase_bin_flux": detection.phase_bin_flux,
        },
    )
    return {
        "processed_arrays": _relative(processed_path),
        "detection_arrays": _relative(detection_path),
        "processed_sha256": _sha256(processed_path),
        "detection_sha256": _sha256(detection_path),
    }


def _source_versions() -> dict[str, Any]:
    return {
        "koi_snapshot": {"path": _relative(KOI_PATH), "sha256": _sha256(KOI_PATH)},
        "stage04_split_manifest": {
            "path": _relative(SPLIT_MANIFEST),
            "sha256": _sha256(SPLIT_MANIFEST),
        },
        "software": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "astropy", "lightkurve")
        },
    }


def run_campaign(
    config: DiscoveryConfig,
    *,
    campaign_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run a bounded Kepler campaign and persist every result or failure."""
    identifier = campaign_id or _campaign_id()
    if not identifier.replace("-", "").replace("_", "").isalnum():
        raise ValueError("campaign_id may contain only letters, numbers, hyphens, and underscores")
    campaign_directory = CAMPAIGN_ROOT / identifier
    if campaign_directory.exists():
        raise FileExistsError(f"campaign already exists: {identifier}")
    campaign_directory.mkdir(parents=True)
    targets = select_exploration_targets(config.campaign)
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": identifier,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "mission": config.campaign.mission,
        "selection_strategy": config.campaign.strategy,
        "configuration": config.to_dict(),
        "source_versions": _source_versions(),
        "targets": [target.to_dict() for target in targets],
        "results": [],
        "failures": [],
        "summary": {},
        "scientific_boundaries": [
            "Targets are unlabeled and excluded from all labeled train/validation/test host stars.",
            "Frozen-test host identifiers are read only from the split manifest to enforce exclusion; no frozen-test feature, observation, label, or metric enters the campaign.",
            "No ANN-01 or ANN-02 inference, training, or confirmation is performed.",
            "A BLS screening detection is not validated statistical significance.",
            "No unmatched signal is presented as a new exoplanet discovery.",
        ],
    }
    campaign_path = campaign_directory / "campaign.json"
    _atomic_json(campaign_path, report)
    new_downloads = 0
    observatory_config = ObservatoryConfig(
        mission="Kepler",
        author="Kepler",
        cadence=config.campaign.cadence,
        quarter=config.campaign.quarter,
        max_products=config.campaign.max_products_per_star,
        timeout_seconds=config.campaign.per_star_timeout_seconds,
        retries=2,
    )

    for index, target in enumerate(targets, start=1):
        if progress_callback:
            progress_callback("target", index, len(targets), f"KIC {target.kic}")
        target_started = time.perf_counter()
        if not target.cache_available:
            if not config.campaign.allow_network:
                report["failures"].append(
                    {"kic": target.kic, "reason": "network_disabled_and_cache_unavailable"}
                )
                _atomic_json(campaign_path, report)
                continue
            if new_downloads >= config.campaign.max_new_downloads:
                report["failures"].append(
                    {"kic": target.kic, "reason": "new_download_limit_reached"}
                )
                _atomic_json(campaign_path, report)
                continue
            if _directory_size(DISCOVERY_ROOT) >= config.campaign.storage_limit_mb * 1024**2:
                report["failures"].append(
                    {"kic": target.kic, "reason": "discovery_storage_limit_reached"}
                )
                _atomic_json(campaign_path, report)
                continue
            new_downloads += 1
        try:
            def acquisition_progress(stage: str, _: float, message: str) -> None:
                if progress_callback:
                    progress_callback(stage, index, len(targets), message)

            raw = load_or_acquire_raw(
                target.kic,
                observatory_config,
                allow_network=config.campaign.allow_network,
                progress_callback=acquisition_progress,
            )
            if progress_callback:
                progress_callback("processing", index, len(targets), f"KIC {target.kic}")
            processed = process_discovery_light_curve(raw, config.detection)
            if progress_callback:
                progress_callback("bls", index, len(targets), f"KIC {target.kic}")
            detection = detect_periodic_signal(processed, config.detection)
            associations = crossmatch_detection(
                target.kic,
                detection.period_days,
                tolerance=config.detection.alias_ratio_tolerance,
            )
            target_directory = campaign_directory / "targets" / f"kic_{target.kic:09d}"
            artifacts = _persist_target_arrays(target_directory, processed, detection)
            crossmatch = crossmatch_summary(associations)
            interpretation = (
                "catalog_associated_signal"
                if detection.status == "screening_detection" and crossmatch["period_association_found"]
                else "unassociated_inconclusive_signal"
                if detection.status == "screening_detection"
                else "no_signal_above_screening_threshold"
            )
            target_report = {
                "schema_version": 1,
                "target": target.to_dict(),
                "elapsed_seconds": float(time.perf_counter() - target_started),
                "raw_provenance": raw.provenance,
                "processing": processed.diagnostics,
                "detection": detection.to_dict(),
                "catalog_associations": [asdict(item) for item in associations],
                "crossmatch": crossmatch,
                "interpretation": interpretation,
                "artifacts": artifacts,
            }
            target_path = _atomic_json(target_directory / "result.json", target_report)
            target_report["result_path"] = _relative(target_path)
            report["results"].append(target_report)
        except Exception as exc:
            report["failures"].append(
                {
                    "kic": target.kic,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": float(time.perf_counter() - target_started),
                }
            )
        _atomic_json(campaign_path, report)

    results = report["results"]
    detections = [item for item in results if item["detection"]["status"] == "screening_detection"]
    associated = [item for item in detections if item["crossmatch"]["period_association_found"]]
    inconclusive = [item for item in detections if not item["crossmatch"]["period_association_found"]]
    report["summary"] = {
        "requested_stars": len(targets),
        "investigated_stars": len(results),
        "failed_or_excluded_stars": len(report["failures"]),
        "screening_detections": len(detections),
        "catalog_associated_detections": len(associated),
        "unassociated_inconclusive_detections": len(inconclusive),
        "below_threshold": len(results) - len(detections),
        "new_downloads_started": new_downloads,
        "confirmed_discoveries": 0,
        "test_partition_accessed": False,
        "frozen_test_records_accessed": False,
        "split_manifest_identifiers_read_for_exclusion": True,
        "neural_models_invoked": False,
    }
    report["status"] = "completed_with_failures" if report["failures"] else "completed"
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = float(time.perf_counter() - started_clock)
    _atomic_json(campaign_path, report)
    _atomic_json(
        LATEST_CAMPAIGN,
        {
            "campaign_id": identifier,
            "campaign_path": _relative(campaign_path),
            "updated_at_utc": report["completed_at_utc"],
        },
    )
    return report


def load_campaign(path: Path | None = None) -> dict[str, Any] | None:
    """Load the latest or explicitly selected persisted campaign report."""
    if path is not None:
        candidate = Path(path)
    elif LATEST_CAMPAIGN.is_file():
        pointer = json.loads(LATEST_CAMPAIGN.read_text(encoding="utf-8"))
        candidate = PROJECT_ROOT / pointer["campaign_path"]
    else:
        return None
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))
