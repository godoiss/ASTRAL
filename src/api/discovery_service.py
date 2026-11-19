"""Public, bounded access to persisted Stage 10 discovery evidence."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.api.errors import ApiError
from src.discovery.campaign import CAMPAIGN_ROOT, PROJECT_ROOT, run_campaign
from src.discovery.catalog import protected_host_stars, select_exploration_targets
from src.discovery.config import CampaignConfig, DiscoveryConfig


DEFAULT_DISCOVERY_CONFIG = PROJECT_ROOT / "configs" / "discovery_engine.json"


def _sample_indices(records: int, limit: int) -> np.ndarray:
    if records <= limit:
        return np.arange(records, dtype=np.int64)
    return np.unique(np.linspace(0, records - 1, limit).astype(np.int64))


def _finite(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


class DiscoveryService:
    """Read safe public projections of discovery plans and campaigns."""

    def __init__(self, root: Path = CAMPAIGN_ROOT) -> None:
        self.root = root

    def plan(self, sample_size: int) -> dict[str, Any]:
        config = CampaignConfig(sample_size=sample_size, max_new_downloads=0, allow_network=False)
        targets = select_exploration_targets(config)
        protected = protected_host_stars()
        overlap = sorted({target.kic for target in targets} & protected)
        return {
            "sample_size": sample_size,
            "cached_targets": sum(target.cache_available for target in targets),
            "network_enabled": False,
            "protected_host_overlap": overlap,
            "targets": [
                {
                    "kic": target.kic,
                    "mission": "Kepler",
                    "cache_available": target.cache_available,
                    "selection_reference": target.selection_reference,
                    "label": -1,
                }
                for target in targets
            ],
        }

    def _campaign_path(self, campaign_id: str) -> Path:
        if not campaign_id.replace("-", "").replace("_", "").isalnum():
            raise ApiError(422, "invalid_campaign_id", "The campaign identifier is invalid.")
        path = self.root / campaign_id / "campaign.json"
        if not path.is_file():
            raise ApiError(404, "campaign_not_found", "No persisted discovery campaign matches the identifier.")
        return path

    def _load(self, campaign_id: str) -> dict[str, Any]:
        try:
            return json.loads(self._campaign_path(campaign_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiError(503, "campaign_unreadable", "The discovery campaign could not be read.") from exc

    @staticmethod
    def _summary(payload: dict[str, Any]) -> dict[str, Any]:
        summary = payload.get("summary", {})
        return {
            "campaign_id": str(payload["campaign_id"]),
            "status": str(payload["status"]),
            "started_at_utc": str(payload["started_at_utc"]),
            "completed_at_utc": payload.get("completed_at_utc"),
            "investigated_stars": int(summary.get("investigated_stars", 0)),
            "screening_detections": int(summary.get("screening_detections", 0)),
            "catalog_associated_detections": int(summary.get("catalog_associated_detections", 0)),
            "unassociated_inconclusive_detections": int(summary.get("unassociated_inconclusive_detections", 0)),
            "below_threshold": int(summary.get("below_threshold", 0)),
            "failed_or_excluded_stars": int(summary.get("failed_or_excluded_stars", 0)),
            "confirmed_discoveries": 0,
        }

    def campaigns(self, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        records: list[dict[str, Any]] = []
        for path in self.root.glob("*/campaign.json") if self.root.is_dir() else []:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records.append(self._summary(payload))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        records.sort(key=lambda item: item["started_at_utc"], reverse=True)
        return records[offset : offset + limit], len(records)

    def detail(self, campaign_id: str) -> dict[str, Any]:
        payload = self._load(campaign_id)
        results: list[dict[str, Any]] = []
        for item in payload.get("results", []):
            detection = item["detection"]
            relations = [
                str(association["relation"])
                for association in item.get("catalog_associations", [])
                if association.get("relation")
            ]
            results.append(
                {
                    "kic": int(item["target"]["kic"]),
                    "selection_reference": item["target"].get("selection_reference"),
                    "status": str(detection["status"]),
                    "period_days": float(detection["period_days"]),
                    "duration_hours": float(detection["duration_hours"]),
                    "depth_ppm": float(detection["depth_ppm"]),
                    "depth_snr": float(detection["depth_snr"]),
                    "ranking_score": float(detection["ranking_score"]),
                    "observed_events": int(detection["observed_events"]),
                    "event_coverage": float(detection["event_coverage"]),
                    "phase_coverage": float(detection["phase_coverage"]),
                    "possible_eclipsing_binary": bool(detection["possible_eclipsing_binary"]),
                    "catalog_period_association": bool(item["crossmatch"]["period_association_found"]),
                    "catalog_relations": relations,
                    "interpretation": str(item["interpretation"]),
                    "limitations": [str(value) for value in detection.get("limitations", [])],
                }
            )
        return {
            "campaign": self._summary(payload),
            "targets": results,
            "failures": [
                {"kic": failure.get("kic"), "reason": str(failure.get("reason", "unavailable"))[:240]}
                for failure in payload.get("failures", [])
            ],
            "scientific_boundaries": [str(value) for value in payload.get("scientific_boundaries", [])],
        }

    def series(self, campaign_id: str, kic: int, limit: int) -> dict[str, Any]:
        payload = self._load(campaign_id)
        item = next(
            (record for record in payload.get("results", []) if int(record["target"]["kic"]) == kic),
            None,
        )
        if item is None:
            raise ApiError(404, "target_not_found", "The target is not part of this discovery campaign.")
        processed = (PROJECT_ROOT / item["artifacts"]["processed_arrays"]).resolve()
        detection = (PROJECT_ROOT / item["artifacts"]["detection_arrays"]).resolve()
        root = PROJECT_ROOT.resolve()
        if root not in processed.parents or root not in detection.parents:
            raise ApiError(503, "artifact_invalid", "A campaign artifact is invalid.")
        try:
            with np.load(processed, allow_pickle=False) as arrays:
                time = np.asarray(arrays["time_bkjd"], dtype=float)
                flux = np.asarray(arrays["detrended_flux"], dtype=float)
            with np.load(detection, allow_pickle=False) as arrays:
                periods = np.asarray(arrays["periods"], dtype=float)
                powers = np.asarray(arrays["powers"], dtype=float)
                phase = np.asarray(arrays["phase"], dtype=float)
                phase_flux = np.asarray(arrays["phase_flux"], dtype=float)
                phase_centers = np.asarray(arrays["phase_bin_centers"], dtype=float)
                phase_bins = np.asarray(arrays["phase_bin_flux"], dtype=float)
        except (OSError, KeyError, ValueError) as exc:
            raise ApiError(503, "artifact_unreadable", "The discovery arrays could not be read.") from exc
        light_indices = _sample_indices(time.size, limit)
        period_indices = _sample_indices(periods.size, limit)
        phase_indices = _sample_indices(phase.size, limit)
        return {
            "campaign_id": campaign_id,
            "kic": kic,
            "records_total": int(time.size),
            "records_returned": int(light_indices.size),
            "downsampled": bool(time.size > light_indices.size),
            "time_bkjd": _finite(time[light_indices]),
            "detrended_flux": _finite(flux[light_indices]),
            "periods": _finite(periods[period_indices]),
            "powers": _finite(powers[period_indices]),
            "phase": _finite(phase[phase_indices]),
            "phase_flux": _finite(phase_flux[phase_indices]),
            "phase_bin_centers": _finite(phase_centers),
            "phase_bin_flux": _finite(phase_bins),
        }


class DiscoveryJobManager:
    """Run at most one cache-only campaign and expose real progress safely."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="astral-discovery")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(self, sample_size: int) -> dict[str, Any]:
        with self._lock:
            if any(job["status"] in {"queued", "running"} for job in self._jobs.values()):
                raise ApiError(409, "campaign_in_progress", "A discovery campaign is already running.")
            job_id = str(uuid.uuid4())
            campaign_id = f"web-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{job_id[:8]}"
            job = {
                "job_id": job_id,
                "status": "queued",
                "stage": "queued",
                "current": 0,
                "total": sample_size,
                "message": "Campaign accepted; network acquisition is disabled.",
                "campaign_id": campaign_id,
                "error_code": None,
            }
            self._jobs[job_id] = job
            self._executor.submit(self._execute, job_id, campaign_id, sample_size)
            return dict(job)

    def _execute(self, job_id: str, campaign_id: str, sample_size: int) -> None:
        def progress(stage: str, current: int, total: int, message: str) -> None:
            with self._lock:
                self._jobs[job_id].update(
                    status="running", stage=stage, current=current, total=total, message=message[:240]
                )

        try:
            base = DiscoveryConfig.load(DEFAULT_DISCOVERY_CONFIG)
            campaign = replace(
                base.campaign,
                sample_size=sample_size,
                allow_network=False,
                max_new_downloads=0,
            )
            run_campaign(replace(base, campaign=campaign), campaign_id=campaign_id, progress_callback=progress)
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].update(
                    status="failed",
                    stage="failed",
                    message="Campaign stopped safely.",
                    error_code=type(exc).__name__,
                )
        else:
            with self._lock:
                self._jobs[job_id].update(
                    status="completed",
                    stage="completed",
                    current=sample_size,
                    total=sample_size,
                    message="Campaign evidence persisted.",
                )

    def get(self, job_id: str) -> dict[str, Any]:
        try:
            uuid.UUID(job_id)
        except ValueError as exc:
            raise ApiError(422, "invalid_job_id", "The discovery job identifier is invalid.") from exc
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, "job_not_found", "No discovery job matches the identifier.")
            return dict(job)
