"""Public, bounded access to persisted Stage 10 discovery evidence."""

from __future__ import annotations

import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.api.errors import ApiError
from src.discovery.acquisition import DISCOVERY_ROOT
from src.discovery.campaign import CAMPAIGN_ROOT, PROJECT_ROOT, run_campaign
from src.discovery.catalog import protected_host_stars, select_exploration_targets
from src.discovery.config import CampaignConfig, DiscoveryConfig
from src.research.campaign_store import ACTIVE_STATES, CampaignStore


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
    """Persist and execute one bounded local campaign at a time."""

    def __init__(
        self,
        *,
        store: CampaignStore | None = None,
        runner: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="astral-discovery")
        self._lock = threading.Lock()
        self.store = store or CampaignStore()
        self._runner = runner or run_campaign

    @staticmethod
    def _storage_bytes(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0

    @staticmethod
    def _job_error(exc: Exception) -> ApiError:
        if isinstance(exc, ApiError):
            return exc
        if isinstance(exc, ValueError):
            return ApiError(422, "invalid_campaign_state", str(exc))
        if isinstance(exc, KeyError):
            return ApiError(404, "job_not_found", "No discovery job matches the identifier.")
        return ApiError(500, "campaign_state_error", "Campaign state could not be updated.")

    def plan(
        self,
        sample_size: int,
        *,
        allow_network: bool = False,
        max_new_downloads: int = 0,
    ) -> dict[str, Any]:
        base = DiscoveryConfig.load(DEFAULT_DISCOVERY_CONFIG)
        campaign = replace(
            base.campaign,
            sample_size=sample_size,
            allow_network=allow_network,
            max_new_downloads=max_new_downloads if allow_network else 0,
        )
        config = replace(base, campaign=campaign)
        targets = select_exploration_targets(campaign)
        overlap = sorted({target.kic for target in targets} & protected_host_stars())
        if overlap:
            raise ApiError(409, "protected_host_overlap", "Campaign selection intersects protected host stars.")
        campaign_id = f"research-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        try:
            return self.store.create_plan(
                campaign_id=campaign_id,
                mission=campaign.mission,
                selection_strategy=campaign.strategy,
                total=sample_size,
                allow_network=allow_network,
                max_new_downloads=campaign.max_new_downloads,
                campaign_config=config.to_dict()["campaign"],
                detector_config=config.to_dict()["detection"],
                provenance={
                    "config_source": "configs/discovery_engine.json",
                    "protected_host_overlap": overlap,
                    "frozen_test_records_accessed": False,
                    "targets": [target.to_dict() for target in targets],
                },
                resources={
                    "cached_targets": sum(target.cache_available for target in targets),
                    "discovery_cache_bytes_at_plan": self._storage_bytes(DISCOVERY_ROOT),
                    "storage_limit_bytes": int(campaign.storage_limit_mb * 1024**2),
                    "worker_concurrency_limit": 1,
                    "per_star_timeout_seconds": campaign.per_star_timeout_seconds,
                },
            )
        except Exception as exc:
            raise self._job_error(exc) from exc

    def _ensure_no_active_job(self, excluding: str | None = None) -> None:
        active, _ = self.store.list(offset=0, limit=10, statuses=ACTIVE_STATES)
        if any(item["job_id"] != excluding for item in active):
            raise ApiError(409, "campaign_in_progress", "A discovery campaign is already queued or running.")

    def start_planned(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                job = self.store.get(job_id)
                if job["status"] != "planned":
                    raise ApiError(409, "invalid_campaign_state", "Only a planned campaign can be started.")
                self._ensure_no_active_job(excluding=job_id)
                queued = self.store.transition(
                    job_id,
                    "queued",
                    stage="queued",
                    message="Campaign queued for the single local research worker.",
                )
                self._executor.submit(self._execute, job_id, False)
                return queued
            except Exception as exc:
                raise self._job_error(exc) from exc

    def start(self, sample_size: int) -> dict[str, Any]:
        """Compatibility path: persist a cache-only plan and start it."""
        planned = self.plan(sample_size)
        return self.start_planned(planned["job_id"])

    def resume(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                job = self.store.get(job_id)
                if job["status"] not in {"interrupted", "failed", "cancelled"}:
                    raise ApiError(409, "campaign_not_resumable", "Campaign is not in a resumable state.")
                self._ensure_no_active_job(excluding=job_id)
                queued = self.store.transition(
                    job_id,
                    "queued",
                    stage="queued",
                    message="Persisted campaign queued for safe resume.",
                    event_type="campaign_resume_queued",
                )
                self._executor.submit(self._execute, job_id, True)
                return queued
            except Exception as exc:
                raise self._job_error(exc) from exc

    def cancel(self, job_id: str) -> dict[str, Any]:
        try:
            return self.store.request_cancel(job_id)
        except Exception as exc:
            raise self._job_error(exc) from exc

    def _execute(self, job_id: str, resume: bool) -> None:
        started = perf_counter()

        def progress(stage: str, current: int, total: int, message: str) -> None:
            match = re.search(r"KIC\s+(\d+)", message)
            self.store.progress(
                job_id,
                stage=stage,
                current=current,
                total=total,
                message=message,
                kic=int(match.group(1)) if match else None,
                elapsed_seconds=float(perf_counter() - started),
            )

        try:
            job = self.store.transition(
                job_id,
                "running",
                stage="initializing",
                message="Scientific configuration and persisted state are being validated.",
            )
            config = DiscoveryConfig.from_dict(
                {"campaign": job["campaign_config"], "detection": job["detector_config"]}
            )
            report = self._runner(
                config,
                campaign_id=job["campaign_id"],
                progress_callback=progress,
                cancellation_callback=lambda: self.store.is_cancel_requested(job_id),
                resume=resume,
            )
        except Exception as exc:
            try:
                if self.store.is_cancel_requested(job_id):
                    self.store.transition(
                        job_id,
                        "cancelled",
                        stage="cancelled",
                        message="Campaign cancelled at a safe scientific boundary.",
                    )
                else:
                    self.store.transition(
                        job_id,
                        "failed",
                        stage="failed",
                        message="Campaign stopped safely; persisted evidence was retained.",
                        error_code=type(exc).__name__,
                        level="error",
                    )
            except Exception:
                pass
        else:
            operational_status = "cancelled" if report["status"] == "cancelled" else "completed"
            self.store.transition(
                job_id,
                operational_status,
                stage=str(report["status"]),
                message=(
                    "Campaign cancelled at a safe boundary; partial evidence remains resumable."
                    if operational_status == "cancelled"
                    else "Campaign evidence and provenance persisted successfully."
                ),
                result_summary=report.get("summary", {}),
                artifact_path=f"experiments/stage10/campaigns/{report['campaign_id']}/campaign.json",
                event_type=f"campaign_{operational_status}",
                level="warning" if operational_status == "cancelled" else "info",
            )

    def get(self, job_id: str) -> dict[str, Any]:
        try:
            return self.store.get(job_id)
        except Exception as exc:
            raise self._job_error(exc) from exc

    def list(self, offset: int, limit: int, statuses: tuple[str, ...] = ()) -> tuple[list[dict[str, Any]], int]:
        try:
            return self.store.list(offset=offset, limit=limit, statuses=statuses)
        except Exception as exc:
            raise self._job_error(exc) from exc

    def events(self, job_id: str, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        try:
            return self.store.events(job_id, offset=offset, limit=limit)
        except Exception as exc:
            raise self._job_error(exc) from exc

    def provenance(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        return {
            "job_id": job["job_id"],
            "campaign_id": job["campaign_id"],
            "mission": job["mission"],
            "selection_strategy": job["selection_strategy"],
            "campaign_config": job["campaign_config"],
            "detector_config": job["detector_config"],
            "provenance": job["provenance"],
            "resources": {
                **job["resource"],
                "discovery_cache_bytes_current": self._storage_bytes(DISCOVERY_ROOT),
                "campaign_artifact_bytes": self._storage_bytes(CAMPAIGN_ROOT / job["campaign_id"]),
            },
            "artifact_path": job["artifact_path"],
            "scientific_guardrails": {
                "frozen_test_records_accessed": False,
                "neural_models_invoked": False,
                "mission_data_mixed": False,
            },
        }
