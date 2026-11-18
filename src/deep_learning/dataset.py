"""Incremental real-data preparation and PyTorch datasets for ANN-02."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.deep_learning.config import DatasetBuildConfig
from src.observatory.catalog import load_observatory_catalog, metadata_from_row
from src.observatory.config import ObservatoryConfig
from src.observatory.dataset import validate_group_separation
from src.observatory.models import LightCurveArtifact, NeuralLightCurveExample
from src.observatory.persistence import artifact_is_cached
from src.observatory.persistence import load_artifact
from src.observatory.dataset import prepare_neural_example
from src.observatory.pipeline import acquire_observation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "ann02"
DEFAULT_DATASET_PATH = DATA_ROOT / "ann02_dataset.npz"
DEFAULT_MANIFEST_PATH = DATA_ROOT / "manifest.json"
DEFAULT_CHECKPOINT_PATH = DATA_ROOT / "acquisition_checkpoint.json"
OBSERVATORY_ROOT = PROJECT_ROOT / "data" / "observatory"
ProgressCallback = Callable[[str, int, int, str], None]


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acquire_with_hard_timeout(
    metadata: Any,
    observatory_config: ObservatoryConfig,
    timeout_seconds: int,
) -> tuple[LightCurveArtifact, NeuralLightCurveExample]:
    """Acquire in an isolated process so stalled archive I/O can be terminated."""
    command = [
        sys.executable,
        "-m",
        "src.deep_learning.acquire_worker",
        metadata.object_id,
        "--max-products",
        str(observatory_config.max_products),
    ]
    if observatory_config.quarter is not None:
        command.extend(("--quarter", str(observatory_config.quarter)))
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"MAST acquisition exceeded the {timeout_seconds}s per-object limit"
        ) from exc
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip().splitlines()
        message = details[-1] if details else f"worker exit code {result.returncode}"
        raise RuntimeError(f"isolated MAST worker failed: {message}")
    if not artifact_is_cached(metadata.object_id):
        raise RuntimeError("isolated MAST worker completed without persisted artifacts")
    artifact = load_artifact(metadata)
    return artifact, prepare_neural_example(artifact, observatory_config, persist=True)


def select_balanced_cohort(
    catalog: pd.DataFrame,
    config: DatasetBuildConfig,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced pilot while honoring frozen Stage 04 splits."""
    eligible = catalog[
        catalog["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
        & catalog["split"].isin(["train", "validation"])
        & catalog["koi_period"].notna()
        & catalog["koi_time0bk"].notna()
        & catalog["koi_duration"].notna()
        & catalog["koi_period"].le(config.max_period_days)
    ].copy()
    eligible["class_label"] = eligible["koi_disposition"].map(
        {"FALSE POSITIVE": 0, "CONFIRMED": 1}
    )
    eligible["cached"] = eligible["kepoi_name"].map(artifact_is_cached)

    labelled_per_class = config.pilot_size // 2
    train_per_class = int(round(labelled_per_class * 0.75))
    validation_per_class = labelled_per_class - train_per_class
    rng = random.Random(config.seed)
    selected: list[dict[str, Any]] = []
    used_kic: set[int] = set()
    for split, count in (("train", train_per_class), ("validation", validation_per_class)):
        for label in (0, 1):
            subset = eligible[(eligible["split"] == split) & (eligible["class_label"] == label)]
            rows = [row for _, row in subset.iterrows()]
            rng.shuffle(rows)
            rows.sort(
                key=lambda row: (
                    not bool(row["cached"]),
                    -float(row["koi_model_snr"])
                    if pd.notna(row["koi_model_snr"])
                    else float("inf"),
                )
            )
            taken = 0
            for row in rows:
                kic = int(row["kepid"])
                if kic in used_kic:
                    continue
                used_kic.add(kic)
                selected.append(
                    {
                        "object_id": str(row["kepoi_name"]),
                        "kic": kic,
                        "split": split,
                        "disposition": str(row["koi_disposition"]),
                        "class_label": label,
                        "cached_at_selection": bool(row["cached"]),
                    }
                )
                taken += 1
                if taken == count:
                    break
            if taken != count:
                raise RuntimeError(f"Insufficient eligible {split} records for class {label}")

    if config.candidate_size:
        candidates = catalog[
            catalog["koi_disposition"].eq("CANDIDATE")
            & catalog["koi_period"].notna()
            & catalog["koi_time0bk"].notna()
            & catalog["koi_duration"].notna()
            & catalog["koi_period"].le(config.max_period_days)
        ].copy()
        candidates["preferred"] = candidates["kepoi_name"].eq("K00753.01")
        candidates["cached"] = candidates["kepoi_name"].map(artifact_is_cached)
        rows = [row for _, row in candidates.iterrows()]
        rng.shuffle(rows)
        rows.sort(key=lambda row: (not bool(row["preferred"]), not bool(row["cached"])))
        for row in rows:
            kic = int(row["kepid"])
            if kic in used_kic:
                continue
            used_kic.add(kic)
            selected.append(
                {
                    "object_id": str(row["kepoi_name"]),
                    "kic": kic,
                    "split": "candidate",
                    "disposition": "CANDIDATE",
                    "class_label": -1,
                    "cached_at_selection": bool(row["cached"]),
                }
            )
            if sum(item["split"] == "candidate" for item in selected) == config.candidate_size:
                break
    return selected


def signal_quality(
    artifact: LightCurveArtifact,
    example: NeuralLightCurveExample,
) -> dict[str, Any]:
    """Compute transparent coverage and transit-observation diagnostics."""
    metadata = artifact.metadata
    offsets = artifact.clean_time_bkjd - metadata.epoch_bkjd
    cycles = np.rint(offsets / metadata.period_days).astype(np.int64)
    distance_hours = np.abs(offsets - cycles * metadata.period_days) * 24.0
    tolerance = max(metadata.duration_hours / 2.0, artifact.cadence_seconds / 3600.0)
    observed_cycles = np.unique(cycles[distance_hours <= tolerance])
    return {
        "global_coverage": float(np.mean(example.global_mask)),
        "local_coverage": float(np.mean(example.local_mask)),
        "observed_transits": int(observed_cycles.size),
        "raw_cadences": artifact.raw_records,
        "retained_cadences": artifact.processed_records,
        "retention_fraction": float(artifact.processed_records / max(artifact.raw_records, 1)),
        "segments": len(artifact.quarters),
        "cadence_seconds": float(artifact.cadence_seconds),
        "observation_span_days": float(
            artifact.observation_end_bkjd - artifact.observation_start_bkjd
        ),
    }


def quality_exclusion(quality: dict[str, Any], config: DatasetBuildConfig) -> str | None:
    """Return the first pre-registered quality exclusion, if any."""
    if quality["global_coverage"] < config.min_global_coverage:
        return "global_coverage_below_threshold"
    if quality["local_coverage"] < config.min_local_coverage:
        return "local_coverage_below_threshold"
    if quality["observed_transits"] < config.min_observed_transits:
        return "insufficient_observed_transits"
    return None


def validate_stage05_pipeline(catalog: pd.DataFrame | None = None) -> dict[str, Any]:
    """Validate cached real examples across the three original dispositions."""
    catalog = load_observatory_catalog() if catalog is None else catalog
    results: list[dict[str, Any]] = []
    for disposition in ("CONFIRMED", "FALSE POSITIVE", "CANDIDATE"):
        subset = catalog[catalog["koi_disposition"].eq(disposition)]
        cached = subset[subset["kepoi_name"].map(artifact_is_cached)]
        if cached.empty:
            results.append({"disposition": disposition, "status": "unavailable"})
            continue
        row = cached.iloc[0]
        metadata = metadata_from_row(row)
        artifact, example = acquire_observation(metadata, ObservatoryConfig())
        valid = (
            example.global_flux.shape == (2001,)
            and example.local_flux.shape == (201,)
            and np.array_equal(np.isnan(example.global_flux), ~example.global_mask)
            and np.array_equal(np.isnan(example.local_flux), ~example.local_mask)
            and artifact.provenance_path is not None
        )
        results.append(
            {
                "object_id": metadata.object_id,
                "kic": metadata.kic,
                "disposition": disposition,
                "status": "passed" if valid else "failed",
                "quality": signal_quality(artifact, example),
            }
        )
    return {
        "passed": all(item["status"] == "passed" for item in results),
        "checks": results,
    }


def _save_collection(examples: list[NeuralLightCurveExample], path: Path) -> Path:
    validate_group_separation(examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".npz.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                object_ids=np.asarray([item.object_id for item in examples]),
                kic=np.asarray([item.kic for item in examples], dtype=np.int64),
                splits=np.asarray([item.split for item in examples]),
                class_labels=np.asarray([item.class_label for item in examples], dtype=np.int64),
                global_flux=np.stack([item.global_flux for item in examples]),
                global_mask=np.stack([item.global_mask for item in examples]),
                local_flux=np.stack([item.local_flux for item in examples]),
                local_mask=np.stack([item.local_mask for item in examples]),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_incremental_dataset(
    config: DatasetBuildConfig,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Acquire a bounded cohort incrementally and persist progress after every object."""
    catalog = load_observatory_catalog()
    cohort = select_balanced_cohort(catalog, config)
    cohort_ids = [item["object_id"] for item in cohort]
    checkpoint_path = Path(checkpoint_path)
    checkpoint: dict[str, Any] = {
        "schema_version": 1,
        "configuration": config.to_dict(),
        "cohort": cohort,
        "completed": {},
        "failed": {},
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    if checkpoint_path.is_file():
        prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if prior.get("configuration") == config.to_dict() and [
            item["object_id"] for item in prior.get("cohort", [])
        ] == cohort_ids:
            checkpoint = prior

    stage05_validation = validate_stage05_pipeline(catalog)
    observatory_config = ObservatoryConfig(
        quarter=config.quarter,
        max_products=config.max_products_per_object,
    )
    examples: list[NeuralLightCurveExample] = []
    records: list[dict[str, Any]] = []
    new_downloads = 0
    for index, selected in enumerate(cohort, start=1):
        object_id = selected["object_id"]
        if progress_callback:
            progress_callback("acquisition", index, len(cohort), object_id)
        row = catalog[catalog["kepoi_name"].eq(object_id)].iloc[0]
        metadata = metadata_from_row(row)
        was_cached = artifact_is_cached(object_id)
        if not was_cached:
            if new_downloads >= config.max_new_downloads:
                checkpoint["failed"][object_id] = "new_download_limit_reached"
                _atomic_json(checkpoint_path, checkpoint)
                continue
            if _directory_size(OBSERVATORY_ROOT) >= config.storage_limit_mb * 1024**2:
                checkpoint["failed"][object_id] = "storage_limit_reached"
                _atomic_json(checkpoint_path, checkpoint)
                continue
            new_downloads += 1
        try:
            if was_cached:
                artifact, example = acquire_observation(metadata, observatory_config)
            else:
                artifact, example = _acquire_with_hard_timeout(
                    metadata,
                    observatory_config,
                    config.per_object_timeout_seconds,
                )
            quality = signal_quality(artifact, example)
            exclusion = quality_exclusion(quality, config)
            record = {
                **selected,
                "status": "excluded" if exclusion else "accepted",
                "exclusion_reason": exclusion,
                "quality": quality,
                "provenance_path": str(artifact.provenance_path.resolve()),
                "raw_path": str(artifact.raw_path.resolve()),
                "processed_path": str(artifact.processed_path.resolve()),
            }
            checkpoint["completed"][object_id] = record
            checkpoint["failed"].pop(object_id, None)
            records.append(record)
            if exclusion is None:
                examples.append(example)
        except Exception as exc:  # archive/network/scientific errors are recorded per object
            checkpoint["failed"][object_id] = f"{type(exc).__name__}: {exc}"
        checkpoint["updated_at_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(checkpoint_path, checkpoint)

    if not examples:
        raise RuntimeError("No observation passed the ANN-02 quality criteria")
    _save_collection(examples, Path(dataset_path))
    counts: dict[str, Any] = {
        "requested": len(cohort),
        "accepted": len(examples),
        "excluded": sum(record["status"] == "excluded" for record in records),
        "failed": len(checkpoint["failed"]),
        "new_downloads_this_run": new_downloads,
        "by_split": {},
        "by_disposition": {},
    }
    for value in ("train", "validation", "candidate"):
        counts["by_split"][value] = sum(item.split == value for item in examples)
    for value in ("CONFIRMED", "FALSE POSITIVE", "CANDIDATE"):
        counts["by_disposition"][value] = sum(
            record["disposition"] == value and record["status"] == "accepted"
            for record in records
        )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": _file_sha256(Path(dataset_path)),
        "configuration": config.to_dict(),
        "stage05_pipeline_validation": stage05_validation,
        "counts": counts,
        "records": records,
        "failures": checkpoint["failed"],
        "selection_biases": [
            "The pilot is class-balanced and therefore does not represent catalog prevalence.",
            f"Periods above {config.max_period_days:g} days are excluded to improve transit coverage in bounded observations.",
            "Cached targets are preferred to reduce bandwidth; this can introduce availability bias.",
            "Within uncached strata, higher catalog model SNR is preferred for pilot viability; metrics will overstate performance on difficult low-SNR signals.",
            "Quality thresholds can preferentially remove faint, sparse, or long-period signals.",
        ],
        "split_policy": {
            "source": "Stage 04 frozen KIC-grouped split manifest",
            "test_records_acquired": 0,
            "test_used_for_selection": False,
            "candidates_used_for_training": False,
        },
    }
    _atomic_json(Path(manifest_path), manifest)
    return manifest


def _tensor_view(flux: np.ndarray, mask: np.ndarray) -> Tensor:
    """Center observed flux, scale robustly, and keep missing values neutral at zero."""
    observed = np.asarray(flux, dtype=np.float32)[mask]
    transformed = np.zeros_like(flux, dtype=np.float32)
    if observed.size:
        center = float(np.median(observed))
        residuals = observed - center
        scale = max(float(np.percentile(np.abs(residuals), 95)), 1e-6)
        transformed[mask] = np.clip((np.asarray(flux, dtype=np.float32)[mask] - center) / scale, -8.0, 8.0)
    return torch.from_numpy(np.stack((transformed, mask.astype(np.float32))))


class KeplerSignalDataset(Dataset[dict[str, Any]]):
    """Map the immutable ANN-02 NPZ collection into mask-aware tensors."""

    def __init__(self, path: Path = DEFAULT_DATASET_PATH, split: str | None = None) -> None:
        with np.load(Path(path), allow_pickle=False) as payload:
            self.values = {key: payload[key] for key in payload.files}
        all_indices = np.arange(len(self.values["object_ids"]))
        self.indices = (
            all_indices
            if split is None
            else all_indices[self.values["splits"].astype(str) == split]
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        return {
            "global_view": _tensor_view(
                self.values["global_flux"][index], self.values["global_mask"][index]
            ),
            "local_view": _tensor_view(
                self.values["local_flux"][index], self.values["local_mask"][index]
            ),
            "label": torch.tensor(float(self.values["class_labels"][index]), dtype=torch.float32),
            "object_id": str(self.values["object_ids"][index]),
            "kic": torch.tensor(int(self.values["kic"][index]), dtype=torch.int64),
            "split": str(self.values["splits"][index]),
            "source_index": torch.tensor(index, dtype=torch.int64),
        }
