"""Atomic persistence for raw, processed, provenance, and ANN-02 artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.observatory.models import KoiMetadata, LightCurveArtifact, NeuralLightCurveExample


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBSERVATORY_DATA_ROOT = PROJECT_ROOT / "data" / "observatory"
RAW_DIRECTORY = OBSERVATORY_DATA_ROOT / "raw"
PROCESSED_DIRECTORY = OBSERVATORY_DATA_ROOT / "processed"
PROVENANCE_DIRECTORY = OBSERVATORY_DATA_ROOT / "provenance"
NEURAL_DIRECTORY = OBSERVATORY_DATA_ROOT / "neural"


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_", suffix=".json.tmp", dir=path.parent
    )
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


def _atomic_npz(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_", suffix=".npz.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def artifact_paths(object_id: str) -> tuple[Path, Path, Path]:
    """Return deterministic raw, processed, and provenance paths."""
    safe_name = object_id.replace("/", "_").replace("\\", "_")
    return (
        RAW_DIRECTORY / f"{safe_name}.npz",
        PROCESSED_DIRECTORY / f"{safe_name}.npz",
        PROVENANCE_DIRECTORY / f"{safe_name}.json",
    )


def artifact_is_cached(object_id: str) -> bool:
    """Return whether all required local derivatives exist."""
    return all(path.is_file() for path in artifact_paths(object_id))


def save_artifact(artifact: LightCurveArtifact) -> LightCurveArtifact:
    """Persist original cadences separately from all derived values."""
    raw_path, processed_path, provenance_path = artifact_paths(artifact.metadata.object_id)
    _atomic_npz(
        raw_path,
        {
            "time_bkjd": artifact.raw_time_bkjd,
            "flux": artifact.raw_flux,
            "flux_error": artifact.raw_flux_error,
            "quality": artifact.raw_quality,
            "segment": artifact.raw_segment,
        },
    )
    _atomic_npz(
        processed_path,
        {
            "clean_time_bkjd": artifact.clean_time_bkjd,
            "normalized_flux": artifact.normalized_flux,
            "detrended_flux": artifact.detrended_flux,
            "phase": artifact.phase,
            "local_phase_hours": artifact.local_phase_hours,
            "local_flux": artifact.local_flux,
        },
    )
    payload = {
        **artifact.provenance,
        "catalog": artifact.metadata.to_dict(),
        "observation": {
            "mission": artifact.mission,
            "author": artifact.author,
            "cadence_seconds": artifact.cadence_seconds,
            "start_bkjd": artifact.observation_start_bkjd,
            "end_bkjd": artifact.observation_end_bkjd,
            "start_bjd": artifact.observation_start_bkjd + 2454833.0,
            "end_bjd": artifact.observation_end_bkjd + 2454833.0,
            "segments": list(artifact.quarters),
        },
        "artifacts": {
            "raw_npz": str(raw_path.resolve()),
            "processed_npz": str(processed_path.resolve()),
        },
    }
    _atomic_json(provenance_path, payload)
    artifact.raw_path = raw_path
    artifact.processed_path = processed_path
    artifact.provenance_path = provenance_path
    return artifact


def load_artifact(metadata: KoiMetadata) -> LightCurveArtifact:
    """Restore a cached artifact without contacting MAST."""
    raw_path, processed_path, provenance_path = artifact_paths(metadata.object_id)
    with np.load(raw_path, allow_pickle=False) as raw:
        raw_values = {key: raw[key] for key in raw.files}
    with np.load(processed_path, allow_pickle=False) as processed:
        processed_values = {key: processed[key] for key in processed.files}
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    observation = provenance["observation"]
    return LightCurveArtifact(
        metadata=metadata,
        mission=str(observation["mission"]),
        author=str(observation["author"]),
        cadence_seconds=float(observation["cadence_seconds"]),
        observation_start_bkjd=float(observation["start_bkjd"]),
        observation_end_bkjd=float(observation["end_bkjd"]),
        quarters=tuple(observation["segments"]),
        raw_time_bkjd=np.asarray(raw_values["time_bkjd"], dtype=np.float64),
        raw_flux=np.asarray(raw_values["flux"], dtype=np.float64),
        raw_flux_error=np.asarray(raw_values["flux_error"], dtype=np.float64),
        raw_quality=np.asarray(raw_values["quality"], dtype=np.int64),
        raw_segment=np.asarray(raw_values["segment"], dtype=np.int64),
        clean_time_bkjd=np.asarray(processed_values["clean_time_bkjd"], dtype=np.float64),
        normalized_flux=np.asarray(processed_values["normalized_flux"], dtype=np.float64),
        detrended_flux=np.asarray(processed_values["detrended_flux"], dtype=np.float64),
        phase=np.asarray(processed_values["phase"], dtype=np.float64),
        local_phase_hours=np.asarray(processed_values["local_phase_hours"], dtype=np.float64),
        local_flux=np.asarray(processed_values["local_flux"], dtype=np.float64),
        provenance=provenance,
        raw_path=raw_path,
        processed_path=processed_path,
        provenance_path=provenance_path,
    )


def save_neural_example(example: NeuralLightCurveExample) -> Path:
    """Persist fixed views and masks for future ANN-02 consumers."""
    path = NEURAL_DIRECTORY / f"{example.object_id}.npz"
    return _atomic_npz(path, example.to_npz_payload())
