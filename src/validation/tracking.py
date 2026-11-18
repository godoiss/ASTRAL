"""Local, JSON-based experiment tracking for Stage 04."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(path: Path, payload: Any) -> Path:
    """Write strict JSON atomically, rejecting NaN and infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".json.tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def dataset_version(path: Path, records: int, columns: int) -> dict[str, Any]:
    """Return a content-addressed identity for the exact source dataset."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    file_hash = digest.hexdigest()
    file_stat = Path(path).stat()
    return {
        "version_id": f"sha256:{file_hash[:16]}",
        "sha256": file_hash,
        "path": str(Path(path).resolve()),
        "bytes": file_stat.st_size,
        "records": int(records),
        "columns": int(columns),
    }


class ExperimentTracker:
    """Persist immutable run files and an incrementally updated registry."""

    def __init__(
        self,
        root: Path,
        campaign_id: str,
        dataset: dict[str, Any],
    ) -> None:
        self.root = Path(root)
        self.runs_directory = self.root / "runs"
        self.runs_directory.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "experiment_registry.json"
        self.registry: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "runs": [],
        }
        write_json_atomic(self.registry_path, self.registry)

    def record(self, run: dict[str, Any]) -> Path:
        """Persist a complete run, including parameters and real history."""
        experiment_id = run["experiment_id"]
        run_path = self.runs_directory / f"{experiment_id}.json"
        if run_path.exists():
            raise FileExistsError(f"Experiment already exists: {experiment_id}")
        write_json_atomic(run_path, run)
        self.registry["runs"].append(
            {
                "experiment_id": experiment_id,
                "path": str(run_path.relative_to(self.root)),
                "category": run["category"],
                "validation_metrics": run["metrics"]["validation"],
                "training_seconds": run["training_seconds"],
            }
        )
        write_json_atomic(self.registry_path, self.registry)
        return run_path

    def attach_campaign_artifacts(self, artifacts: list[str]) -> None:
        """Attach generated scientific graphics and outputs to the registry."""
        self.registry["artifacts"] = artifacts
        write_json_atomic(self.registry_path, self.registry)

