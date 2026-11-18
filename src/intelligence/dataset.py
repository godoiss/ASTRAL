"""Scientific summaries and bias diagnostics for the real ANN-02 signal cohort."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.deep_learning.dataset import DEFAULT_DATASET_PATH, DEFAULT_MANIFEST_PATH
from src.observatory.catalog import load_observatory_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "experiments" / "stage07" / "dataset_intelligence.json"
DEFAULT_RECORDS_PATH = PROJECT_ROOT / "experiments" / "stage07" / "dataset_records.csv"


def _atomic_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _summary(values: pd.Series | list[float]) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {"records": 0, "minimum": None, "q25": None, "median": None, "q75": None, "maximum": None, "mean": None}
    return {
        "records": int(numeric.size),
        "minimum": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q75": float(numeric.quantile(0.75)),
        "maximum": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_intelligence(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path | None = DEFAULT_REPORT_PATH,
    records_path: Path | None = DEFAULT_RECORDS_PATH,
) -> dict[str, Any]:
    """Build a real, serializable cohort report without touching the frozen test set."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = pd.json_normalize(manifest.get("records", []), sep=".")
    if records.empty:
        raise ValueError("ANN-02 manifest contains no processed records")
    catalog = load_observatory_catalog()
    metadata_columns = [
        "kepoi_name", "kepid", "koi_disposition", "koi_period", "koi_duration",
        "koi_depth", "koi_model_snr", "koi_prad", "koi_teq", "koi_kepmag", "split",
    ]
    records = records.merge(
        catalog.loc[:, metadata_columns],
        left_on="object_id",
        right_on="kepoi_name",
        how="left",
        validate="one_to_one",
        suffixes=("", "_catalog"),
    )
    accepted = records[records["status"].eq("accepted")].copy()
    eligible = catalog[
        catalog["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
        & catalog["split"].isin(["train", "validation"])
        & catalog["koi_period"].notna()
    ].copy()
    max_period = float(manifest["configuration"]["max_period_days"])
    eligible_short = eligible[eligible["koi_period"].le(max_period)]
    quality_fields = (
        "global_coverage", "local_coverage", "observed_transits", "raw_cadences",
        "retained_cadences", "retention_fraction", "segments", "cadence_seconds",
        "observation_span_days",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "sha256": _sha256(Path(dataset_path)),
            "requested": int(manifest["counts"]["requested"]),
            "accepted": int(len(accepted)),
            "excluded": int((records["status"] == "excluded").sum()),
            "failed": int(len(manifest.get("failures", {}))),
            "stars": int(accepted["kic"].nunique()),
        },
        "class_distribution": {
            split: {
                disposition: int(count)
                for disposition, count in group["disposition"].value_counts().items()
            }
            for split, group in accepted.groupby("split")
        },
        "quality": {
            field: _summary(accepted.get(f"quality.{field}", pd.Series(dtype=float)))
            for field in quality_fields
        },
        "astronomy": {
            "orbital_period_days": _summary(accepted["koi_period"]),
            "transit_duration_hours": _summary(accepted["koi_duration"]),
            "transit_depth_ppm": _summary(accepted["koi_depth"]),
            "catalog_model_snr": _summary(accepted["koi_model_snr"]),
            "kepler_magnitude": _summary(accepted["koi_kepmag"]),
        },
        "exclusions": dict(Counter(records.loc[records["status"].eq("excluded"), "exclusion_reason"].dropna().astype(str))),
        "failures": manifest.get("failures", {}),
        "bias_analysis": {
            "quarter": manifest["configuration"].get("quarter"),
            "period_limit_days": max_period,
            "eligible_labelled_records_before_period_limit": int(len(eligible)),
            "eligible_labelled_records_after_period_limit": int(len(eligible_short)),
            "fraction_eligible_removed_by_period_limit": float(1.0 - len(eligible_short) / max(len(eligible), 1)),
            "eligible_period_days": _summary(eligible["koi_period"]),
            "selected_period_days": _summary(accepted.loc[accepted["class_label"].ge(0), "koi_period"]),
            "eligible_model_snr": _summary(eligible_short["koi_model_snr"]),
            "selected_model_snr": _summary(accepted.loc[accepted["class_label"].ge(0), "koi_model_snr"]),
            "documented_effects": [
                "Quarter 3-only observations shorten the temporal baseline and do not measure quarter-to-quarter robustness.",
                "The <=30-day period rule favors systems with multiple transits in one quarter and excludes long-period planets.",
                "Cached and higher-SNR targets are preferred, so difficult low-SNR signals are under-represented.",
                "Class balancing is useful for optimization but does not represent KOI catalog prevalence.",
                "Quality thresholds can preferentially exclude sparse, faint, or incompletely covered signals.",
            ],
        },
        "guardrails": {
            "frozen_test_accessed": False,
            "candidates_used_as_labels": False,
            "cross_mission_training": False,
        },
    }
    if records_path is not None:
        Path(records_path).parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(records_path, index=False)
    if report_path is not None:
        _atomic_json(Path(report_path), report)
    return report
