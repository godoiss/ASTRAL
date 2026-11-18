"""Like-for-like ANN-01 and ANN-02 comparison on an identical validation cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing import OBJECT_ID_COLUMN
from src.deep_learning.dataset import DEFAULT_DATASET_PATH, KeplerSignalDataset
from src.deep_learning.training import ANN02Bundle, infer_ann02
from src.lab.services import LabModelState, infer_record
from src.validation.metrics import compute_classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "experiments" / "stage07" / "model_comparison.json"
DEFAULT_ROWS_PATH = PROJECT_ROOT / "experiments" / "stage07" / "model_comparison.csv"


def compare_ann01_ann02(
    ann01_state: LabModelState,
    ann02_bundle: ANN02Bundle,
    ann_frame: pd.DataFrame,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path | None = DEFAULT_REPORT_PATH,
    rows_path: Path | None = DEFAULT_ROWS_PATH,
) -> dict[str, Any]:
    """Evaluate both models on the exact same ANN-02 validation object identities."""
    dataset = KeplerSignalDataset(dataset_path, split="validation")
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        matches = ann_frame[ann_frame[OBJECT_ID_COLUMN].eq(sample["object_id"])]
        if len(matches) != 1:
            raise ValueError(f"ANN-01 metadata unavailable or ambiguous for {sample['object_id']}")
        ann01 = infer_record(ann01_state, matches.iloc[0])
        ann02 = infer_ann02(ann02_bundle, sample)
        target = int(sample["label"].item())
        ann01_class = int(ann01.score >= ann01.threshold)
        ann02_class = int(ann02["score"] >= ann02["threshold"])
        rows.append(
            {
                "object_id": sample["object_id"],
                "kic": int(sample["kic"]),
                "target": target,
                "catalog_disposition": "CONFIRMED" if target == 1 else "FALSE POSITIVE",
                "ann01_score": ann01.score,
                "ann01_class": ann01_class,
                "ann02_score": ann02["score"],
                "ann02_class": ann02_class,
                "models_agree": ann01_class == ann02_class,
                "ann01_correct": ann01_class == target,
                "ann02_correct": ann02_class == target,
            }
        )
    frame = pd.DataFrame(rows)
    targets = frame["target"].to_numpy(dtype=np.int64)
    report = {
        "schema_version": 1,
        "cohort": {
            "definition": "ANN-02 validation identities from the frozen Stage 04 KIC-grouped validation partition",
            "records": int(len(frame)),
            "stars": int(frame["kic"].nunique()),
            "class_0": int((targets == 0).sum()),
            "class_1": int((targets == 1).sum()),
            "test_accessed": False,
        },
        "ann01_metrics": compute_classification_metrics(targets, frame["ann01_score"].to_numpy()),
        "ann02_metrics": compute_classification_metrics(targets, frame["ann02_score"].to_numpy()),
        "agreement": {
            "records": int(frame["models_agree"].sum()),
            "fraction": float(frame["models_agree"].mean()),
            "divergences": int((~frame["models_agree"]).sum()),
        },
        "rows": rows,
        "limitations": [
            "Both models are compared only on identical object identities and labels.",
            "This validation cohort was used for ANN-02 checkpoint selection, so it is not an unseen final test.",
            "ANN-01 uses catalog features while ANN-02 uses Q3 luminosity morphology; scores are not calibrated probabilities.",
            "No automatic ensemble is justified or implemented from this small cohort.",
        ],
    }
    if rows_path is not None:
        Path(rows_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(rows_path, index=False)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report
