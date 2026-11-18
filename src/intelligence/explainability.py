"""Model-faithful local and cohort explanations for ANN-01 and ANN-02."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

from src.data.preprocessing import FEATURE_COLUMNS, LABEL_MAPPING, OBJECT_ID_COLUMN, TARGET_COLUMN
from src.deep_learning.dataset import KeplerSignalDataset
from src.deep_learning.training import ANN02Bundle
from src.lab.services import LabModelState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPLAINABILITY_REPORT = PROJECT_ROOT / "experiments" / "stage07" / "explainability.json"


def explain_ann01_local(
    state: LabModelState,
    record: pd.Series | pd.DataFrame,
    epsilon: float = 1e-4,
) -> dict[str, Any]:
    """Explain one ANN-01 score using real finite-difference input sensitivity."""
    frame = record.to_frame().T if isinstance(record, pd.Series) else record.copy()
    if len(frame) != 1:
        raise ValueError("ANN-01 local explanation requires exactly one record")
    transformed = state.preprocessor.transform(frame)
    base_score = float(state.model.predict_proba(transformed)[0, 0])
    rows: list[dict[str, float | str | None]] = []
    assert state.preprocessor.scales is not None
    for index, feature in enumerate(FEATURE_COLUMNS):
        plus = transformed.copy()
        minus = transformed.copy()
        plus[0, index] += epsilon
        minus[0, index] -= epsilon
        plus_score = float(state.model.predict_proba(plus)[0, 0])
        minus_score = float(state.model.predict_proba(minus)[0, 0])
        normalized_gradient = (plus_score - minus_score) / (2.0 * epsilon)
        raw_value = pd.to_numeric(frame.iloc[0][feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "raw_value": None if pd.isna(raw_value) else float(raw_value),
                "normalized_value": float(transformed[0, index]),
                "normalized_gradient": float(normalized_gradient),
                "raw_unit_gradient": float(normalized_gradient / state.preprocessor.scales[index]),
                "local_contribution": float(normalized_gradient * transformed[0, index]),
            }
        )
    rows.sort(key=lambda item: abs(float(item["local_contribution"])), reverse=True)
    return {
        "method": "central finite-difference sensitivity with first-order baseline contribution",
        "score": base_score,
        "baseline": "training-feature mean (zero in normalized coordinates)",
        "features": rows,
        "limitations": [
            "Contributions are local first-order approximations, not causal effects.",
            "Correlated astronomical features can share or redistribute apparent importance.",
        ],
    }


def permutation_importance_ann01(
    state: LabModelState,
    cohort: pd.DataFrame,
    repeats: int = 32,
    seed: int = 2027,
) -> dict[str, Any]:
    """Measure ANN-01 feature importance as validation BCE increase after permutation."""
    labelled = cohort[cohort[TARGET_COLUMN].isin(LABEL_MAPPING)].copy()
    if len(labelled) < 2 or labelled[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Permutation importance requires both labelled classes")
    features = state.preprocessor.transform(labelled)
    targets = labelled[TARGET_COLUMN].map(LABEL_MAPPING).to_numpy(dtype=np.int64)
    baseline_scores = state.model.predict_proba(features).reshape(-1)
    baseline_bce = float(log_loss(targets, baseline_scores, labels=[0, 1]))
    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(FEATURE_COLUMNS):
        increases: list[float] = []
        for _ in range(repeats):
            permuted = features.copy()
            permuted[:, index] = generator.permutation(permuted[:, index])
            scores = state.model.predict_proba(permuted).reshape(-1)
            increases.append(float(log_loss(targets, scores, labels=[0, 1]) - baseline_bce))
        rows.append(
            {
                "feature": feature,
                "mean_bce_increase": float(np.mean(increases)),
                "std_bce_increase": float(np.std(increases)),
            }
        )
    rows.sort(key=lambda item: item["mean_bce_increase"], reverse=True)
    return {
        "method": "permutation importance on a labelled validation cohort",
        "records": int(len(labelled)),
        "baseline_bce": baseline_bce,
        "repeats": repeats,
        "seed": seed,
        "features": rows,
        "test_accessed": False,
    }


def _regions(length: int, count: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, min(count, length) + 1, dtype=int)
    return [(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:], strict=True) if end > start]


def explain_ann02_signal(
    bundle: ANN02Bundle,
    sample: dict[str, Any],
    global_regions: int = 40,
    local_regions: int = 20,
) -> dict[str, Any]:
    """Compute real gradients and region occlusion deltas for one ANN-02 signal."""
    model = bundle.model
    model.eval()
    global_view = sample["global_view"].unsqueeze(0).to(bundle.device).clone().detach().requires_grad_(True)
    local_view = sample["local_view"].unsqueeze(0).to(bundle.device).clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    score = torch.sigmoid(model(global_view, local_view))[0]
    score.backward()
    base_score = float(score.detach().cpu())
    gradient_payload = {
        "global": (global_view.grad[0, 0] * global_view.detach()[0, 0]).cpu().numpy(),
        "local": (local_view.grad[0, 0] * local_view.detach()[0, 0]).cpu().numpy(),
    }

    region_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for view_name, source, other, count in (
            ("global", global_view.detach(), local_view.detach(), global_regions),
            ("local", local_view.detach(), global_view.detach(), local_regions),
        ):
            length = int(source.shape[-1])
            for start, end in _regions(length, count):
                occluded = source.clone()
                occluded[:, 0, start:end] = 0.0
                if view_name == "global":
                    changed = torch.sigmoid(model(occluded, other))[0]
                else:
                    changed = torch.sigmoid(model(other, occluded))[0]
                mask_fraction = float(source[0, 1, start:end].mean().cpu())
                region_rows.append(
                    {
                        "view": view_name,
                        "start_bin": start,
                        "end_bin": end,
                        "start_coordinate": float(start / length - 0.5),
                        "end_coordinate": float(end / length - 0.5),
                        "observed_fraction": mask_fraction,
                        "score_without_region": float(changed.cpu()),
                        "score_delta": float(base_score - float(changed.cpu())),
                        "mean_abs_gradient_x_input": float(np.mean(np.abs(gradient_payload[view_name][start:end]))),
                    }
                )
    region_rows.sort(key=lambda item: abs(item["score_delta"]), reverse=True)
    return {
        "method": "median-baseline region occlusion plus input-gradient sensitivity",
        "object_id": str(sample["object_id"]),
        "kic": int(sample["kic"]),
        "score": base_score,
        "regions": region_rows,
        "saliency": {name: values.tolist() for name, values in gradient_payload.items()},
        "limitations": [
            "Occlusion measures model sensitivity to replacing a region by normalized median flux; it is not causal astronomy.",
            "Gradient saliency is local and can be unstable in saturated or ReLU-inactive regions.",
            "A high-influence region is not by itself evidence of a planetary transit.",
        ],
    }


def build_explainability_report(
    ann01_state: LabModelState,
    ann02_bundle: ANN02Bundle,
    catalog_frame: pd.DataFrame,
    report_path: Path = DEFAULT_EXPLAINABILITY_REPORT,
) -> dict[str, Any]:
    """Persist cohort importance and representative real local explanations.

    The representative set contains one validation object from each labelled
    class plus the unlabelled candidate, when those strata are available.
    """
    dataset = KeplerSignalDataset(split=None)
    samples = [dataset[index] for index in range(len(dataset))]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    validation_ids = {str(sample["object_id"]) for sample in validation}
    validation_frame = catalog_frame[catalog_frame[OBJECT_ID_COLUMN].astype(str).isin(validation_ids)]
    importance = permutation_importance_ann01(
        ann01_state,
        validation_frame,
        repeats=32,
        seed=2027,
    )

    representatives: list[dict[str, Any]] = []
    for disposition in ("CONFIRMED", "FALSE POSITIVE", "CANDIDATE"):
        eligible = samples if disposition == "CANDIDATE" else validation
        expected_label = {"CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": -1}[disposition]
        match = next(
            (sample for sample in eligible if int(sample["label"].item()) == expected_label),
            None,
        )
        if match is None:
            continue
        object_id = str(match["object_id"])
        catalog_match = catalog_frame[catalog_frame[OBJECT_ID_COLUMN].astype(str).eq(object_id)]
        representatives.append(
            {
                "object_id": object_id,
                "kic": int(match["kic"]),
                "split": str(match["split"]),
                "catalog_disposition": disposition,
                "ann01": (
                    explain_ann01_local(ann01_state, catalog_match.iloc[0])
                    if len(catalog_match) == 1
                    else None
                ),
                "ann02": explain_ann02_signal(ann02_bundle, match),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "ann01_permutation_importance": importance,
        "representative_local_explanations": representatives,
        "guardrails": {
            "test_accessed": False,
            "candidate_used_as_label": False,
            "causal_or_astrophysical_claim": False,
        },
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload
