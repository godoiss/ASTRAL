"""Build reproducible scientific dossiers from persisted Stage 10 evidence."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.candidate_intelligence.catalogs import ExoplanetArchiveClient
from src.discovery.catalog import _period_relation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CAMPAIGN = PROJECT_ROOT / "experiments" / "stage10" / "campaigns" / "stage10-kepler-reference-final" / "campaign.json"
STAGE11_ROOT = PROJECT_ROOT / "experiments" / "stage11"
INDEX_PATH = STAGE11_ROOT / "candidate_intelligence.json"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _catalog_records(rows: list[dict[str, Any]], kic: int, period: float, prefix: str) -> list[dict[str, Any]]:
    period_key = f"{prefix}_period"
    records: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("kepid", -1)) != kic:
            continue
        catalog_period = _finite(row.get(period_key))
        projected = {key: value for key, value in row.items() if value is not None}
        projected["period_relation"] = _period_relation(period, catalog_period, 0.02) if catalog_period else None
        projected["period_ratio"] = period / catalog_period if catalog_period and catalog_period > 0 else None
        records.append(projected)
    return records


def _priority(detection: dict[str, Any], processing: dict[str, Any], associated: bool, koi: list[dict[str, Any]], tce: list[dict[str, Any]]) -> dict[str, Any]:
    depth = max(abs(float(detection.get("depth_fraction", 0.0))), 1e-12)
    mad = _finite(detection.get("event_depth_mad"))
    stability = 1.0 if mad is None else 1.0 - _clamp(mad / depth)
    repeatability = 0.55 * _clamp(float(detection.get("observed_events", 0)) / 5.0) + 0.45 * stability
    observational_quality = _clamp(float(processing.get("retention_fraction", 0.0)))
    temporal_coverage = 0.5 * _clamp(float(detection.get("event_coverage", 0.0))) + 0.5 * _clamp(float(detection.get("phase_coverage", 0.0)))
    boundary = any("search boundary" in str(item).lower() for item in detection.get("limitations", []))
    processing_robustness = 0.7 * observational_quality + 0.3 * (0.0 if boundary else 1.0)
    odd_even = abs(_finite(detection.get("odd_even_sigma")) or 0.0)
    secondary = max(_finite(detection.get("secondary_sigma")) or 0.0, 0.0)
    harmonic_alias = any(item.get("relation") for item in detection.get("aliases", [])[:3])
    transit_consistency = 1.0 - _clamp(0.5 * odd_even / 6.0 + 0.35 * secondary / 6.0 + (0.15 if harmonic_alias else 0.0))
    fp_flags = sum(int(record.get(key) or 0) for record in koi for key in ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"))
    tce_false_positive = any(str(record.get("av_pred_class", "")).upper() in {"AFP", "NTP"} for record in tce)
    fp_evidence = _clamp((0.65 if detection.get("possible_eclipsing_binary") else 0.0) + min(fp_flags, 2) * 0.18 + (0.25 if tce_false_positive else 0.0))
    signal_quality = _clamp(float(detection.get("depth_snr", 0.0)) / 12.0)
    unresolved_value = 0.3 if associated else 1.0
    score = 100.0 * _clamp(
        0.24 * signal_quality
        + 0.15 * observational_quality
        + 0.16 * repeatability
        + 0.15 * temporal_coverage
        + 0.10 * processing_robustness
        + 0.12 * transit_consistency
        + 0.08 * unresolved_value
        - 0.25 * fp_evidence
        - (0.15 if boundary else 0.0)
    )
    tier = "high" if score >= 70 else "medium" if score >= 50 else "low"
    return {
        "score": round(score, 3),
        "tier": tier,
        "meaning": "human_review_priority_not_planet_probability",
        "components": {
            "signal_quality": round(signal_quality, 4),
            "observational_quality": round(observational_quality, 4),
            "repeatability": round(repeatability, 4),
            "temporal_coverage": round(temporal_coverage, 4),
            "processing_robustness": round(processing_robustness, 4),
            "transit_consistency": round(transit_consistency, 4),
            "false_positive_evidence": round(fp_evidence, 4),
            "catalog_unresolved_value": unresolved_value,
            "search_boundary_penalty": 1.0 if boundary else 0.0,
        },
        "formula_version": "stage11-review-priority-v1",
    }


def _evidence(detection: dict[str, Any], crossmatch: dict[str, Any], koi: list[dict[str, Any]], tce: list[dict[str, Any]]) -> dict[str, list[str]]:
    favorable = [
        f"BLS screening threshold passed with depth SNR {float(detection['depth_snr']):.2f}.",
        f"{int(detection['observed_events'])} events were measured with {100 * float(detection['event_coverage']):.1f}% temporal coverage.",
        f"Phase-bin coverage is {100 * float(detection['phase_coverage']):.1f}% in the retained Quarter 3 series.",
    ]
    contrary: list[str] = []
    if detection.get("possible_eclipsing_binary"):
        contrary.append("Discovery diagnostics are compatible with an eclipsing-binary explanation.")
    if (_finite(detection.get("odd_even_sigma")) or 0.0) >= 3.0:
        contrary.append(f"Odd/even depth difference reaches {float(detection['odd_even_sigma']):.2f} sigma.")
    if (_finite(detection.get("secondary_sigma")) or 0.0) >= 3.0:
        contrary.append(f"A phase-0.5 secondary event reaches {float(detection['secondary_sigma']):.2f} sigma.")
    if any("search boundary" in str(item).lower() for item in detection.get("limitations", [])):
        contrary.append("The strongest BLS peak is on the configured period-search boundary.")
    fp_flags = [key for record in koi for key in ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec") if int(record.get(key) or 0)]
    if fp_flags:
        contrary.append("Official KOI records contain false-positive flags: " + ", ".join(sorted(set(fp_flags))) + ".")
    if any(str(record.get("av_pred_class", "")).upper() in {"AFP", "NTP"} for record in tce):
        contrary.append("At least one official DR25 TCE record carries an automated false-positive/non-transit class.")
    inconclusive = [
        "Only Kepler Quarter 3 was analyzed; multi-quarter repeatability was not tested.",
        "No centroid, pixel-level contamination, or nearby-source analysis was performed.",
        "The BLS depth SNR has not been calibrated with bootstrap or injection-recovery tests.",
        "A missing KOI/TCE period association does not establish novelty or planetary nature.",
    ]
    if crossmatch.get("period_association_found"):
        favorable.append("The measured period has a simple relation to an existing KOI record on this host.")
    return {"favorable": favorable, "contrary": contrary, "inconclusive": inconclusive}


def _markdown(dossier: dict[str, Any]) -> str:
    measured = dossier["measured_event"]
    lines = [
        f"# ASTRAL Candidate Intelligence — KIC {dossier['kic']}", "",
        f"Campaign: `{dossier['campaign_id']}`", "",
        f"Investigation state: **{dossier['investigation_state'].replace('_', ' ')}**", "",
        f"Human-review priority: **{dossier['priority']['score']:.3f}/100 ({dossier['priority']['tier']})**", "",
        "## Measured screening event", "",
        f"- Period: {measured['period_days']:.8f} days",
        f"- Duration: {measured['duration_hours']:.3f} hours",
        f"- Depth: {measured['depth_ppm']:.3f} ppm",
        f"- Depth SNR: {measured['depth_snr']:.3f}",
        f"- Observed events: {measured['observed_events']}", "",
        "## Evidence", "",
    ]
    for group in ("favorable", "contrary", "inconclusive"):
        lines.extend([f"### {group.title()}", ""])
        lines.extend(f"- {item}" for item in dossier["evidence"][group])
        lines.append("")
    lines.extend([
        "## Scientific boundary", "",
        "This dossier prioritizes human review. It does not report a planet probability, confirmation, or discovery.", "",
    ])
    return "\n".join(lines)


def build_candidate_intelligence(
    *,
    campaign_path: Path = REFERENCE_CAMPAIGN,
    output_root: Path = STAGE11_ROOT,
    catalog_client: ExoplanetArchiveClient | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Investigate all real screening detections and persist Stage 11 dossiers."""
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    detections = [item for item in campaign.get("results", []) if item.get("detection", {}).get("status") == "screening_detection"]
    if len(detections) != int(campaign.get("summary", {}).get("screening_detections", -1)):
        raise ValueError("campaign screening-detection count is inconsistent")
    if not detections:
        raise ValueError("campaign contains no screening detections")
    kics = [int(item["target"]["kic"]) for item in detections]
    if len(kics) != len(set(kics)):
        raise ValueError("campaign contains duplicate screening hosts")
    verification = (catalog_client or ExoplanetArchiveClient()).verify(kics)
    generated = generated_at_utc or datetime.now(UTC).isoformat()
    dossiers: list[dict[str, Any]] = []
    for item in detections:
        kic = int(item["target"]["kic"])
        detection = item["detection"]
        period = float(detection["period_days"])
        koi = _catalog_records(verification["koi_records"], kic, period, "koi")
        tce = _catalog_records(verification["tce_records"], kic, period, "tce")
        associated = bool(item["crossmatch"]["period_association_found"])
        priority = _priority(detection, item["processing"], associated, koi, tce)
        dossier = {
            "schema_version": 1,
            "dossier_id": f"{campaign['campaign_id']}-kic-{kic}",
            "campaign_id": campaign["campaign_id"],
            "generated_at_utc": generated,
            "kic": kic,
            "selection_reference": item["target"].get("selection_reference"),
            "mission": item["target"].get("mission", "Kepler"),
            "investigation_state": "catalog_associated" if associated else "unassociated_inconclusive",
            "interpretation": item["interpretation"],
            "observation": {
                "source_kind": item["raw_provenance"].get("source_kind"),
                "products": item["raw_provenance"].get("products", []),
                "raw_sha256": item["raw_provenance"].get("raw_sha256"),
                "start_bkjd": item["processing"].get("observation_start_bkjd"),
                "end_bkjd": item["processing"].get("observation_end_bkjd"),
                "span_days": item["processing"].get("observation_span_days"),
                "raw_records": item["processing"].get("raw_records"),
                "retained_records": item["processing"].get("retained_records"),
                "retention_fraction": item["processing"].get("retention_fraction"),
                "quality_rejected": item["processing"].get("quality_rejected"),
                "quarter_bias": "Quarter 3 long cadence only",
            },
            "measured_event": {key: detection.get(key) for key in (
                "period_days", "period_uncertainty_days", "transit_time_bkjd", "transit_time_uncertainty_days",
                "duration_hours", "duration_uncertainty_hours", "depth_ppm", "depth_error_fraction", "depth_snr",
                "expected_events", "observed_events", "event_coverage", "phase_coverage", "event_depths", "event_depth_mad",
            )},
            "diagnostics": {
                "aliases": detection.get("aliases", []),
                "odd_depth": detection.get("odd_depth"),
                "even_depth": detection.get("even_depth"),
                "odd_even_sigma": detection.get("odd_even_sigma"),
                "secondary_depth": detection.get("secondary_depth"),
                "secondary_sigma": detection.get("secondary_sigma"),
                "possible_eclipsing_binary": detection.get("possible_eclipsing_binary", False),
            },
            "catalog_verification": {
                "checked_at_utc": generated,
                "koi_records": koi,
                "tce_records": tce,
                "local_period_association_found": associated,
                "local_associations": item.get("catalog_associations", []),
                "eclipsing_binary_catalog": verification["eclipsing_binary_catalog"],
                "novelty_established": False,
            },
            "instrumental_hypotheses": [
                "Quarter-level systematics or residual trends can produce coherent BLS peaks.",
                "Flux contamination from nearby sources has not been excluded without pixel-level centroid analysis.",
                "The centered rolling-median detrending can attenuate or reshape broad variability.",
            ],
            "evidence": _evidence(detection, item["crossmatch"], koi, tce),
            "priority": priority,
            "neural_compatibility": {
                "ann01": {"available": False, "reason": "The detected event lacks an independently fitted planet radius, equilibrium temperature, and catalog model SNR required by ANN-01."},
                "ann02": {"available": False, "reason": "Discovery Engine arrays are not the validated catalog-centered ANN-02 representation; no out-of-distribution justification exists."},
                "models_invoked": False,
            },
            "review_history": [],
            "limitations": list(dict.fromkeys(detection.get("limitations", []) + [
                "Official cross-checks are identifier and period comparisons, not independent astrophysical validation.",
                "No frozen-test data, retraining, or model selection was used.",
            ])),
            "artifacts": item.get("artifacts", {}),
        }
        dossiers.append(dossier)
    dossiers.sort(key=lambda dossier: dossier["priority"]["score"], reverse=True)
    output = {
        "schema_version": 1,
        "stage": "stage11",
        "campaign_id": campaign["campaign_id"],
        "generated_at_utc": generated,
        "dataset_version": campaign.get("source_versions", {}),
        "summary": {
            "investigated_signals": len(dossiers),
            "catalog_associated": sum(item["investigation_state"] == "catalog_associated" for item in dossiers),
            "unassociated_inconclusive": sum(item["investigation_state"] == "unassociated_inconclusive" for item in dossiers),
            "possible_eclipsing_binary": sum(bool(item["diagnostics"]["possible_eclipsing_binary"]) for item in dossiers),
            "confirmed_discoveries": 0,
            "neural_models_invoked": False,
            "frozen_test_accessed": False,
        },
        "ranking_method": {
            "name": "transparent human-review priority",
            "version": "stage11-review-priority-v1",
            "not_a_probability": True,
        },
        "catalog_sources": verification["sources"],
        "dossiers": dossiers,
        "scientific_boundaries": [
            "Priority scores rank review effort; they are not planet probabilities.",
            "Unassociated signals remain inconclusive and are not discoveries.",
            "ANN-01 and ANN-02 were not invoked because their validated input contracts are not met.",
            "The frozen test partition was not accessed.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    dossier_root = output_root / "dossiers"
    dossier_root.mkdir(parents=True, exist_ok=True)
    for dossier in dossiers:
        stem = f"kic_{dossier['kic']:09d}"
        (dossier_root / f"{stem}.json").write_text(json.dumps(dossier, indent=2, ensure_ascii=False), encoding="utf-8")
        (dossier_root / f"{stem}.md").write_text(_markdown(dossier), encoding="utf-8")
    (output_root / "catalog_verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    encoded = json.dumps(output, indent=2, ensure_ascii=False)
    output["artifact_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    (output_root / "candidate_intelligence.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
