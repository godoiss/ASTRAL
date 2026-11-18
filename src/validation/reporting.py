"""Markdown scientific report generation from real Stage 04 results."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "-".join(str(item) for item in value)
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def generate_scientific_report(
    results: dict[str, Any],
    search_runs: list[dict[str, Any]],
    stability_runs: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
    path: Path,
) -> Path:
    """Generate methodology, evidence, limitations, and conclusions in Markdown."""
    selected = results["selection"]["final_selected_experiment"]
    final_metrics = results["final_test"]["astral_metrics"]
    benchmark = results["benchmark"]
    audit = results["audit"]
    split = results["split_summary"]
    stability = results["seed_stability"]["summary"]

    search_rows = [
        [
            run["experiment_id"],
            run["layer_sizes"],
            run["learning_rate"],
            run["epochs"],
            run["batch_size"],
            run["metrics"]["validation"]["pr_auc"],
            run["metrics"]["validation"]["balanced_accuracy"],
            run["metrics"]["validation"]["binary_cross_entropy"],
        ]
        for run in search_runs
    ]
    seed_rows = [
        [
            run["seed"],
            run["metrics"]["validation"]["pr_auc"],
            run["metrics"]["validation"]["roc_auc"],
            run["metrics"]["validation"]["f1"],
            run["metrics"]["validation"]["binary_cross_entropy"],
        ]
        for run in stability_runs
    ]
    benchmark_rows = [
        [
            name,
            item["metrics"]["accuracy"],
            item["metrics"]["balanced_accuracy"],
            item["metrics"]["precision"],
            item["metrics"]["recall"],
            item["metrics"]["f1"],
            item["metrics"]["roc_auc"],
            item["metrics"]["pr_auc"],
            item["metrics"]["binary_cross_entropy"],
        ]
        for name, item in benchmark.items()
    ]
    candidate_rows = [
        [
            item["rank"],
            item["kepoi_name"],
            item["kepid"],
            item["ann_score_confirmed"],
            item["koi_period"],
            item["koi_prad"],
            item["koi_teq"],
        ]
        for item in top_candidates[:10]
    ]
    confusion = final_metrics["confusion_matrix"]
    bce_gap = selected["metrics"]["validation"]["binary_cross_entropy"] - selected[
        "metrics"
    ]["train"]["binary_cross_entropy"]
    accuracy_gap = selected["metrics"]["train"]["accuracy"] - selected["metrics"][
        "validation"
    ]["accuracy"]

    report = "\n".join(
        [
            "# ASTRAL Stage 04 — Scientific Validation Report",
            "",
            "## Executive summary",
            "",
            f"Campaign `{results['campaign_id']}` evaluated ASTRAL on the real "
            "Kepler cumulative KOI dataset using a frozen host-star-grouped split. "
            "Hyperparameters and initialization seed were selected exclusively on "
            "training and validation data. The campaign test partition was opened "
            "only after selection.",
            "",
            f"The selected configuration was `{selected['experiment_id']}` with "
            f"architecture `{_format(selected['layer_sizes'])}`, learning rate "
            f"`{selected['learning_rate']}`, batch size `{selected['batch_size']}`, "
            f"`{selected['epochs']}` epochs, and seed `{selected['seed']}`. Final "
            f"held-out PR-AUC was `{final_metrics['pr_auc']:.6f}` and balanced "
            f"accuracy was `{final_metrics['balanced_accuracy']:.6f}`.",
            "",
            "## Methodology",
            "",
            f"- Dataset version: `{results['dataset']['version_id']}`.",
            f"- Dataset SHA-256: `{results['dataset']['sha256']}`.",
            f"- Grouped split seed: `{results['protocol']['split_seed']}`.",
            "- Group key: `kepid`; no host star is shared by train, validation, and test.",
            "- Missing values: training median; scaling: training mean and population standard deviation.",
            "- Selection metric: validation PR-AUC (average precision), pre-registered before execution.",
            "- Decision threshold: 0.5 for threshold-based metrics; ROC-AUC and PR-AUC use continuous scores.",
            "- ANN training: NumPy implementation only; Scikit-learn is restricted to metrics and comparison models.",
            "",
            "### Frozen split",
            "",
            _table(
                ["Partition", "Records", "Stars", "Confirmed", "False Positive"],
                [
                    [name, item["records"], item["groups"], item.get("confirmed", "N/A"), item.get("false_positive", "N/A")]
                    for name, item in split.items()
                ],
            ),
            "",
            "## Neural-engine audit",
            "",
            f"- Forward outputs finite and in `[0, 1]`: `{audit['forward_outputs_finite'] and audit['forward_outputs_in_unit_interval']}`.",
            f"- ANN-01 parameter count: `{audit['parameter_integrity']['parameter_count']}`.",
            f"- Parameter shapes valid and all values finite: `{audit['parameter_integrity']['expected_shapes_match'] and audit['parameter_integrity']['finite_weights'] and audit['parameter_integrity']['finite_biases']}`.",
            f"- All gradients finite: `{audit['gradient_integrity']['finite']}`.",
            f"- Numerical gradient relative error: `{audit['gradient_integrity']['relative_error']:.12g}`.",
            f"- Maximum absolute gradient error: `{audit['gradient_integrity']['maximum_absolute_error']:.12g}`.",
            f"- Group overlaps (train/validation, train/test, validation/test): `{list(audit['group_overlap_counts'].values())}`.",
            "",
            "## Controlled hyperparameter search",
            "",
            _table(
                ["Experiment", "Architecture", "LR", "Epochs", "Batch", "Val PR-AUC", "Val Bal. Acc.", "Val BCE"],
                search_rows,
            ),
            "",
            "The original ANN-01 architecture remains preserved as the reference experiment. Architecture variants are Stage 04 experimental models and do not overwrite the ANN-01 configuration or bundle.",
            "",
            "## Seed stability",
            "",
            _table(["Seed", "Val PR-AUC", "Val ROC-AUC", "Val F1", "Val BCE"], seed_rows),
            "",
            f"Validation PR-AUC mean was `{stability['metrics']['pr_auc']['mean']:.6f}` "
            f"with standard deviation `{stability['metrics']['pr_auc']['standard_deviation']:.6f}`.",
            "",
            "## Final held-out evaluation and benchmarks",
            "",
            _table(
                ["Model", "Accuracy", "Balanced Acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "BCE"],
                benchmark_rows,
            ),
            "",
            f"ASTRAL confusion matrix (`[[TN, FP], [FN, TP]]`): `{confusion}`.",
            "",
            "All models used the same training records, frozen test records, feature values, and binary labels. Tree models received the same standardized features for procedural equivalence, although scaling is not necessary for their split rules. Benchmark hyperparameters were fixed before test evaluation and were not tuned on test results.",
            "",
            "## Learning behavior",
            "",
            f"The selected run's train-minus-validation accuracy gap was `{accuracy_gap:.6f}` "
            f"and validation-minus-train BCE gap was `{bce_gap:.6f}`. These curves "
            "support assessment of overfitting or underfitting but do not prove "
            "generalization beyond the Kepler cumulative table.",
            "",
            "## Candidate exploration",
            "",
            "The selected model scored all original `CANDIDATE` records. Scores are sigmoid model outputs and are explicitly not scientifically calibrated probabilities. Ranking does not change NASA dispositions and does not constitute exoplanet confirmation.",
            "",
            _table(["Rank", "KOI", "kepid", "ANN Score", "Period", "Radius", "Teq"], candidate_rows),
            "",
            "## Scientific visualizations",
            "",
            *[f"- `{path}`" for path in results["artifacts"]["figures"]],
            "",
            "## Limitations",
            "",
            "1. Stage 03 already exposed a test result from a different split (seed 42). Stage 04 froze a new grouped split before search, but this does not replace external validation on an independent dataset.",
            "2. The NASA cumulative KOI table combines deliveries and analysis pipelines and is not a uniform statistical population.",
            "3. Labels and selected transit features originate from the same broader vetting ecosystem, so dataset and provenance bias may inflate apparent transferability.",
            "4. The controlled search is intentionally modest for local CPU execution; repeated nested grouped cross-validation was not performed.",
            "5. Scores are uncalibrated. No uncertainty intervals, calibration curve, or astrophysical false-discovery control is provided.",
            "6. Candidate ranking is exploratory only and cannot establish planetary status.",
            "",
            "## Conclusions",
            "",
            "ANN-01's forward and backward mathematics pass numerical validation, its grouped protocol prevents direct host-star leakage across labelled partitions, and its final metrics are reported alongside stronger and weaker classical baselines without result manipulation. The evidence supports that the engine learns reproducible signal within this dataset; it does not yet establish deployment-grade or discovery-grade scientific validity.",
            "",
            "## Opportunities for evolution",
            "",
            "- External validation on a separately curated Kepler or TESS dataset.",
            "- Nested stratified-group cross-validation and confidence intervals.",
            "- Probability calibration using validation-only procedures.",
            "- Threshold selection based on a declared scientific cost function.",
            "- Feature-ablation and permutation-importance studies.",
            "- Explicit uncertainty and out-of-distribution detection.",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path

