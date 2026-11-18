"""End-to-end Stage 04 scientific validation campaign."""

from __future__ import annotations

import json
import logging
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.data.preprocessing import (
    FEATURE_COLUMNS,
    LABEL_MAPPING,
    DataPartition,
    PreparedDataset,
    export_split_manifest,
    prepare_kepler_dataset,
)
from src.validation.audit import audit_engine_and_dataset
from src.validation.benchmarks import evaluate_benchmarks
from src.validation.config import CampaignConfig
from src.validation.experiment import (
    NeuralExperimentResult,
    run_neural_experiment,
    select_by_validation_pr_auc,
    summarize_seed_stability,
)
from src.validation.metrics import compute_classification_metrics, compute_curve_data
from src.validation.reporting import generate_scientific_report
from src.validation.tracking import (
    ExperimentTracker,
    dataset_version,
    write_json_atomic,
)
from src.validation.visualizations import (
    plot_activation_distributions,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_model_benchmark,
    plot_precision_recall_curves,
    plot_prediction_score_distributions,
    plot_roc_curves,
    plot_weight_distributions,
)


LOGGER = logging.getLogger(__name__)


def _partition_summary(partition: DataPartition) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "records": partition.size,
        "groups": partition.group_count,
    }
    if partition.targets is not None:
        confirmed = int(np.sum(partition.targets == 1.0))
        summary["confirmed"] = confirmed
        summary["false_positive"] = partition.size - confirmed
    return summary


def _result_summary(result: NeuralExperimentResult) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.record.items()
        if key not in {"parameters", "history"}
    }


def _candidate_ranking(
    data_path: Path,
    candidate_scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    columns = ["kepid", "kepoi_name", "koi_disposition", *FEATURE_COLUMNS]
    frame = pd.read_csv(data_path, usecols=columns)
    candidates = frame[frame["koi_disposition"] == "CANDIDATE"].copy()
    if len(candidates) != len(candidate_scores):
        raise RuntimeError("Candidate identities and transformed features are misaligned")
    candidates["ann_score_confirmed"] = candidate_scores
    candidates["threshold_decision"] = np.where(
        candidate_scores >= threshold,
        "POSITIVE_SIDE",
        "NEGATIVE_SIDE",
    )
    candidates = candidates.sort_values(
        "ann_score_confirmed",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def _ranking_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def run_campaign(
    config_path: Path,
    data_path: Path,
    output_directory: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Execute search, selection, one final test evaluation, and reporting."""
    config = CampaignConfig.load(config_path)
    output_directory = Path(output_directory)
    if (output_directory / "experiment_registry.json").exists():
        raise FileExistsError(
            f"Campaign output already exists and is immutable: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    figures_directory = output_directory / "figures"

    raw_shape = pd.read_csv(data_path, usecols=["kepoi_name"]).shape
    raw_columns = len(pd.read_csv(data_path, nrows=0).columns)
    identity = dataset_version(
        data_path,
        records=raw_shape[0],
        columns=raw_columns,
    )
    dataset = prepare_kepler_dataset(
        data_path,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
    )
    split_summary = {
        "train": _partition_summary(dataset.train),
        "validation": _partition_summary(dataset.validation),
        "test": _partition_summary(dataset.test),
        "candidates": _partition_summary(dataset.candidates),
    }
    tracker = ExperimentTracker(output_directory, config.campaign_id, identity)
    write_json_atomic(output_directory / "campaign_config.json", config.to_dict())
    export_split_manifest(dataset, output_directory / "split_manifest.csv")

    audit = audit_engine_and_dataset(
        dataset,
        dataset_identity=identity,
        split_seed=config.split_seed,
    )
    write_json_atomic(output_directory / "engine_audit.json", audit)
    LOGGER.info(
        "Stage 04 audit completed",
        extra={"event": "stage04_audit", "details": audit["group_overlap_counts"]},
    )

    # Search receives only train and validation. The test partition is not passed.
    search_results: list[NeuralExperimentResult] = []
    for spec in config.experiments:
        result = run_neural_experiment(
            spec,
            dataset.train,
            dataset.validation,
            threshold=config.classification_threshold,
            tracker=tracker,
            dataset_version_id=identity["version_id"],
        )
        search_results.append(result)
        LOGGER.info(
            "Controlled experiment completed",
            extra={
                "event": "experiment_completed",
                "details": {
                    "experiment_id": spec.experiment_id,
                    "validation_pr_auc": result.record["metrics"]["validation"][
                        "pr_auc"
                    ],
                },
            },
        )
    search_selected = select_by_validation_pr_auc(search_results)

    stability_results: list[NeuralExperimentResult] = []
    for index, seed in enumerate(config.stability_seeds, start=1):
        stability_spec = search_selected.spec.with_seed(
            seed,
            experiment_id=f"stability-{index:03d}-seed-{seed}",
        )
        stability_results.append(
            run_neural_experiment(
                stability_spec,
                dataset.train,
                dataset.validation,
                threshold=config.classification_threshold,
                tracker=tracker,
                dataset_version_id=identity["version_id"],
            )
        )
    stability_summary = summarize_seed_stability(stability_results)
    final_selected = select_by_validation_pr_auc(stability_results)

    # This is the campaign's first and only performance evaluation on its test split.
    if dataset.test.targets is None:
        raise RuntimeError("Frozen test targets are unavailable")
    test_scores = final_selected.model.predict_proba(dataset.test.features).reshape(-1)
    test_metrics = compute_classification_metrics(
        dataset.test.targets,
        test_scores,
        threshold=config.classification_threshold,
    )
    benchmark_results, benchmark_scores = evaluate_benchmarks(
        dataset.train,
        dataset.test,
        astral_scores=test_scores,
        threshold=config.classification_threshold,
        seed=final_selected.spec.seed,
        settings=config.benchmarks,
    )
    curve_data = {
        name: compute_curve_data(dataset.test.targets, scores)
        for name, scores in benchmark_scores.items()
    }

    candidate_scores = final_selected.model.predict_proba(
        dataset.candidates.features
    ).reshape(-1)
    ranking = _candidate_ranking(
        data_path,
        candidate_scores,
        threshold=config.classification_threshold,
    )
    ranking.to_csv(output_directory / "candidate_ranking.csv", index=False)
    ranking_records = _ranking_records(ranking)
    write_json_atomic(output_directory / "candidate_ranking.json", ranking_records)

    # Capture actual held-out activations before restoring a train-context snapshot.
    final_selected.model.forward(dataset.test.features, retain_cache=True)
    held_out_activations = [item.copy() for item in final_selected.model.activations]

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_learning_curves(
            final_selected.model.training_history,
            figures_directory,
            config.plot_dpi,
        )
    )
    figure_paths.append(
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            figures_directory,
            config.plot_dpi,
        )
    )
    figure_paths.append(
        plot_roc_curves(
            dataset.test.targets,
            benchmark_scores,
            benchmark_results,
            figures_directory,
            config.plot_dpi,
        )
    )
    figure_paths.append(
        plot_precision_recall_curves(
            dataset.test.targets,
            benchmark_scores,
            benchmark_results,
            figures_directory,
            config.plot_dpi,
        )
    )
    figure_paths.append(
        plot_model_benchmark(
            benchmark_results,
            figures_directory,
            config.plot_dpi,
        )
    )
    figure_paths.append(
        plot_weight_distributions(
            final_selected.model,
            figures_directory,
            config.plot_dpi,
        )
    )
    figure_paths.append(
        plot_activation_distributions(
            held_out_activations,
            figures_directory,
            config.plot_dpi,
        )
    )
    figure_paths.append(
        plot_prediction_score_distributions(
            dataset.test.targets,
            test_scores,
            candidate_scores,
            figures_directory,
            config.plot_dpi,
        )
    )

    selected_directory = output_directory / "selected_model"
    selected_directory.mkdir(parents=True, exist_ok=True)
    final_selected.model.save(selected_directory / "model.npz")
    dataset.preprocessor.save(selected_directory / "preprocessor.json")
    write_json_atomic(
        selected_directory / "experiment_config.json",
        final_selected.spec.to_dict(),
    )
    write_json_atomic(
        selected_directory / "history.json",
        final_selected.model.training_history,
    )
    write_json_atomic(selected_directory / "test_metrics.json", test_metrics)

    snapshot_size = min(16, dataset.train.size)
    assert dataset.train.targets is not None
    snapshot_scores = final_selected.model.forward(
        dataset.train.features[:snapshot_size],
        retain_cache=True,
    )
    final_selected.model.backward(dataset.train.targets[:snapshot_size])
    final_selected.model.export_snapshot(
        selected_directory / "network_snapshot.json",
        metadata={
            "context": "real_training_sample_after_stage04_selection",
            "object_ids": dataset.train.object_ids[:snapshot_size].tolist(),
            "group_ids": dataset.train.group_ids[:snapshot_size].tolist(),
            "targets": dataset.train.targets[:snapshot_size].reshape(-1).tolist(),
            "scores": snapshot_scores.reshape(-1).tolist(),
        },
    )

    relative_figures = [
        str(path.relative_to(output_directory)).replace("\\", "/")
        for path in figure_paths
    ]
    results: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": config.campaign_id,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": identity,
        "protocol": config.to_dict(),
        "audit": audit,
        "split_summary": split_summary,
        "search": {
            "runs": [_result_summary(item) for item in search_results],
            "selected_experiment_id": search_selected.spec.experiment_id,
        },
        "seed_stability": {
            "runs": [_result_summary(item) for item in stability_results],
            "summary": stability_summary,
        },
        "selection": {
            "metric": config.selection_metric,
            "final_selected_experiment": final_selected.record,
        },
        "final_test": {
            "opened_after_selection": True,
            "evaluation_count_in_campaign": 1,
            "astral_metrics": test_metrics,
            "curve_data": curve_data,
        },
        "benchmark": benchmark_results,
        "candidate_exploration": {
            "records": len(ranking),
            "score_interpretation": "uncalibrated_model_score",
            "top_candidate": ranking_records[0],
            "csv": "candidate_ranking.csv",
            "json": "candidate_ranking.json",
        },
        "artifacts": {
            "figures": relative_figures,
            "selected_model": "selected_model/model.npz",
            "split_manifest": "split_manifest.csv",
        },
        "limitations": [
            audit["prior_test_usage"]["limitation"],
            "The cumulative KOI table is not an external independent validation set.",
            "Candidate scores are not calibrated scientific probabilities.",
        ],
    }
    write_json_atomic(output_directory / "scientific_validation_results.json", results)
    tracker.attach_campaign_artifacts(
        [
            *relative_figures,
            "candidate_ranking.csv",
            "candidate_ranking.json",
            "selected_model/model.npz",
            "selected_model/network_snapshot.json",
            "scientific_validation_results.json",
        ]
    )

    generated_report = generate_scientific_report(
        results,
        [item.record for item in search_results],
        [item.record for item in stability_results],
        ranking_records[:10],
        report_path,
    )
    shutil.copyfile(generated_report, output_directory / "scientific_report.md")
    LOGGER.info(
        "Stage 04 scientific validation completed",
        extra={
            "event": "stage04_completed",
            "details": {
                "selected_experiment": final_selected.spec.experiment_id,
                "test_metrics": test_metrics,
                "output_directory": str(output_directory),
            },
        },
    )
    return results

