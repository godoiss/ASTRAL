"""High-resolution scientific visualization suite for Stage 04."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from src.core.mlp import MLP


BACKGROUND = "#050816"
PANEL = "#0B1026"
GRID = "#24304F"
TEXT = "#E6F7FF"
MUTED = "#8FA7C6"
CYAN = "#00E5FF"
VIOLET = "#9B5DE5"
BLUE = "#246BFD"
MAGENTA = "#F15BB5"
GREEN = "#00F5A0"
ORANGE = "#FFB703"
SERIES_COLORS = (CYAN, VIOLET, GREEN, ORANGE, MAGENTA)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
        }
    )


def _save(figure: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=BACKGROUND,
    )
    plt.close(figure)
    return path


def plot_learning_curves(
    history: list[dict[str, int | float]],
    output_directory: Path,
    dpi: int,
) -> list[Path]:
    """Create separate high-resolution loss and accuracy figures."""
    _style()
    epochs = [int(item["epoch"]) for item in history]
    paths: list[Path] = []
    for metric, title, ylabel, filename in (
        ("loss", "Training and Validation Loss", "Binary Cross-Entropy", "01_loss.png"),
        ("accuracy", "Training and Validation Accuracy", "Accuracy", "02_accuracy.png"),
    ):
        figure, axis = plt.subplots(figsize=(12, 7))
        axis.plot(
            epochs,
            [float(item[f"train_{metric}"]) for item in history],
            color=CYAN,
            linewidth=2.2,
            label="Training",
        )
        axis.plot(
            epochs,
            [float(item[f"validation_{metric}"]) for item in history],
            color=VIOLET,
            linewidth=2.2,
            label="Validation",
        )
        axis.set_title(title, fontsize=18, pad=16)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        if metric == "accuracy":
            axis.set_ylim(0.0, 1.0)
        axis.grid(True)
        axis.legend()
        paths.append(_save(figure, output_directory / filename, dpi))
    return paths


def plot_confusion_matrix(
    matrix: list[list[int]],
    output_directory: Path,
    dpi: int,
) -> Path:
    """Plot the final held-out ASTRAL confusion matrix."""
    _style()
    values = np.asarray(matrix, dtype=np.int64)
    colormap = LinearSegmentedColormap.from_list("astral", [PANEL, BLUE, CYAN])
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(values, cmap=colormap)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                fontsize=18,
                color=TEXT,
            )
    axis.set_xticks([0, 1], labels=["False Positive", "Confirmed"])
    axis.set_yticks([0, 1], labels=["False Positive", "Confirmed"])
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    axis.set_title("ASTRAL Held-Out Confusion Matrix", fontsize=18, pad=16)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save(figure, output_directory / "03_confusion_matrix.png", dpi)


def plot_roc_curves(
    targets: np.ndarray,
    score_map: dict[str, np.ndarray],
    benchmark_results: dict[str, Any],
    output_directory: Path,
    dpi: int,
) -> Path:
    """Plot ROC curves for ASTRAL and every benchmark."""
    _style()
    figure, axis = plt.subplots(figsize=(10, 8))
    for (name, scores), color in zip(score_map.items(), SERIES_COLORS, strict=True):
        false_positive_rate, true_positive_rate, _ = roc_curve(targets, scores)
        auc = benchmark_results[name]["metrics"]["roc_auc"]
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.0,
            color=color,
            label=f"{name} (AUC={auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color=MUTED, label="Random")
    axis.set_title("Receiver Operating Characteristic", fontsize=18, pad=16)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(True)
    axis.legend(loc="lower right", fontsize=9)
    return _save(figure, output_directory / "04_roc_curve.png", dpi)


def plot_precision_recall_curves(
    targets: np.ndarray,
    score_map: dict[str, np.ndarray],
    benchmark_results: dict[str, Any],
    output_directory: Path,
    dpi: int,
) -> Path:
    """Plot precision-recall curves for all final models."""
    _style()
    figure, axis = plt.subplots(figsize=(10, 8))
    for (name, scores), color in zip(score_map.items(), SERIES_COLORS, strict=True):
        precision, recall, _ = precision_recall_curve(targets, scores)
        auc = benchmark_results[name]["metrics"]["pr_auc"]
        axis.plot(
            recall,
            precision,
            linewidth=2.0,
            color=color,
            label=f"{name} (AP={auc:.3f})",
        )
    prevalence = float(np.mean(targets))
    axis.axhline(
        prevalence,
        linestyle="--",
        color=MUTED,
        label=f"Prevalence ({prevalence:.3f})",
    )
    axis.set_title("Precision-Recall Curve", fontsize=18, pad=16)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(True)
    axis.legend(loc="lower left", fontsize=9)
    return _save(figure, output_directory / "05_precision_recall_curve.png", dpi)


def plot_model_benchmark(
    benchmark_results: dict[str, Any],
    output_directory: Path,
    dpi: int,
) -> Path:
    """Compare threshold and ranking metrics without hiding weak results."""
    _style()
    names = list(benchmark_results)
    metric_names = ("balanced_accuracy", "f1", "roc_auc", "pr_auc")
    labels = ("Balanced Accuracy", "F1", "ROC-AUC", "PR-AUC")
    x_positions = np.arange(len(names))
    width = 0.18
    figure, axis = plt.subplots(figsize=(14, 8))
    for index, (metric_name, label, color) in enumerate(
        zip(metric_names, labels, (CYAN, VIOLET, GREEN, ORANGE), strict=True)
    ):
        values = [
            benchmark_results[name]["metrics"][metric_name] for name in names
        ]
        axis.bar(
            x_positions + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
            alpha=0.9,
        )
    axis.set_xticks(x_positions, labels=names, rotation=18, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Held-Out Model Benchmark", fontsize=18, pad=16)
    axis.grid(True, axis="y")
    axis.legend(ncol=2)
    return _save(figure, output_directory / "06_model_benchmark.png", dpi)


def plot_weight_distributions(
    model: MLP,
    output_directory: Path,
    dpi: int,
) -> Path:
    """Plot the selected model's real trained connection weights."""
    _style()
    figure, axis = plt.subplots(figsize=(12, 7))
    for index, (weights, color) in enumerate(
        zip(model.weights, SERIES_COLORS, strict=False), start=1
    ):
        axis.hist(
            weights.reshape(-1),
            bins=28,
            alpha=0.52,
            color=color,
            label=f"Layer {index} ({weights.size} weights)",
        )
    axis.axvline(0.0, color=TEXT, linewidth=1.0, alpha=0.7)
    axis.set_title("Trained Weight Distribution", fontsize=18, pad=16)
    axis.set_xlabel("Weight Value")
    axis.set_ylabel("Frequency")
    axis.grid(True)
    axis.legend()
    return _save(figure, output_directory / "07_weight_distribution.png", dpi)


def plot_activation_distributions(
    activations: list[np.ndarray],
    output_directory: Path,
    dpi: int,
) -> Path:
    """Plot real held-out neuron activations from each non-input layer."""
    _style()
    figure, axis = plt.subplots(figsize=(12, 7))
    for index, (values, color) in enumerate(
        zip(activations[1:], SERIES_COLORS, strict=False), start=1
    ):
        axis.hist(
            values.reshape(-1),
            bins=45,
            density=True,
            alpha=0.45,
            color=color,
            label=f"Layer {index}",
        )
    axis.set_title("Held-Out Neuron Activation Distribution", fontsize=18, pad=16)
    axis.set_xlabel("Activation Value")
    axis.set_ylabel("Density")
    axis.grid(True)
    axis.legend()
    return _save(figure, output_directory / "08_activation_distribution.png", dpi)


def plot_prediction_score_distributions(
    test_targets: np.ndarray,
    test_scores: np.ndarray,
    candidate_scores: np.ndarray,
    output_directory: Path,
    dpi: int,
) -> Path:
    """Plot test scores by true class and unlabelled candidate scores."""
    _style()
    targets = np.asarray(test_targets).reshape(-1)
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.hist(
        test_scores[targets == 0],
        bins=45,
        density=True,
        alpha=0.52,
        color=VIOLET,
        label="Test: False Positive",
    )
    axis.hist(
        test_scores[targets == 1],
        bins=45,
        density=True,
        alpha=0.52,
        color=CYAN,
        label="Test: Confirmed",
    )
    axis.hist(
        candidate_scores,
        bins=45,
        density=True,
        histtype="step",
        linewidth=2.2,
        color=ORANGE,
        label="Unlabelled Candidates",
    )
    axis.axvline(0.5, linestyle="--", color=TEXT, label="Decision Threshold")
    axis.set_title("ASTRAL Prediction Score Distribution", fontsize=18, pad=16)
    axis.set_xlabel("ANN Score (Not Calibrated Probability)")
    axis.set_ylabel("Density")
    axis.set_xlim(0.0, 1.0)
    axis.grid(True)
    axis.legend()
    return _save(figure, output_directory / "09_prediction_score_distribution.png", dpi)

