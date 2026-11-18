"""Scientific visualizations built from real ANN-01 state."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.core.mlp import MLP
from src.data.preprocessing import FEATURE_COLUMNS


BACKGROUND = "#050816"
PANEL = "#0A1028"
GRID = "#1B2A4A"
TEXT = "#E8F7FF"
MUTED = "#8EA6C9"
CYAN = "#00E5FF"
BLUE = "#247BFF"
VIOLET = "#9D5CFF"


def close_figure(figure: Figure) -> None:
    """Release a Matplotlib figure after Streamlit has rendered it."""
    plt.close(figure)


def activation_sample_count(activations: Sequence[np.ndarray]) -> int:
    """Return the shared batch size represented by cached layer activations."""
    if not activations:
        return 0
    counts = {
        1 if np.asarray(values).ndim == 1 else int(np.asarray(values).shape[0])
        for values in activations
    }
    if len(counts) != 1:
        raise ValueError("Activation layers do not share a batch dimension")
    return counts.pop()


def _activation_sample(values: np.ndarray, units: int, sample_index: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if sample_index != 0:
            raise IndexError("One-dimensional activation cache has only sample zero")
        selected = array
    elif array.ndim == 2:
        if not 0 <= sample_index < array.shape[0]:
            raise IndexError("Activation sample index is outside the cached batch")
        selected = array[sample_index]
    else:
        raise ValueError("Neuron activations must have one or two dimensions")
    if selected.size != units:
        raise ValueError(f"Expected {units} neuron activations, received {selected.size}")
    return selected.reshape(-1)


def _node_positions(layer_sizes: tuple[int, ...]) -> list[list[tuple[float, float]]]:
    positions: list[list[tuple[float, float]]] = []
    maximum = max(layer_sizes)
    for layer_index, size in enumerate(layer_sizes):
        offset = (maximum - size) / 2.0
        positions.append(
            [(float(layer_index), float(maximum - (offset + index))) for index in range(size)]
        )
    return positions


def _activation_color(value: float, scale: float) -> str:
    normalized = float(np.clip(value / max(scale, 1e-12), -1.0, 1.0))
    if normalized >= 0.0:
        return colors.to_hex(
            (0.04, 0.09 + 0.55 * normalized, 0.18 + 0.75 * normalized)
        )
    magnitude = abs(normalized)
    return colors.to_hex((0.12 + 0.48 * magnitude, 0.05, 0.25 + 0.65 * magnitude))


def network_figure(
    model: MLP,
    activations: Sequence[np.ndarray] | None = None,
    sample_index: int = 0,
) -> Figure:
    """Render every real neuron and weighted connection in the loaded MLP."""
    figure, axis = plt.subplots(figsize=(12, 6.5), facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    positions = _node_positions(model.layer_sizes)
    maximum_weight = max(float(np.max(np.abs(weight))) for weight in model.weights)

    for layer_index, weights in enumerate(model.weights):
        for source_index, source in enumerate(positions[layer_index]):
            for target_index, target in enumerate(positions[layer_index + 1]):
                weight = float(weights[source_index, target_index])
                strength = abs(weight) / max(maximum_weight, 1e-12)
                axis.plot(
                    (source[0], target[0]),
                    (source[1], target[1]),
                    color=CYAN if weight >= 0.0 else VIOLET,
                    alpha=0.08 + 0.62 * strength,
                    linewidth=0.35 + 2.4 * strength,
                    zorder=1,
                )

    for layer_index, layer_positions in enumerate(positions):
        values: np.ndarray | None = None
        if activations is not None and layer_index < len(activations):
            values = _activation_sample(
                np.asarray(activations[layer_index]),
                model.layer_sizes[layer_index],
                sample_index,
            )
        scale = float(np.max(np.abs(values))) if values is not None and values.size else 1.0
        for neuron_index, (x_position, y_position) in enumerate(layer_positions):
            face = PANEL if values is None else _activation_color(float(values[neuron_index]), scale)
            axis.scatter(
                x_position,
                y_position,
                s=560 if layer_index == 0 else 440,
                color=face,
                edgecolor=CYAN if layer_index in (0, len(positions) - 1) else BLUE,
                linewidth=1.6,
                zorder=3,
            )
            if layer_index == 0:
                label = FEATURE_COLUMNS[neuron_index].replace("koi_", "").replace("model_", "")
            elif layer_index == len(positions) - 1:
                label = "SCORE"
            else:
                label = f"H{layer_index}.{neuron_index + 1}"
            axis.text(
                x_position,
                y_position,
                label,
                color=TEXT,
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

    names = ["INPUT", *[f"HIDDEN {i}" for i in range(1, len(positions) - 1)], "OUTPUT"]
    maximum = max(model.layer_sizes)
    for index, name in enumerate(names):
        axis.text(index, maximum + 0.65, name, color=MUTED, ha="center", fontsize=10, weight="bold")
    axis.text(
        0.0,
        0.02,
        "cyan = positive weight  •  violet = negative weight  •  line width = |weight|",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=9,
    )
    axis.set_xlim(-0.45, len(positions) - 0.55)
    axis.set_ylim(0.0, maximum + 1.05)
    axis.axis("off")
    figure.tight_layout()
    return figure


def training_history_figure(history: list[dict[str, int | float]]) -> Figure:
    """Plot the real train and validation observations stored by Trainer."""
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=BACKGROUND)
    for axis in axes:
        axis.set_facecolor(PANEL)
        axis.tick_params(colors=MUTED)
        axis.grid(color=GRID, alpha=0.55, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_color(GRID)
    axes[0].plot(frame["epoch"], frame["train_loss"], color=CYAN, label="Training")
    axes[0].plot(frame["epoch"], frame["validation_loss"], color=VIOLET, label="Validation")
    axes[0].set_title("Binary Cross-Entropy", color=TEXT)
    axes[0].set_ylabel("Loss", color=MUTED)
    axes[1].plot(frame["epoch"], frame["train_accuracy"], color=CYAN, label="Training")
    axes[1].plot(frame["epoch"], frame["validation_accuracy"], color=VIOLET, label="Validation")
    axes[1].set_title("Classification Accuracy", color=TEXT)
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xlabel("Epoch", color=MUTED)
        legend = axis.legend(facecolor=PANEL, edgecolor=GRID)
        for label in legend.get_texts():
            label.set_color(TEXT)
    figure.tight_layout()
    return figure


def parameter_frame(model: MLP, layer_index: int) -> pd.DataFrame:
    """Expose one exact weight matrix with source/target neuron labels."""
    weights = model.weights[layer_index]
    source = [f"L{layer_index}.N{index + 1}" for index in range(weights.shape[0])]
    target = [f"L{layer_index + 1}.N{index + 1}" for index in range(weights.shape[1])]
    return pd.DataFrame(weights, index=source, columns=target)


def activation_frame(
    model: MLP,
    activations: Sequence[np.ndarray] | None = None,
    sample_index: int = 0,
) -> pd.DataFrame:
    """Return individual real neuron activations from the latest forward pass."""
    values = activations if activations is not None else model.activations
    records: list[dict[str, int | float | str]] = []
    for layer_index, layer_values in enumerate(values):
        selected = _activation_sample(
            np.asarray(layer_values),
            model.layer_sizes[layer_index],
            sample_index,
        )
        for neuron_index, value in enumerate(selected):
            records.append(
                {
                    "sample_index": sample_index,
                    "layer": model.architecture[layer_index]["name"],
                    "neuron": neuron_index + 1,
                    "activation": float(value),
                }
            )
    return pd.DataFrame(records)
