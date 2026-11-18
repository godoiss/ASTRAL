"""Fixed-size, mask-aware representations for the future ANN-02."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.observatory.config import ObservatoryConfig
from src.observatory.models import LightCurveArtifact, NeuralLightCurveExample
from src.observatory.persistence import save_neural_example


def median_bin_view(
    coordinates: np.ndarray,
    flux: np.ndarray,
    lower: float,
    upper: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-bin observed samples; leave empty bins NaN with mask false."""
    edges = np.linspace(lower, upper, size + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    output = np.full(size, np.nan, dtype=np.float64)
    mask = np.zeros(size, dtype=np.bool_)
    finite = np.isfinite(coordinates) & np.isfinite(flux)
    indices = np.searchsorted(edges, coordinates[finite], side="right") - 1
    values = flux[finite]
    valid = (indices >= 0) & (indices < size)
    for bin_index in np.unique(indices[valid]):
        observed = values[valid & (indices == bin_index)]
        output[bin_index] = float(np.median(observed))
        mask[bin_index] = True
    return centers, output, mask


def prepare_neural_example(
    artifact: LightCurveArtifact,
    config: ObservatoryConfig,
    persist: bool = True,
) -> NeuralLightCurveExample:
    """Create global/local phase vectors without interpolating missing bins."""
    global_coordinates, global_flux, global_mask = median_bin_view(
        artifact.phase,
        artifact.detrended_flux,
        -0.5,
        0.5,
        config.global_view_size,
    )
    half_width = max(
        artifact.metadata.duration_hours * config.local_view_duration_multiplier,
        3.0,
    )
    local_coordinates, local_flux, local_mask = median_bin_view(
        artifact.local_phase_hours,
        artifact.local_flux,
        -half_width,
        half_width,
        config.local_view_size,
    )
    example = NeuralLightCurveExample(
        object_id=artifact.metadata.object_id,
        kic=artifact.metadata.kic,
        split=artifact.metadata.split,
        class_label=artifact.metadata.class_label,
        global_coordinates=global_coordinates,
        global_flux=global_flux,
        global_mask=global_mask,
        local_coordinates_hours=local_coordinates,
        local_flux=local_flux,
        local_mask=local_mask,
    )
    if persist:
        save_neural_example(example)
    return example


def validate_group_separation(examples: list[NeuralLightCurveExample]) -> None:
    """Reject labelled host stars assigned to multiple model partitions."""
    assignments: dict[int, set[str]] = {}
    for example in examples:
        if example.class_label < 0:
            continue
        assignments.setdefault(example.kic, set()).add(example.split)
    conflicts = {kic: splits for kic, splits in assignments.items() if len(splits) > 1}
    if conflicts:
        raise ValueError(f"Host-star split conflicts detected: {conflicts}")


def save_dataset_collection(
    examples: list[NeuralLightCurveExample],
    output_path: Path,
) -> Path:
    """Stack compatible examples while preserving IDs, stars, splits, and masks."""
    if not examples:
        raise ValueError("At least one neural example is required")
    validate_group_separation(examples)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        object_ids=np.asarray([item.object_id for item in examples]),
        kic=np.asarray([item.kic for item in examples], dtype=np.int64),
        splits=np.asarray([item.split for item in examples]),
        class_labels=np.asarray([item.class_label for item in examples], dtype=np.int64),
        global_flux=np.stack([item.global_flux for item in examples]),
        global_mask=np.stack([item.global_mask for item in examples]),
        local_flux=np.stack([item.local_flux for item in examples]),
        local_mask=np.stack([item.local_mask for item in examples]),
    )
    return output_path
