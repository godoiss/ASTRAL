"""Automated mathematical and leakage audit for Stage 04."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core.gradient_check import check_gradients
from src.core.mlp import MLP
from src.data.preprocessing import PreparedDataset


def audit_engine_and_dataset(
    dataset: PreparedDataset,
    dataset_identity: dict[str, Any],
    split_seed: int,
) -> dict[str, Any]:
    """Audit ANN-01 mathematics, parameter integrity, and grouped partitions."""
    if dataset.train.targets is None:
        raise ValueError("Training targets are required for the audit")
    model = MLP(layer_sizes=(6, 8, 4, 1), seed=23)
    sample_features = dataset.train.features[:5]
    sample_targets = dataset.train.targets[:5]
    outputs = model.forward(sample_features, retain_cache=True)
    gradients = model.backward(sample_targets)
    gradient_result = check_gradients(
        model,
        sample_features,
        sample_targets,
        epsilon=1e-5,
    )

    expected_weight_shapes = [(6, 8), (8, 4), (4, 1)]
    expected_bias_shapes = [(1, 8), (1, 4), (1, 1)]
    parameter_integrity = {
        "parameter_count": model.parameter_count,
        "weight_shapes": [list(array.shape) for array in model.weights],
        "bias_shapes": [list(array.shape) for array in model.biases],
        "finite_weights": bool(all(np.all(np.isfinite(item)) for item in model.weights)),
        "finite_biases": bool(all(np.all(np.isfinite(item)) for item in model.biases)),
        "expected_shapes_match": bool(
            [item.shape for item in model.weights] == expected_weight_shapes
            and [item.shape for item in model.biases] == expected_bias_shapes
        ),
    }
    gradient_integrity = {
        "finite": bool(
            all(
                np.all(np.isfinite(item))
                for item in [*gradients["weights"], *gradients["biases"]]
            )
        ),
        "weight_shapes": [list(array.shape) for array in gradients["weights"]],
        "bias_shapes": [list(array.shape) for array in gradients["biases"]],
        "relative_error": gradient_result.relative_error,
        "maximum_absolute_error": gradient_result.maximum_absolute_error,
        "checked_parameters": gradient_result.checked_parameters,
    }

    split_groups = {
        "train": set(dataset.train.group_ids.tolist()),
        "validation": set(dataset.validation.group_ids.tolist()),
        "test": set(dataset.test.group_ids.tolist()),
    }
    overlaps = {
        "train_validation": len(split_groups["train"] & split_groups["validation"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "validation_test": len(split_groups["validation"] & split_groups["test"]),
    }
    return {
        "dataset": dataset_identity,
        "split_seed": split_seed,
        "forward_output_shape": list(outputs.shape),
        "forward_outputs_finite": bool(np.all(np.isfinite(outputs))),
        "forward_outputs_in_unit_interval": bool(
            np.all((outputs >= 0.0) & (outputs <= 1.0))
        ),
        "parameter_integrity": parameter_integrity,
        "gradient_integrity": gradient_integrity,
        "group_overlap_counts": overlaps,
        "preprocessor_fitted_rows": dataset.preprocessor.fitted_on_rows,
        "split_records": {
            "train": dataset.train.size,
            "validation": dataset.validation.size,
            "test": dataset.test.size,
            "candidates": dataset.candidates.size,
        },
        "prior_test_usage": {
            "previous_test_was_evaluated": True,
            "previous_split_seed": 42,
            "previous_test_accuracy": 0.8012313104661389,
            "limitation": (
                "The earlier Stage 03 test metric was observed and reported. "
                "Stage 04 therefore freezes a new grouped split before search, but "
                "this is not equivalent to an external independent dataset."
            ),
        },
    }

