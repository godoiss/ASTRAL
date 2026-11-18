"""Finite-difference validation for ANN-01 analytical gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.losses import binary_cross_entropy
from src.core.mlp import MLP


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GradientCheckResult:
    """Summary of analytical versus central-difference gradients."""

    relative_error: float
    maximum_absolute_error: float
    checked_parameters: int


def check_gradients(
    model: MLP,
    inputs: FloatArray,
    targets: FloatArray,
    epsilon: float = 1e-5,
) -> GradientCheckResult:
    """Compare every backpropagation gradient with a numerical derivative."""
    if epsilon <= 0.0:
        raise ValueError("Finite-difference epsilon must be positive")

    model.forward(inputs, retain_cache=True)
    model.backward(targets)
    analytical_arrays = [
        *(gradient.copy() for gradient in model.gradients["weights"]),
        *(gradient.copy() for gradient in model.gradients["biases"]),
    ]
    parameter_arrays = [*model.weights, *model.biases]
    numerical_arrays: list[FloatArray] = []

    for parameters in parameter_arrays:
        numerical = np.zeros_like(parameters)
        for index in np.ndindex(parameters.shape):
            original_value = parameters[index]
            parameters[index] = original_value + epsilon
            positive_loss = binary_cross_entropy(
                targets,
                model.forward(inputs, retain_cache=False),
            )
            parameters[index] = original_value - epsilon
            negative_loss = binary_cross_entropy(
                targets,
                model.forward(inputs, retain_cache=False),
            )
            parameters[index] = original_value
            numerical[index] = (positive_loss - negative_loss) / (2.0 * epsilon)
        numerical_arrays.append(numerical)

    analytical_vector = np.concatenate(
        [gradient.reshape(-1) for gradient in analytical_arrays]
    )
    numerical_vector = np.concatenate(
        [gradient.reshape(-1) for gradient in numerical_arrays]
    )
    denominator = (
        np.linalg.norm(analytical_vector) + np.linalg.norm(numerical_vector) + 1e-12
    )
    relative_error = float(
        np.linalg.norm(analytical_vector - numerical_vector) / denominator
    )
    maximum_absolute_error = float(
        np.max(np.abs(analytical_vector - numerical_vector))
    )

    model.forward(inputs, retain_cache=True)
    model.backward(targets)
    return GradientCheckResult(
        relative_error=relative_error,
        maximum_absolute_error=maximum_absolute_error,
        checked_parameters=analytical_vector.size,
    )

