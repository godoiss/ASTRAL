"""Activation functions implemented directly with NumPy.

No automatic-differentiation or neural-network framework is used here. The
derivatives are explicit because ANN-01 performs its own backpropagation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def relu(values: FloatArray) -> FloatArray:
    """Return ``max(0, x)`` element by element."""
    return np.maximum(0.0, values)


def relu_derivative(pre_activations: FloatArray) -> FloatArray:
    """Return the ReLU derivative evaluated at the pre-activation values.

    The derivative at exactly zero is defined as zero, a conventional and
    deterministic subgradient choice.
    """
    return (pre_activations > 0.0).astype(np.float64)


def sigmoid(values: FloatArray) -> FloatArray:
    """Compute a numerically stable logistic sigmoid.

    Positive and negative inputs use separate algebraic forms so that neither
    branch evaluates an overflowing exponential.
    """
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    non_negative = values >= 0.0
    result[non_negative] = 1.0 / (1.0 + np.exp(-values[non_negative]))
    exp_values = np.exp(values[~non_negative])
    result[~non_negative] = exp_values / (1.0 + exp_values)
    return result

