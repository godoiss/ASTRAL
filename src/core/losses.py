"""Loss functions for the ASTRAL neural engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def binary_cross_entropy(
    targets: FloatArray,
    probabilities: FloatArray,
    epsilon: float = 1e-12,
) -> float:
    """Return mean binary cross-entropy for probabilities and binary targets.

    ``epsilon`` clipping affects only the logarithm evaluation and prevents
    ``log(0)``; it does not mutate the supplied probabilities.
    """
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape:
        raise ValueError(
            "Targets and probabilities must have identical shapes, "
            f"received {targets.shape} and {probabilities.shape}"
        )
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("Binary cross-entropy targets must contain only 0 or 1")

    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    losses = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    return float(np.mean(losses))

