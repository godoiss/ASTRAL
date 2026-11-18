"""Tests for ANN-01 activation and loss mathematics."""

import numpy as np
import pytest

from src.core.activations import relu, relu_derivative, sigmoid
from src.core.losses import binary_cross_entropy


def test_relu_and_explicit_derivative() -> None:
    """Verify ReLU and its selected subgradient at zero."""
    values = np.array([[-2.0, 0.0, 3.0]])

    np.testing.assert_array_equal(relu(values), [[0.0, 0.0, 3.0]])
    np.testing.assert_array_equal(relu_derivative(values), [[0.0, 0.0, 1.0]])


def test_sigmoid_is_stable_for_extreme_inputs() -> None:
    """Ensure extreme logits do not overflow or produce non-finite values."""
    result = sigmoid(np.array([[-1000.0, 0.0, 1000.0]]))

    assert np.all(np.isfinite(result))
    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, 1] == pytest.approx(0.5)
    assert result[0, 2] == pytest.approx(1.0)


def test_binary_cross_entropy_matches_known_value() -> None:
    """Verify the direct binary cross-entropy equation."""
    targets = np.array([[1.0], [0.0]])
    probabilities = np.array([[0.8], [0.25]])
    expected = -(np.log(0.8) + np.log(0.75)) / 2.0

    assert binary_cross_entropy(targets, probabilities) == pytest.approx(expected)

