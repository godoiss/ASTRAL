"""Unit and numerical-gradient tests for the ANN-01 MLP."""

import json

import numpy as np

from src.core.gradient_check import check_gradients
from src.core.mlp import MLP


def test_ann01_architecture_has_exactly_97_parameters() -> None:
    """Protect the intended 6-8-4-1 architecture and parameter budget."""
    model = MLP(seed=7)

    assert model.layer_sizes == (6, 8, 4, 1)
    assert model.parameter_count == 97
    assert [weights.shape for weights in model.weights] == [(6, 8), (8, 4), (4, 1)]
    assert [biases.shape for biases in model.biases] == [(1, 8), (1, 4), (1, 1)]


def test_forward_backward_update_and_observability_are_real() -> None:
    """Exercise neural state, gradients, updates, and JSON observability."""
    model = MLP(seed=11)
    inputs = np.array(
        [
            [0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
            [-0.3, 0.2, 0.7, -0.1, 0.4, -0.8],
        ]
    )
    targets = np.array([[1.0], [0.0]])
    probabilities = model.forward(inputs)
    original_weights = [weights.copy() for weights in model.weights]
    gradients = model.backward(targets)
    model.apply_gradients(learning_rate=0.01)
    snapshot = model.observability_snapshot(
        metadata={"object_ids": ["test-object-1", "test-object-2"]}
    )

    assert probabilities.shape == (2, 1)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert len(gradients["weights"]) == 3
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(original_weights, model.weights, strict=True)
    )
    assert snapshot["activations"]["input"] == inputs.tolist()
    assert snapshot["gradients"]["layer_3"]["weight_gradients"]
    json.dumps(snapshot, allow_nan=False)


def test_backpropagation_matches_all_97_numerical_gradients() -> None:
    """Validate every analytical derivative using central differences."""
    generator = np.random.default_rng(19)
    model = MLP(seed=23)
    inputs = generator.normal(size=(5, 6))
    targets = np.array([[1.0], [0.0], [1.0], [0.0], [1.0]])

    result = check_gradients(model, inputs, targets, epsilon=1e-5)

    assert result.checked_parameters == 97
    assert result.relative_error < 1e-7
    assert result.maximum_absolute_error < 1e-6


def test_model_parameter_round_trip(tmp_path) -> None:
    """Ensure persisted model parameters reproduce identical probabilities."""
    model = MLP(seed=29)
    inputs = np.arange(18, dtype=np.float64).reshape(3, 6) / 10.0
    expected = model.predict_proba(inputs)
    model_path = tmp_path / "model.npz"

    model.save(model_path)
    restored = MLP.load(model_path, seed=29)

    np.testing.assert_allclose(restored.predict_proba(inputs), expected)
    assert restored.parameter_count == 97

