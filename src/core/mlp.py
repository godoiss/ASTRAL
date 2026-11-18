"""ANN-01 multilayer perceptron implemented from first principles in NumPy."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.activations import relu, relu_derivative, sigmoid


FloatArray = NDArray[np.float64]
HistoryRecord = dict[str, int | float]


class MLP:
    """A dense binary classifier with explicit forward and backward passes.

    Weight matrices use the convention ``(input_units, output_units)``. Hidden
    layers use ReLU and the output layer uses sigmoid. Hidden weights use He
    initialization; output weights use Xavier-style scaling.
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (6, 8, 4, 1),
        seed: int = 42,
        engine_name: str = "ANN-01",
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("At least an input and output layer are required")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("Every layer size must be positive")
        if layer_sizes[-1] != 1:
            raise ValueError("ANN-01 binary classification requires one output unit")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.seed = int(seed)
        self.engine_name = engine_name
        self.weights: list[FloatArray] = []
        self.biases: list[FloatArray] = []
        self.gradients: dict[str, list[FloatArray]] = {
            "weights": [],
            "biases": [],
        }
        self.activations: list[FloatArray] = []
        self.pre_activations: list[FloatArray] = []
        self.training_history: list[HistoryRecord] = []
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters reproducibly from the configured seed."""
        generator = np.random.default_rng(self.seed)
        final_connection_index = len(self.layer_sizes) - 2
        for index, (input_units, output_units) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True)
        ):
            if index == final_connection_index:
                scale = np.sqrt(1.0 / input_units)
            else:
                scale = np.sqrt(2.0 / input_units)
            self.weights.append(
                generator.normal(0.0, scale, size=(input_units, output_units)).astype(
                    np.float64
                )
            )
            self.biases.append(np.zeros((1, output_units), dtype=np.float64))

    @property
    def parameter_count(self) -> int:
        """Return the exact number of trainable scalar parameters."""
        return sum(array.size for array in [*self.weights, *self.biases])

    @property
    def architecture(self) -> list[dict[str, Any]]:
        """Return a JSON-ready description of every neural layer."""
        layers: list[dict[str, Any]] = [
            {
                "index": 0,
                "name": "input",
                "units": self.layer_sizes[0],
                "activation": "identity",
            }
        ]
        last_index = len(self.layer_sizes) - 1
        for index, units in enumerate(self.layer_sizes[1:], start=1):
            layers.append(
                {
                    "index": index,
                    "name": "output" if index == last_index else f"hidden_{index}",
                    "units": units,
                    "activation": "sigmoid" if index == last_index else "relu",
                }
            )
        return layers

    def forward(self, inputs: FloatArray, retain_cache: bool = True) -> FloatArray:
        """Run forward propagation and optionally retain neuron-level state."""
        current = np.asarray(inputs, dtype=np.float64)
        if current.ndim == 1:
            current = current.reshape(1, -1)
        if current.ndim != 2 or current.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Expected input shape (samples, {self.layer_sizes[0]}), "
                f"received {current.shape}"
            )
        if not np.all(np.isfinite(current)):
            raise ValueError("Network inputs must contain only finite values")

        cached_activations = [current]
        cached_pre_activations: list[FloatArray] = []
        for index, (weights, biases) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            linear_output = current @ weights + biases
            cached_pre_activations.append(linear_output)
            if index == len(self.weights) - 1:
                current = sigmoid(linear_output)
            else:
                current = relu(linear_output)
            cached_activations.append(current)

        if retain_cache:
            self.activations = cached_activations
            self.pre_activations = cached_pre_activations
        return current

    def backward(self, targets: FloatArray) -> dict[str, list[FloatArray]]:
        """Compute all gradients using explicit reverse-mode differentiation.

        For sigmoid followed by binary cross-entropy, the output derivative
        simplifies exactly to ``(prediction - target) / batch_size``.
        """
        if not self.activations or not self.pre_activations:
            raise RuntimeError("forward(..., retain_cache=True) must run first")

        expected_shape = self.activations[-1].shape
        targets = np.asarray(targets, dtype=np.float64).reshape(expected_shape)
        if not np.all((targets == 0.0) | (targets == 1.0)):
            raise ValueError("Targets must contain only binary values")

        batch_size = targets.shape[0]
        weight_gradients: list[FloatArray] = [
            np.empty_like(weights) for weights in self.weights
        ]
        bias_gradients: list[FloatArray] = [
            np.empty_like(biases) for biases in self.biases
        ]

        delta = (self.activations[-1] - targets) / batch_size
        for layer_index in range(len(self.weights) - 1, -1, -1):
            previous_activation = self.activations[layer_index]
            weight_gradients[layer_index] = previous_activation.T @ delta
            bias_gradients[layer_index] = np.sum(delta, axis=0, keepdims=True)

            if layer_index > 0:
                upstream = delta @ self.weights[layer_index].T
                delta = upstream * relu_derivative(
                    self.pre_activations[layer_index - 1]
                )

        self.gradients = {
            "weights": weight_gradients,
            "biases": bias_gradients,
        }
        return self.gradients

    def apply_gradients(self, learning_rate: float) -> None:
        """Apply one full-batch or mini-batch gradient-descent update."""
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be positive")
        if len(self.gradients["weights"]) != len(self.weights):
            raise RuntimeError("backward() must compute gradients before an update")

        for index in range(len(self.weights)):
            self.weights[index] -= learning_rate * self.gradients["weights"][index]
            self.biases[index] -= learning_rate * self.gradients["biases"][index]

    def predict_proba(self, inputs: FloatArray) -> FloatArray:
        """Return a probability for every input record."""
        return self.forward(inputs, retain_cache=False)

    def predict(self, inputs: FloatArray, threshold: float = 0.5) -> NDArray[np.int64]:
        """Return individual binary predictions at the selected threshold."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("Classification threshold must be between zero and one")
        return (self.predict_proba(inputs) >= threshold).astype(np.int64)

    def state_dict(self) -> dict[str, FloatArray]:
        """Return copies of trainable parameters keyed by layer."""
        state: dict[str, FloatArray] = {}
        for index, (weights, biases) in enumerate(
            zip(self.weights, self.biases, strict=True), start=1
        ):
            state[f"W{index}"] = weights.copy()
            state[f"b{index}"] = biases.copy()
        return state

    def load_state_dict(self, state: dict[str, FloatArray]) -> None:
        """Validate and load trainable parameters into this architecture."""
        for index, (weights, biases) in enumerate(
            zip(self.weights, self.biases, strict=True), start=1
        ):
            weight_key = f"W{index}"
            bias_key = f"b{index}"
            if weight_key not in state or bias_key not in state:
                raise ValueError(f"State is missing {weight_key} or {bias_key}")
            loaded_weights = np.asarray(state[weight_key], dtype=np.float64)
            loaded_biases = np.asarray(state[bias_key], dtype=np.float64)
            if loaded_weights.shape != weights.shape or loaded_biases.shape != biases.shape:
                raise ValueError(f"Parameter shape mismatch in layer {index}")
            self.weights[index - 1] = loaded_weights.copy()
            self.biases[index - 1] = loaded_biases.copy()

    def save(self, path: Path) -> Path:
        """Persist architecture and parameters in a compressed NumPy archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{path.stem}_",
            suffix=".npz.tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        payload: dict[str, FloatArray] = {
            "layer_sizes": np.asarray(self.layer_sizes, dtype=np.int64),
            **self.state_dict(),
        }
        try:
            with temporary_path.open("wb") as output_file:
                np.savez_compressed(output_file, **payload)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        seed: int = 42,
        engine_name: str = "ANN-01",
    ) -> MLP:
        """Restore a network from ``save`` output without pickle data."""
        with np.load(path, allow_pickle=False) as archive:
            layer_sizes = tuple(int(value) for value in archive["layer_sizes"])
            model = cls(layer_sizes=layer_sizes, seed=seed, engine_name=engine_name)
            state = {
                key: archive[key]
                for key in archive.files
                if key != "layer_sizes"
            }
        model.load_state_dict(state)
        return model

    def observability_snapshot(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return real parameters, neural state, gradients, and history.

        Activations represent the most recent retained forward pass. Gradients
        represent the most recent backward pass. Empty sections mean the
        corresponding operation has not yet occurred; values are never mocked.
        """
        parameters: dict[str, Any] = {}
        gradient_payload: dict[str, Any] = {}
        for index, (weights, biases) in enumerate(
            zip(self.weights, self.biases, strict=True), start=1
        ):
            parameters[f"layer_{index}"] = {
                "weights": weights.tolist(),
                "biases": biases.reshape(-1).tolist(),
                "weight_shape": list(weights.shape),
                "bias_shape": list(biases.shape),
            }
            if self.gradients["weights"]:
                gradient_payload[f"layer_{index}"] = {
                    "weight_gradients": self.gradients["weights"][index - 1].tolist(),
                    "bias_gradients": self.gradients["biases"][index - 1]
                    .reshape(-1)
                    .tolist(),
                }

        activation_payload = {
            self.architecture[index]["name"]: values.tolist()
            for index, values in enumerate(self.activations)
        }
        pre_activation_payload = {
            self.architecture[index + 1]["name"]: values.tolist()
            for index, values in enumerate(self.pre_activations)
        }

        return {
            "schema_version": 1,
            "engine_name": self.engine_name,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "seed": self.seed,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "parameters": parameters,
            "activations": activation_payload,
            "pre_activations": pre_activation_payload,
            "gradients": gradient_payload,
            "training_history": self.training_history,
            "metadata": metadata or {},
        }

    def export_snapshot(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically export the real observable network state as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.observability_snapshot(metadata=metadata)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{path.stem}_",
            suffix=".json.tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            temporary_path.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

