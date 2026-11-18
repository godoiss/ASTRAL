"""Real ANN-02 parameter, activation, feature-map, and gradient snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.deep_learning.model import DualBranchCNN


def _tensor_payload(tensor: torch.Tensor | None, sample_only: bool = False) -> Any:
    if tensor is None:
        return None
    values = tensor.detach().cpu()
    if sample_only and values.ndim and values.shape[0] > 0:
        values = values[0]
    return values.numpy().tolist()


def build_snapshot(
    model: DualBranchCNN,
    object_id: str,
    device: torch.device,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Serialize the exact retained network state for one astronomical record."""
    parameters: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    filters: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        parameters[name] = {
            "shape": list(parameter.shape),
            "values": _tensor_payload(parameter),
        }
        gradients[name] = {
            "shape": list(parameter.shape),
            "values": _tensor_payload(parameter.grad),
        }
        if name.endswith("weight") and "conv_" in name:
            filters[name] = _tensor_payload(parameter)
    activations = {
        name: {
            "shape": list(values.shape),
            "sample_index": 0,
            "values": _tensor_payload(values, sample_only=True),
            "gradients": _tensor_payload(values.grad, sample_only=True),
        }
        for name, values in model.last_activations.items()
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "object_id": object_id,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "architecture": model.architecture(),
        "parameters": parameters,
        "convolutional_filters": filters,
        "activations": activations,
        "notes": [
            "Feature maps and activations belong only to sample_index 0 and are not simulated.",
            "A null gradient means the tensor was captured during inference without backward propagation.",
        ],
    }


def export_snapshot(path: Path, snapshot: dict[str, Any]) -> Path:
    """Atomically persist a standards-compliant observability JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json.tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def activation_summary(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact real activation statistics suitable for UI tables."""
    rows: list[dict[str, Any]] = []
    for name, payload in snapshot.get("activations", {}).items():
        values = np.asarray(payload["values"], dtype=np.float64)
        rows.append(
            {
                "layer": name,
                "shape": " × ".join(str(value) for value in payload["shape"]),
                "minimum": float(values.min()) if values.size else None,
                "maximum": float(values.max()) if values.size else None,
                "mean": float(values.mean()) if values.size else None,
                "standard_deviation": float(values.std()) if values.size else None,
            }
        )
    return rows
