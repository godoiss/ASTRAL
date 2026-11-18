"""Runtime and accelerator inspection for local ANN-02 execution."""

from __future__ import annotations

import platform
import sys
from typing import Any

import torch


def hardware_report() -> dict[str, Any]:
    """Describe the installed PyTorch build and visible CUDA runtime."""
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_gb": round(properties.total_memory / 1024**3, 3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "cuda_device_count": torch.cuda.device_count() if available else 0,
        "cuda_devices": devices,
        "selected_default": "cuda" if available else "cpu",
    }


def select_device(preference: str = "auto") -> torch.device:
    """Select CUDA when requested and available, otherwise use CPU safely."""
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch runtime cannot access it")
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")
