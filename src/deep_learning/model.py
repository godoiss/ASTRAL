"""Dual-branch one-dimensional convolutional network for Kepler signals."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn

from src.deep_learning.config import ANN02Config


class SignalBranch(nn.Module):
    """Extract compact features from one fixed-length light-curve view."""

    def __init__(self, config: ANN02Config, pool_size: int) -> None:
        super().__init__()
        first, second = config.branch_channels
        padding = config.kernel_size // 2
        self.layers = nn.ModuleDict(
            OrderedDict(
                [
                    ("conv_1", nn.Conv1d(config.input_channels, first, config.kernel_size, padding=padding)),
                    ("relu_1", nn.ReLU()),
                    ("pool_1", nn.MaxPool1d(pool_size)),
                    ("conv_2", nn.Conv1d(first, second, config.kernel_size, padding=padding)),
                    ("relu_2", nn.ReLU()),
                    ("adaptive_pool", nn.AdaptiveAvgPool1d(1)),
                ]
            )
        )

    def forward(self, values: Tensor, prefix: str, capture: dict[str, Tensor] | None) -> Tensor:
        output = values
        for name, layer in self.layers.items():
            output = layer(output)
            if capture is not None:
                if output.requires_grad:
                    output.retain_grad()
                capture[f"{prefix}.{name}"] = output
        return output.flatten(start_dim=1)


class DualBranchCNN(nn.Module):
    """ANN-02 with independent global/local signal branches and late fusion."""

    def __init__(self, config: ANN02Config | None = None) -> None:
        super().__init__()
        self.config = config or ANN02Config()
        self.global_branch = SignalBranch(self.config, self.config.global_pool_size)
        self.local_branch = SignalBranch(self.config, self.config.local_pool_size)
        branch_features = self.config.branch_channels[-1] * 2
        self.fusion = nn.Linear(branch_features, self.config.fused_features)
        self.fusion_activation = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.fused_features, 1)
        self.last_activations: dict[str, Tensor] = {}

    def forward(
        self,
        global_view: Tensor,
        local_view: Tensor,
        retain_observability: bool = False,
    ) -> Tensor:
        """Return binary logits; optionally retain real intermediate tensors."""
        capture: dict[str, Tensor] | None = {} if retain_observability else None
        global_features = self.global_branch(global_view, "global", capture)
        local_features = self.local_branch(local_view, "local", capture)
        fused = torch.cat((global_features, local_features), dim=1)
        if capture is not None:
            if fused.requires_grad:
                fused.retain_grad()
            capture["fusion.concat"] = fused
        hidden = self.fusion_activation(self.fusion(fused))
        if capture is not None:
            if hidden.requires_grad:
                hidden.retain_grad()
            capture["fusion.hidden"] = hidden
        logits = self.classifier(self.dropout(hidden)).squeeze(1)
        if capture is not None:
            if logits.requires_grad:
                logits.retain_grad()
            capture["output.logits"] = logits
            self.last_activations = capture
        return logits

    @property
    def trainable_parameter_count(self) -> int:
        """Return the exact number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def architecture(self) -> dict[str, Any]:
        """Return a JSON-ready structural description from the real module tree."""
        return {
            "name": "ANN-02 Signal Intelligence",
            "input_views": {"global": [2, 2001], "local": [2, 201]},
            "channels": ["median-centered flux", "availability mask"],
            "modules": [
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                }
                for name, module in self.named_modules()
                if name
            ],
            "trainable_parameters": self.trainable_parameter_count,
            "configuration": self.config.to_dict(),
        }
