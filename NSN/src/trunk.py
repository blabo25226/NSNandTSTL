"""MLP feature trunk: R^p -> R^d."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), act])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feature_dim))
        self.net = nn.Sequential(*layers)
        # A single-Linear trunk (num_layers=1) is exactly linear: z = W x + b.
        self.is_linear = num_layers == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_layers(self) -> list[nn.Linear]:
        """All Linear submodules in order (for layer-selective training / TSTL profiling)."""
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def set_trainable_linear_layers(self, indices: set[int] | None) -> None:
        """
        Enable gradients on selected Linear layers only.

        indices=None trains every Linear layer; an empty set freezes the whole trunk.
        ReLU activations have no parameters and are ignored.
        """
        for i, lin in enumerate(self.linear_layers()):
            train = indices is None or i in indices
            for p in lin.parameters():
                p.requires_grad_(train)

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def linear_weights(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Exact (W, b) with z = W x + b when the trunk is a single Linear layer.

        Returns None for a multi-layer (nonlinear) trunk, where an exact linear
        readout does not exist (use distillation instead).
        """
        if not self.is_linear:
            return None
        lin = self.net[0]
        return lin.weight.detach().clone(), lin.bias.detach().clone()
