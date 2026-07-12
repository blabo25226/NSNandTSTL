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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
