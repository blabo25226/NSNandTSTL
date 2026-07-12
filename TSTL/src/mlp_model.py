"""MLP regression model for TSTL layer-selective benchmarks."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionMLP(nn.Module):
    """
    MLP for synthetic regression.

    TSTL 'layers' are hidden Linear blocks only (indices 0 .. num_hidden-1).
    input_fc and output_fc correspond to embedding / LM head (always trainable).
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_fc(x))
        for layer in self.hidden:
            h = self.act(layer(h))
        return self.output_fc(h).squeeze(-1)

    def num_hidden(self) -> int:
        return len(self.hidden)
