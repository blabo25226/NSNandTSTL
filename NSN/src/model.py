"""Hybrid DNN-EML model."""

from __future__ import annotations

import torch
import torch.nn as nn

from eml_tree import EMLTreeHead
from leaf_softmax import LeafSoftmaxMode
from trunk import MLPTrunk


class DNNEML(nn.Module):
    """Stage 1 MLP trunk + Stage 2 EML tree head."""

    def __init__(self, trunk: MLPTrunk, head: EMLTreeHead, concat_input: bool = False) -> None:
        super().__init__()
        self.concat_input = concat_input
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        if self.concat_input:
            z = torch.cat([x, z], dim=-1)
        return self.head(z)

    @classmethod
    def build(
        cls,
        input_dim: int,
        feature_dim: int = 4,
        head_depth: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "relu",
        concat_input: bool = False,
        leaf_softmax_mode: LeafSoftmaxMode | str = LeafSoftmaxMode.SOFTMAX,
        f_prev_mode: str = "zero",
        f_prev_passes: int = 1,
    ) -> "DNNEML":
        mode = (
            leaf_softmax_mode
            if isinstance(leaf_softmax_mode, LeafSoftmaxMode)
            else LeafSoftmaxMode.parse(leaf_softmax_mode)
        )
        if concat_input:
            trunk_out = max(2, feature_dim - input_dim)
            head_dim = input_dim + trunk_out
            trunk = MLPTrunk(
                input_dim=input_dim,
                feature_dim=trunk_out,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
            )
            head = EMLTreeHead(
                feature_dim=head_dim, depth=head_depth,
                leaf_softmax_mode=mode, f_prev_mode=f_prev_mode,
                f_prev_passes=f_prev_passes,
            )
            return cls(trunk, head, concat_input=True)
        trunk = MLPTrunk(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )
        head = EMLTreeHead(
            feature_dim=feature_dim, depth=head_depth,
            leaf_softmax_mode=mode, f_prev_mode=f_prev_mode,
            f_prev_passes=f_prev_passes,
        )
        return cls(trunk, head, concat_input=False)

    def mse_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        return nn.functional.mse_loss(pred, y)
