"""Binary EML tree head with learnable leaf affine combinations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eml import EMLNode
from utils import as_complex, real_out


class EMLTreeHead(nn.Module):
    """
    Full binary EML tree of depth D.

    Depth D => 2^D leaves, 2^D - 1 internal EML nodes.
    Each leaf: w_a * alpha + w_b * (beta^T z) + w_g * f_prev  (f_prev=0 in v1).
    """

    def __init__(
        self,
        feature_dim: int,
        depth: int = 2,
        temperature: float = 1.0,
        f_prev: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1 or depth > 4:
            raise ValueError("depth must be in [1, 4]")

        self.feature_dim = feature_dim
        self.depth = depth
        self.temperature = temperature
        self.f_prev_const = f_prev

        self.num_leaves = 2**depth
        self.num_internal = self.num_leaves - 1

        self.alpha = nn.Parameter(torch.zeros(self.num_leaves))
        self.beta = nn.Parameter(torch.randn(self.num_leaves, feature_dim) * 0.01)
        self.leaf_logits = nn.Parameter(torch.zeros(self.num_leaves, 3))

        self.eml_node = EMLNode()

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(temperature, 1e-6)

    def set_leaf_constant(self, leaf_index: int, value: float) -> None:
        """Pin leaf to constant value (alpha term only). For tests / symbolic setup."""
        with torch.no_grad():
            self.alpha[leaf_index] = value
            self.beta[leaf_index].zero_()
            self.leaf_logits[leaf_index] = torch.tensor([10.0, -10.0, -10.0])

    def set_leaf_linear(self, leaf_index: int, coeffs: list[float] | torch.Tensor) -> None:
        """Pin leaf to beta^T z (linear combination of features)."""
        with torch.no_grad():
            coeffs_t = torch.as_tensor(coeffs, dtype=self.beta.dtype, device=self.beta.device)
            if coeffs_t.shape != self.beta[leaf_index].shape:
                raise ValueError("coeffs length must match feature_dim")
            self.alpha[leaf_index] = 0.0
            self.beta[leaf_index].copy_(coeffs_t)
            self.leaf_logits[leaf_index] = torch.tensor([-10.0, 10.0, -10.0])

    def evaluate_eml_tree(self, leaf_values: list[torch.Tensor]) -> torch.Tensor:
        """Evaluate perfect binary tree from explicit leaf values (reference helper)."""
        if len(leaf_values) != self.num_leaves:
            raise ValueError(f"expected {self.num_leaves} leaves, got {len(leaf_values)}")
        level = leaf_values
        while len(level) > 1:
            next_level: list[torch.Tensor] = []
            for i in range(0, len(level), 2):
                next_level.append(self.eml_node(level[i], level[i + 1]))
            level = next_level
        return real_out(level[0])

    def leaf_values(self, z: torch.Tensor) -> torch.Tensor:
        """Compute all leaf values; z shape (batch, feature_dim) or (feature_dim,)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        batch = z.shape[0]

        weights = F.softmax(self.leaf_logits / self.temperature, dim=-1)
        w_alpha = weights[..., 0]
        w_beta = weights[..., 1]
        w_gamma = weights[..., 2]

        linear = torch.matmul(z, self.beta.T)  # (batch, num_leaves)
        f_prev = torch.full(
            (batch, self.num_leaves),
            self.f_prev_const,
            device=z.device,
            dtype=z.dtype,
        )

        leaves = (
            w_alpha * self.alpha.unsqueeze(0)
            + w_beta * linear
            + w_gamma * f_prev
        )
        return as_complex(leaves)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        leaves = self.leaf_values(z)
        squeeze = leaves.dim() == 1 or (leaves.dim() == 2 and leaves.shape[0] == 1)

        if leaves.dim() == 1:
            leaves = leaves.unsqueeze(0)

        level = [leaves[:, i] for i in range(self.num_leaves)]

        while len(level) > 1:
            next_level: list[torch.Tensor] = []
            for i in range(0, len(level), 2):
                next_level.append(self.eml_node(level[i], level[i + 1]))
            level = next_level

        out = real_out(level[0])
        if squeeze and out.shape[0] == 1:
            return out.squeeze(0)
        return out

    def parameter_count(self) -> dict[str, int]:
        return {
            "leaves": self.num_leaves,
            "internal_nodes": self.num_internal,
            "leaf_params": self.num_leaves * (1 + self.feature_dim + 3),
        }
