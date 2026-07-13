"""Leaf weight sampling: ordinary softmax vs Gumbel-softmax."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F


class LeafSoftmaxMode(str, Enum):
    SOFTMAX = "softmax"
    GUMBEL = "gumbel"

    @classmethod
    def parse(cls, value: str) -> "LeafSoftmaxMode":
        v = value.lower().strip()
        if v in ("softmax", "ordinary", "plain"):
            return cls.SOFTMAX
        if v in ("gumbel", "gumbel-softmax", "gumbel_softmax"):
            return cls.GUMBEL
        raise ValueError(f"unknown leaf softmax mode: {value}")


def sample_gumbel(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Gumbel(0, 1) noise."""
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    u = u.clamp(1e-20, 1.0 - 1e-20)
    return -torch.log(-torch.log(u))


def leaf_weights(
    logits: torch.Tensor,
    temperature: float,
    mode: LeafSoftmaxMode,
    *,
    training: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Compute leaf softmax weights.

    Gumbel-softmax is used only during training; eval always uses plain softmax.
    """
    temp = max(temperature, 1e-6)
    if mode == LeafSoftmaxMode.GUMBEL and training:
        gumbel = sample_gumbel(
            logits.shape, device=logits.device, dtype=logits.dtype, generator=generator
        )
        return F.softmax((logits + gumbel) / temp, dim=-1)
    return F.softmax(logits / temp, dim=-1)
