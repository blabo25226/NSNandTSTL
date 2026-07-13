"""Utilities for reproducibility and complex tensors."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_complex(
    x: torch.Tensor,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Promote real tensor to complex with zero imaginary part."""
    if x.is_complex():
        return x.to(dtype)
    return torch.complex(x.to(dtype=torch.float32), torch.zeros_like(x, dtype=torch.float32)).to(dtype)


def real_out(z: torch.Tensor) -> torch.Tensor:
    """Return real part; pass through if already real."""
    return z.real if z.is_complex() else z
