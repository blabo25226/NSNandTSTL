"""
Trunk interpretability: make the lower half of the two-layer explanation
(x -> z = trunk(x)) white-box.

The head already exports a closed form yhat = f(z) (see `snap.export_symbolic_
expression`). This module recovers z as a function of the raw inputs x so the two
layers compose into a single auditable expression yhat = f(z(x)):

  - `linear_readout`  : z_j ~= w_j^T x + b_j (exact for a single-Linear trunk,
                        otherwise least-squares distillation with per-component R^2).
  - `compose_symbolic`: substitute the linear z(x) into the head export string,
                        yielding yhat as one closed form in x.
  - `trunk_attribution`: rank input features per z-component (|W| for a linear
                        trunk, mean |dz_j/dx_i| via autograd otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model import DNNEML
from snap import export_symbolic_expression


@dataclass
class LinearReadout:
    W: torch.Tensor  # (feature_dim, input_dim)
    b: torch.Tensor  # (feature_dim,)
    r2: torch.Tensor  # (feature_dim,) per-component R^2 of the linear fit
    exact: bool  # True when the trunk is exactly linear (no distillation)


def _head_features(model: DNNEML, x: torch.Tensor) -> torch.Tensor:
    """z exactly as the head sees it (accounts for concat_input)."""
    z = model.trunk(x)
    if model.concat_input:
        z = torch.cat([x, z], dim=-1)
    return z


def _r2_per_component(z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    ss_res = ((z - z_hat) ** 2).sum(dim=0)
    ss_tot = ((z - z.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = torch.where(
        ss_tot > 1e-12,
        1.0 - ss_res / ss_tot,
        # Constant component: perfect iff the fit is also (near) constant.
        torch.where(ss_res < 1e-9, torch.ones_like(ss_res), torch.zeros_like(ss_res)),
    )
    return r2


def linear_readout(model: DNNEML, x: torch.Tensor) -> LinearReadout:
    """Exact linear weights for a single-Linear trunk, else least-squares distillation."""
    model.eval()
    with torch.no_grad():
        z = _head_features(model, x)

    exact_wb = model.trunk.linear_weights() if not model.concat_input else None
    if exact_wb is not None:
        W, b = exact_wb
        with torch.no_grad():
            z_hat = x @ W.T + b
        return LinearReadout(W=W, b=b, r2=_r2_per_component(z, z_hat), exact=True)

    # Least-squares distillation: fit [x | 1] @ theta ~= z.
    with torch.no_grad():
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        aug = torch.cat([x, ones], dim=1)  # (n, input_dim + 1)
        theta = torch.linalg.lstsq(aug, z).solution  # (input_dim + 1, feature_dim)
        W = theta[:-1].T.contiguous()  # (feature_dim, input_dim)
        b = theta[-1].contiguous()  # (feature_dim,)
        z_hat = aug @ theta
    return LinearReadout(W=W, b=b, r2=_r2_per_component(z, z_hat), exact=False)


def _linear_component_str(w_row: torch.Tensor, b_val: float, x_names: list[str]) -> str:
    terms = [
        f"{w_row[i].item():.6g}*{x_names[i]}"
        for i in range(w_row.shape[0])
        if abs(w_row[i].item()) > 1e-8
    ]
    if abs(b_val) > 1e-8 or not terms:
        terms.append(f"{b_val:.6g}")
    return "(" + " + ".join(terms) + ")"


def compose_symbolic(
    model: DNNEML,
    readout: LinearReadout,
    x_names: list[str] | None = None,
) -> str:
    """
    Compose the head export with the linear z(x) into a single closed form in x.

    The head should already be snapped (one-hot leaf logits) for a faithful
    result; this reuses `export_symbolic_expression` with z_names set to the
    linear-in-x expressions.
    """
    head = model.head
    if x_names is None:
        x_names = [f"x{i}" for i in range(readout.W.shape[1])]
    if readout.W.shape[0] != head.feature_dim:
        raise ValueError("readout feature_dim must match head.feature_dim")

    z_names = [
        _linear_component_str(readout.W[j], readout.b[j].item(), x_names)
        for j in range(head.feature_dim)
    ]
    return export_symbolic_expression(head, z_names)


def trunk_attribution(
    model: DNNEML,
    x: torch.Tensor,
    top_k: int = 3,
) -> list[list[tuple[int, float]]]:
    """
    Per z-component, the top_k most influential input features.

    Linear trunk: score = |W[j, i]|. Nonlinear: score = mean_batch |dz_j/dx_i|.
    Returns, for each head feature j, a list of (feature_index, score) descending.
    """
    model.eval()
    exact_wb = model.trunk.linear_weights() if not model.concat_input else None
    if exact_wb is not None:
        W, _ = exact_wb
        scores = W.abs()  # (feature_dim, input_dim)
    else:
        x_req = x.detach().clone().requires_grad_(True)
        z = _head_features(model, x_req)  # (n, feature_dim)
        feature_dim = z.shape[1]
        grads = []
        for j in range(feature_dim):
            g = torch.autograd.grad(
                z[:, j].sum(), x_req, retain_graph=(j < feature_dim - 1)
            )[0]
            grads.append(g.abs().mean(dim=0))  # (input_dim,)
        scores = torch.stack(grads, dim=0)  # (feature_dim, input_dim)

    out: list[list[tuple[int, float]]] = []
    k = min(top_k, scores.shape[1])
    for j in range(scores.shape[0]):
        vals, idx = torch.topk(scores[j], k)
        out.append([(int(idx[m].item()), float(vals[m].item())) for m in range(k)])
    return out
