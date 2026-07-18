"""Symbolic snapping and hardening schedules."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from eml_tree import EMLTreeHead
    from model import DNNEML


def temperature_schedule(
    step: int,
    total_steps: int,
    start_temp: float = 1.0,
    end_temp: float = 0.1,
    harden_start_frac: float = 0.0,
) -> float:
    """Linear cooling for leaf softmax; optional delay until harden_start_frac."""
    if total_steps <= 0:
        return end_temp
    frac = min(max(step / total_steps, 0.0), 1.0)
    if frac < harden_start_frac:
        return start_temp
    local = (frac - harden_start_frac) / max(1.0 - harden_start_frac, 1e-9)
    local = min(max(local, 0.0), 1.0)
    return start_temp + (end_temp - start_temp) * local


def apply_hardening(
    head: "EMLTreeHead",
    step: int,
    total_steps: int,
    harden_start_frac: float = 0.0,
) -> float:
    temp = temperature_schedule(
        step, total_steps, harden_start_frac=harden_start_frac
    )
    head.set_temperature(temp)
    return temp


def snap_leaf_weights(head: "EMLTreeHead") -> torch.Tensor:
    """Hard snap: one-hot per leaf from argmax over the effective logits."""
    with torch.no_grad():
        idx = head.effective_logits().argmax(dim=-1)
        hard = torch.zeros_like(head.leaf_logits)
        hard.scatter_(1, idx.unsqueeze(-1), 1.0)
    return hard


def apply_snap_to_logits(head: "EMLTreeHead", strength: float = 20.0) -> None:
    """Write snapped one-hot logits back into head (inference / export mode)."""
    with torch.no_grad():
        hard = snap_leaf_weights(head)
        head.leaf_logits.copy_(hard * strength)


def evaluate_snapped(
    model: "DNNEML",
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """
    MSE of the *snapped* model, i.e. the numerical realisation of the exported
    closed-form EML expression (paper Sec. 4.2).

    The snapped model sets each leaf's logits to a one-hot argmax, so its output
    equals `export_symbolic_expression(...)` evaluated on z = trunk(x). Soft
    weights are restored before returning, so the caller's model is unchanged.

    Returns (snapped_mse, snapped_predictions).
    """
    soft_state = copy.deepcopy(model.state_dict())
    was_training = model.training
    model.eval()
    apply_snap_to_logits(model.head)
    with torch.no_grad():
        pred = model(x)
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        y_ref = y.unsqueeze(0) if y.dim() == 0 else y
        mse = torch.mean((pred - y_ref) ** 2).item()
    model.load_state_dict(soft_state)
    model.train(was_training)
    return mse, pred.detach()


def _leaf_alpha_or_beta(head: "EMLTreeHead", leaf_index: int, z_names: list[str]) -> str:
    """Symbol for a leaf snapped to the alpha (constant) or beta (linear) branch."""
    beta = head.beta[leaf_index]
    terms = [
        f"{beta[j].item():.6g}*{z_names[j]}"
        for j in range(head.feature_dim)
        if abs(beta[j].item()) > 1e-8
    ]
    if not terms:
        return "0"
    return "(" + " + ".join(terms) + ")"


def _leaf_symbol_snapped(
    head: "EMLTreeHead",
    leaf_index: int,
    z_names: list[str],
    f_prev_pass: int = 0,
) -> str:
    """
    Symbol of leaf `leaf_index` at recurrence pass `f_prev_pass`.

    Mirrors `EMLTreeHead.leaf_values` under a one-hot (snapped) selection:
      - alpha branch  -> constant string (pass-invariant).
      - beta branch   -> linear-in-z string (pass-invariant).
      - gamma branch  -> f_prev = parent EML node output. At pass 0 f_prev is the
        constant `f_prev_const`; at pass t >= 1 it equals eml(sibling-pair symbols
        evaluated at pass t-1), matching `_parent_outputs`.
    """
    logits = head.effective_logits()[leaf_index]
    mode = int(logits.argmax().item())
    if mode == 0:
        return f"{head.alpha[leaf_index].item():.6g}"
    if mode == 1:
        return _leaf_alpha_or_beta(head, leaf_index, z_names)
    # gamma / f_prev branch (only reachable in "parent" mode, where it is unmasked)
    if f_prev_pass <= 0:
        return f"{head.f_prev_const:.6g}"
    k = leaf_index // 2
    left = _leaf_symbol_snapped(head, 2 * k, z_names, f_prev_pass - 1)
    right = _leaf_symbol_snapped(head, 2 * k + 1, z_names, f_prev_pass - 1)
    return f"eml({left}, {right})"


def export_symbolic_expression(head: "EMLTreeHead", z_names: list[str] | None = None) -> str:
    """
    Build nested eml(...) string from snapped (or nearly snapped) leaf logits.

    Output form: Re[eml(...)] matching paper Eq. (10). In "parent" f_prev mode the
    gamma branch is expanded into the K-pass master-formula recurrence so the
    exported closed form reproduces the snapped forward pass exactly.
    """
    if z_names is None:
        z_names = [f"z{i}" for i in range(head.feature_dim)]
    if len(z_names) != head.feature_dim:
        raise ValueError("z_names length must match feature_dim")

    final_pass = head.f_prev_passes if head.f_prev_mode == "parent" else 0
    leaves = [
        _leaf_symbol_snapped(head, i, z_names, final_pass)
        for i in range(head.num_leaves)
    ]
    level = leaves
    while len(level) > 1:
        next_level: list[str] = []
        for i in range(0, len(level), 2):
            next_level.append(f"eml({level[i]}, {level[i + 1]})")
        level = next_level
    return f"Re[{level[0]}]"


def export_leaf_summary(head: "EMLTreeHead", temperature: float | None = None) -> list[str]:
    """Human-readable summary of each leaf's dominant term."""
    temp = temperature if temperature is not None else head.temperature
    weights = F.softmax(head.leaf_logits / temp, dim=-1)
    lines: list[str] = []
    for i in range(head.num_leaves):
        w = weights[i]
        dominant = ("alpha", "beta·z", "gamma·f_prev")[int(w.argmax())]
        lines.append(
            f"leaf[{i}]: dominant={dominant} "
            f"(w=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f})) "
            f"alpha={head.alpha[i].item():.4f}"
        )
    return lines


def export_tree_structure(depth: int) -> str:
    n_leaves = 2**depth
    n_internal = n_leaves - 1
    return (
        f"EMLTree(depth={depth}, leaves={n_leaves}, "
        f"internal_nodes={n_internal}, root=eml(...))"
    )
