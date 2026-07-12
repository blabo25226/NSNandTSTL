"""Layer contribution C(k) and profiling plots.

Performance score S uses R² for regression (Phase 0-B).
For GRPO / LLM benchmarks (Phase 0-B+), replace S with mean benchmark accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from utils import r2_score


def score_from_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """S = R²."""
    return r2_score(y_true, y_pred)


def layer_contribution(s_base: float, s_full: float, s_k: float) -> float:
    """
    C(k) = (S_k - S_base) / (S_full - S_base).

    Returns float('nan') if denominator is near zero.
    """
    denom = s_full - s_base
    if abs(denom) < 1e-12:
        return float("nan")
    return (s_k - s_base) / denom


def compute_contributions(
    s_base: float,
    s_full: float,
    s_per_layer: dict[int, float],
) -> dict[int, float]:
    return {k: layer_contribution(s_base, s_full, s_k) for k, s_k in s_per_layer.items()}


def plot_contribution_heatmap(
    contributions: dict[int, float],
    out_path: Path | str,
    *,
    title: str = "Layer contribution C(k)",
) -> None:
    """Simple bar chart saved as PNG (PIL, no matplotlib GUI backend)."""
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(contributions.keys())
    vals = [contributions[k] for k in keys]
    if not keys:
        return

    width, height = 640, 360
    margin_l, margin_b, margin_t = 50, 40, 30
    plot_w = width - margin_l - 20
    plot_h = height - margin_b - margin_t

    vmin = min(min(vals), 0.0)
    vmax = max(max(vals), 1.0)
    span = vmax - vmin if vmax > vmin else 1.0

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 8), title, fill="black")

    n = len(keys)
    bar_w = max(8, plot_w // max(n, 1) - 8)
    for i, (k, v) in enumerate(zip(keys, vals)):
        x0 = margin_l + i * (plot_w // n) + 4
        h = int((v - vmin) / span * plot_h)
        y0 = margin_t + plot_h - h
        y1 = margin_t + plot_h
        draw.rectangle([x0, y0, x0 + bar_w, y1], fill="steelblue", outline="black")
        draw.text((x0, y1 + 4), str(k), fill="black")

    # Reference line at C=1.0
    y_one = margin_t + plot_h - int((1.0 - vmin) / span * plot_h)
    draw.line([margin_l, y_one, width - 20, y_one], fill="green", width=1)

    img.save(out_path)


def save_contributions_json(
    path: Path | str,
    *,
    s_base: float,
    s_full: float,
    s_per_layer: dict[int, float],
    contributions: dict[int, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "s_base": s_base,
        "s_full": s_full,
        "s_per_layer": {str(k): v for k, v in s_per_layer.items()},
        "contributions": {str(k): v for k, v in contributions.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def profile_single_layers(
    model_factory: Callable[[], nn.Module],
    train_fn: Callable[[nn.Module, list[int] | None], None],
    eval_fn: Callable[[nn.Module], float],
    num_hidden: int,
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    """
    Run full training then per-layer training from the same initialization.

    train_fn(model, hidden_indices): None = full training; {k} = only layer k.
    Returns (s_base, s_full, s_per_layer, contributions).
    """
    base_model = model_factory()
    s_base = eval_fn(base_model)

    full_model = model_factory()
    train_fn(full_model, None)
    s_full = eval_fn(full_model)

    s_per_layer: dict[int, float] = {}
    for k in range(num_hidden):
        m = model_factory()
        train_fn(m, [k])
        s_per_layer[k] = eval_fn(m)

    contribs = compute_contributions(s_base, s_full, s_per_layer)
    return s_base, s_full, s_per_layer, contribs
