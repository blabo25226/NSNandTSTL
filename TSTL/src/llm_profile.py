"""Layer-contribution profiling orchestration for LLM / GRPO runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch.nn as nn

from layer_contribution import (
    compute_contributions,
    plot_contribution_heatmap,
    save_contributions_json,
)


@dataclass
class LayerScanResult:
    s_base: float
    s_full: float
    s_per_layer: dict[int, float]
    contributions: dict[int, float]
    out_dir: Path


def depth_normalized_positions(num_layers: int) -> dict[int, float]:
    """Map layer index k to relative depth in [0, 1] (paper Figure 1)."""
    if num_layers <= 1:
        return {0: 0.0}
    return {k: k / (num_layers - 1) for k in range(num_layers)}


def plot_depth_normalized(
    contributions: dict[int, float],
    out_path: Path | str,
    *,
    title: str = "C(k) vs depth-normalized position",
) -> None:
    """Scatter-style plot: x = k/(L-1), y = C(k)."""
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(contributions.keys())
    if not keys:
        return

    n = keys[-1] + 1 if keys else 1
    positions = depth_normalized_positions(n)
    vals = [contributions[k] for k in keys]

    width, height = 640, 400
    margin_l, margin_b, margin_t = 60, 50, 30
    plot_w = width - margin_l - 30
    plot_h = height - margin_b - margin_t

    vmin = min(min(vals), 0.0)
    vmax = max(max(vals), 1.0)
    span = vmax - vmin if vmax > vmin else 1.0

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 8), title, fill="black")
    draw.text((margin_l, height - 20), "depth (0=input, 1=output)", fill="black")

    for k, v in zip(keys, vals):
        x = margin_l + int(positions[k] * plot_w)
        y = margin_t + plot_h - int((v - vmin) / span * plot_h)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="steelblue", outline="black")

    y_one = margin_t + plot_h - int((1.0 - vmin) / span * plot_h)
    draw.line([margin_l, y_one, width - 30, y_one], fill="green", width=1)
    img.save(out_path)


def default_r1_out_dir(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1] / "results"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"tstl_r1_{stamp}"


def save_layer_scan(
    out_dir: Path,
    *,
    s_base: float,
    s_full: float,
    s_per_layer: dict[int, float],
    contributions: dict[int, float],
    config: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_contributions_json(
        out_dir / "contributions.json",
        s_base=s_base,
        s_full=s_full,
        s_per_layer=s_per_layer,
        contributions=contributions,
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plot_contribution_heatmap(contributions, out_dir / "contributions_bar.png")
    plot_depth_normalized(contributions, out_dir / "contributions_depth.png")

    best_k = max(contributions, key=lambda k: contributions[k])
    lines = [
        "# TSTL R1 layer scan",
        "",
        f"- S_base: {s_base:.4f}",
        f"- S_full: {s_full:.4f}",
        f"- best layer: k={best_k}, C={contributions[best_k]:.3f}",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def profile_layers_from_scores(
    s_base: float,
    s_full: float,
    s_per_layer: dict[int, float],
    out_dir: Path,
    config: dict,
) -> LayerScanResult:
    contributions = compute_contributions(s_base, s_full, s_per_layer)
    save_layer_scan(
        out_dir,
        s_base=s_base,
        s_full=s_full,
        s_per_layer=s_per_layer,
        contributions=contributions,
        config=config,
    )
    return LayerScanResult(
        s_base=s_base,
        s_full=s_full,
        s_per_layer=s_per_layer,
        contributions=contributions,
        out_dir=out_dir,
    )


def resume_layer_scan(
    out_dir: Path,
    num_layers: int,
    train_and_eval_layer: Callable[[int], float],
    *,
    s_base: float,
    s_full: float,
    config: dict,
    layer_indices: list[int] | None = None,
) -> LayerScanResult:
    """
    Run per-layer training, skipping layers already in contributions.json.

    ``layer_indices`` limits which k to train (default: 0 .. num_layers-1).
    """
    contrib_path = out_dir / "contributions.json"
    s_per_layer: dict[int, float] = {}
    if contrib_path.exists():
        data = json.loads(contrib_path.read_text(encoding="utf-8"))
        s_per_layer = {int(k): float(v) for k, v in data.get("s_per_layer", {}).items()}

    indices = layer_indices if layer_indices is not None else list(range(num_layers))
    for k in indices:
        if k in s_per_layer:
            continue
        s_per_layer[k] = train_and_eval_layer(k)

    return profile_layers_from_scores(s_base, s_full, s_per_layer, out_dir, config)
