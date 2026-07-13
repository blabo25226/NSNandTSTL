"""Odrzywołek 4-stage pipeline: SEARCH → HARDEN → POLISH → SNAP."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from leaf_softmax import LeafSoftmaxMode
from model import DNNEML
from simplify import simplify_eml_expression
from snap import (
    apply_snap_to_logits,
    evaluate_snapped,
    export_symbolic_expression,
    temperature_schedule,
)


@dataclass
class OdrzywolekPipelineConfig:
    """Stage lengths must sum to `total_steps` when all three are set explicitly."""

    total_steps: int = 5000
    search_steps: int | None = None
    harden_steps: int | None = None
    polish_steps: int | None = None
    lr: float = 2e-3
    polish_lr: float = 1e-4
    weight_decay: float = 1e-4
    leaf_softmax_mode: LeafSoftmaxMode = LeafSoftmaxMode.SOFTMAX
    harden_end_temp: float = 0.1
    seed: int = 42
    # Head-capacity study: freeze the trunk after SEARCH so the EML head must
    # carry the symbolic load during HARDEN/POLISH instead of the black-box MLP.
    freeze_trunk_after_search: bool = False
    # Snap-aware POLISH: commit the discrete leaf choices (argmax -> one-hot)
    # *before* polishing, so the continuous params (alpha/beta/trunk) are fit to
    # the exact discrete tree that will be exported. This makes the snapped
    # closed-form faithful by construction (snapped MSE ~= final soft MSE)
    # instead of restoring soft weights that argmax then discards.
    snap_aware_polish: bool = True


@dataclass
class PipelineStageLog:
    search_losses: list[float] = field(default_factory=list)
    harden_losses: list[float] = field(default_factory=list)
    polish_losses: list[float] = field(default_factory=list)


@dataclass
class PipelineResult:
    final_mse: float
    initial_mse: float
    expression_eml: str
    simplified: str | None
    stages: PipelineStageLog
    search_steps: int
    harden_steps: int
    polish_steps: int
    snapped_train_mse: float = float("nan")


def _resolve_stage_lengths(cfg: OdrzywolekPipelineConfig) -> tuple[int, int, int]:
    total = cfg.total_steps
    if cfg.search_steps is not None and cfg.harden_steps is not None and cfg.polish_steps is not None:
        return cfg.search_steps, cfg.harden_steps, cfg.polish_steps
    search = int(total * 0.6)
    harden = int(total * 0.3)
    polish = max(1, total - search - harden)
    return search, harden, polish


def _set_head_mode(head, mode: LeafSoftmaxMode, gen: torch.Generator) -> None:
    head.leaf_softmax_mode = mode
    head.gumbel_generator = gen


def _train_step(model: DNNEML, x: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer) -> float:
    opt.zero_grad()
    loss = model.mse_loss(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError("NaN loss during Odrzywołek pipeline")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def run_odrzywolek_pipeline(
    model: DNNEML,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: OdrzywolekPipelineConfig,
) -> PipelineResult:
    """
    SEARCH: exploratory training at high temperature.
    HARDEN: anneal temperature toward discrete leaf choices.
    POLISH: fine-tune alpha/beta/trunk with frozen leaf logits.
    SNAP: export symbolic EML string (model weights restored to soft form).
    """
    search_n, harden_n, polish_n = _resolve_stage_lengths(cfg)
    gen = torch.Generator().manual_seed(cfg.seed + 17)
    stages = PipelineStageLog()

    head = model.head
    _set_head_mode(head, cfg.leaf_softmax_mode, gen)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- SEARCH ---
    head.set_temperature(1.0)
    for _ in range(search_n):
        model.train()
        stages.search_losses.append(_train_step(model, x, y, opt))

    if cfg.freeze_trunk_after_search:
        for p in model.trunk.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    # --- HARDEN ---
    for step in range(1, harden_n + 1):
        temp = temperature_schedule(
            step, harden_n, end_temp=cfg.harden_end_temp, harden_start_frac=0.0
        )
        head.set_temperature(temp)
        model.train()
        stages.harden_losses.append(_train_step(model, x, y, opt))

    # --- POLISH ---
    if cfg.snap_aware_polish:
        # Commit discrete leaf choices, then fit continuous params to that exact
        # tree. With one-hot(+-20) logits and a low temperature the forward pass
        # already equals the exported/snapped model, so POLISH minimises the
        # snapped loss directly.
        apply_snap_to_logits(head)
    head.leaf_logits.requires_grad_(False)
    polish_params = [
        p for n, p in model.named_parameters()
        if "leaf_logits" not in n and p.requires_grad
    ]
    polish_opt = torch.optim.AdamW(polish_params, lr=cfg.polish_lr, weight_decay=cfg.weight_decay)
    head.set_temperature(cfg.harden_end_temp)
    for _ in range(polish_n):
        model.train()
        stages.polish_losses.append(_train_step(model, x, y, polish_opt))

    head.leaf_logits.requires_grad_(True)
    if cfg.freeze_trunk_after_search:
        for p in model.trunk.parameters():
            p.requires_grad_(True)
    all_losses = stages.search_losses + stages.harden_losses + stages.polish_losses

    # --- SNAP (export + faithfulness measurement) ---
    # snapped_train_mse quantifies how much accuracy the closed-form export loses
    # relative to the soft model (paper's snapping-success criterion).
    snapped_train_mse, _ = evaluate_snapped(model, x, y)

    soft_state = copy.deepcopy(model.state_dict())
    model.eval()
    apply_snap_to_logits(model.head)
    z_names = [f"z{i}" for i in range(model.head.feature_dim)]
    expr = export_symbolic_expression(model.head, z_names)
    simplified = simplify_eml_expression(expr)
    model.load_state_dict(soft_state)

    return PipelineResult(
        final_mse=all_losses[-1] if all_losses else float("nan"),
        initial_mse=all_losses[0] if all_losses else float("nan"),
        expression_eml=expr,
        simplified=simplified if simplified != expr.replace("Re[", "").replace("]", "") else None,
        stages=stages,
        search_steps=search_n,
        harden_steps=harden_n,
        polish_steps=polish_n,
        snapped_train_mse=snapped_train_mse,
    )
