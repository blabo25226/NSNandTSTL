"""Training loop for DNN-EML symbolic-regression experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from leaf_softmax import LeafSoftmaxMode
from model import DNNEML
from pipeline import OdrzywolekPipelineConfig, run_odrzywolek_pipeline
from targets import DEFAULT_NOISE_STD_REL, SRTarget
from utils import set_seed


@dataclass
class TrainConfig:
    feature_dim: int = 6
    head_depth: int = 2
    hidden_dim: int = 128
    num_layers: int = 3
    steps: int = 5000
    lr: float = 2e-3
    seed: int = 42
    n_train: int = 256
    noise_std_rel: float = DEFAULT_NOISE_STD_REL
    weight_decay: float = 1e-4
    leaf_softmax_mode: LeafSoftmaxMode | str = LeafSoftmaxMode.SOFTMAX
    use_odrzywolek_pipeline: bool = True
    polish_lr: float = 1e-4
    f_prev_mode: str = "zero"
    f_prev_passes: int = 1


@dataclass
class TrainResult:
    target_id: str
    final_mse: float
    initial_mse: float
    best_val_mse: float
    expression_eml: str
    simplified: str | None
    steps: int
    train_seconds: float
    stopped_step: int
    pipeline_search_steps: int = 0
    pipeline_harden_steps: int = 0
    pipeline_polish_steps: int = 0
    leaf_softmax_mode: str = "softmax"
    snapped_train_mse: float = float("nan")


def _config_for_target(target: SRTarget, base: TrainConfig) -> TrainConfig:
    cfg = TrainConfig(**vars(base))
    if target.id == "sum":
        cfg.feature_dim = 6
        # D=3 gives the EML tree enough capacity to represent the linear sum
        # (D=2 plateaus ~0.03-0.4); parent-mode f_prev feedback is numerically
        # unstable and does not help, so depth is the stable lever.
        cfg.head_depth = 3
        cfg.steps = 6000
        cfg.lr = 2e-3
    elif target.id == "sin_plus":
        cfg.feature_dim = 8
        cfg.head_depth = 2
        cfg.steps = 6000
        cfg.lr = 3e-3
        cfg.hidden_dim = 128
    elif target.id == "square":
        cfg.feature_dim = 4
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 1e-3
    elif target.id == "exp":
        cfg.feature_dim = 4
        cfg.head_depth = 1
        cfg.steps = 4000
        cfg.lr = 2e-3
    elif target.id == "sin":
        cfg.feature_dim = 6
        cfg.head_depth = 2
        cfg.steps = 6000
        cfg.lr = 2e-3
    elif target.id == "feynman_I9_inv_square":
        cfg.feature_dim = 4
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    elif target.id.startswith("feynman"):
        cfg.feature_dim = 6
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    elif target.input_dim == 1:
        cfg.feature_dim = 4
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    else:
        cfg.feature_dim = 6
        cfg.head_depth = 2
        cfg.steps = 5000
        cfg.lr = 2e-3
    return cfg


def _parse_leaf_mode(mode: LeafSoftmaxMode | str) -> LeafSoftmaxMode:
    if isinstance(mode, LeafSoftmaxMode):
        return mode
    return LeafSoftmaxMode.parse(mode)


def train_target(target: SRTarget, base_config: TrainConfig | None = None) -> tuple[DNNEML, TrainResult]:
    import time

    t0 = time.perf_counter()
    cfg = _config_for_target(target, base_config or TrainConfig())
    leaf_mode = _parse_leaf_mode(cfg.leaf_softmax_mode)
    set_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    x, y = target.sample(cfg.n_train, gen, cfg.noise_std_rel)

    model = DNNEML.build(
        input_dim=target.input_dim,
        feature_dim=cfg.feature_dim,
        head_depth=cfg.head_depth,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        leaf_softmax_mode=leaf_mode,
        f_prev_mode=cfg.f_prev_mode,
        f_prev_passes=cfg.f_prev_passes,
    )

    if cfg.use_odrzywolek_pipeline:
        pipe_cfg = OdrzywolekPipelineConfig(
            total_steps=cfg.steps,
            lr=cfg.lr,
            polish_lr=cfg.polish_lr,
            weight_decay=cfg.weight_decay,
            leaf_softmax_mode=leaf_mode,
            seed=cfg.seed,
        )
        pipe = run_odrzywolek_pipeline(model, x, y, pipe_cfg)
        elapsed = time.perf_counter() - t0
        result = TrainResult(
            target_id=target.id,
            final_mse=pipe.final_mse,
            initial_mse=pipe.initial_mse,
            best_val_mse=pipe.final_mse,
            expression_eml=pipe.expression_eml,
            simplified=pipe.simplified,
            steps=cfg.steps,
            train_seconds=elapsed,
            stopped_step=cfg.steps,
            pipeline_search_steps=pipe.search_steps,
            pipeline_harden_steps=pipe.harden_steps,
            pipeline_polish_steps=pipe.polish_steps,
            leaf_softmax_mode=leaf_mode.value,
            snapped_train_mse=pipe.snapped_train_mse,
        )
        return model, result

    raise RuntimeError("Legacy training loop removed; use Odrzywołek pipeline.")
