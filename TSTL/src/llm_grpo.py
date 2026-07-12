"""GRPO training wrapper for TSTL (TRL-first, Colab GPU)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn

from llm_freeze import freeze_all_except_layers


@dataclass
class GrpoRunConfig:
    """Minimal GRPO run settings for R1/R2."""

    output_dir: Path
    learning_rate: float = 1e-5
    num_train_steps: int = 200
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 256
    max_completion_length: int = 128
    seed: int = 42
    train_layer_indices: list[int] | None = None  # None = full model
    extra: dict[str, Any] = field(default_factory=dict)


def grpo_dependencies_available() -> bool:
    try:
        import transformers  # noqa: F401
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


def apply_layer_policy(model: nn.Module, layer_indices: list[int] | None) -> None:
    """Set requires_grad according to TSTL single-layer / full policy."""
    freeze_all_except_layers(model, layer_indices)


def build_grpo_trainer(model: nn.Module, dataset: Any, config: GrpoRunConfig) -> Any:
    """
    Construct a TRL GRPOTrainer. Requires GPU + transformers + trl (Colab).

    Raises ImportError on CPU-only local dev machines without TRL.
    """
    if not grpo_dependencies_available():
        raise ImportError(
            "GRPO training needs transformers and trl. "
            "Install TSTL/requirements-r.txt on Colab."
        )

    from trl import GRPOConfig, GRPOTrainer

    apply_layer_policy(model, config.train_layer_indices)

    training_args = GRPOConfig(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        max_steps=config.num_train_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        seed=config.seed,
        logging_steps=10,
        save_steps=max(config.num_train_steps, 1),
        report_to="none",
        **config.extra,
    )
    return GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )


def run_grpo_train(model: nn.Module, dataset: Any, config: GrpoRunConfig) -> Path:
    """Run GRPO and return checkpoint directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_grpo_trainer(model, dataset, config)
    trainer.train()
    trainer.save_model(str(config.output_dir))
    return config.output_dir
