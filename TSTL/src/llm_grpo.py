"""GRPO training wrapper for TSTL (TRL-first, Colab GPU)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn

from llm_eval import answers_match, extract_numeric_answer
from llm_freeze import freeze_all_except_layers


@dataclass
class GrpoRunConfig:
    """Minimal GRPO run settings for R1/R2."""

    output_dir: Path
    learning_rate: float = 1e-5
    num_train_steps: int = 200
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_completion_length: int = 256
    num_generations: int = 4
    seed: int = 42
    train_layer_indices: list[int] | None = None  # None = full model
    tokenizer: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def grpo_dependencies_available() -> bool:
    try:
        import transformers  # noqa: F401
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


def completion_text(completion: Any) -> str:
    """
    Normalize TRL completion payloads to plain text.

    TRL standard prompts -> completions are list[str].
    Conversational format -> list[list[dict]] with 'content'.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(first.get("content", first.get("text", first)))
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", completion)))
    return str(completion)


def gsm8k_numeric_reward(
    completions: list[Any],
    answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Exact-match reward on GSM8K numeric answers (no math_verify dependency)."""
    if answer is None:
        answer = kwargs.get("answer") or kwargs.get("solution")
    if answer is None:
        raise ValueError("gsm8k_numeric_reward needs 'answer' column in dataset")

    rewards: list[float] = []
    for completion, gold_text in zip(completions, answer, strict=True):
        pred_text = completion_text(completion)
        pred = extract_numeric_answer(pred_text)
        gold = extract_numeric_answer(gold_text)
        rewards.append(1.0 if answers_match(pred, gold) else 0.0)
    return rewards


def apply_layer_policy(model: nn.Module, layer_indices: list[int] | None) -> None:
    freeze_all_except_layers(model, layer_indices)


def _coerce_train_dataset(dataset: Any) -> Any:
    if isinstance(dataset, list):
        from datasets import Dataset

        return Dataset.from_list(dataset)
    return dataset


def _filter_grpo_config_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keys unsupported by the installed TRL GRPOConfig (version-safe)."""
    from trl import GRPOConfig

    params = inspect.signature(GRPOConfig.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def build_grpo_trainer(
    model: nn.Module,
    dataset: Any,
    config: GrpoRunConfig,
    *,
    tokenizer: Any | None = None,
    reward_funcs: Callable[..., list[float]] | None = None,
) -> Any:
    """
    Construct a TRL GRPOTrainer. Requires GPU + transformers + trl (Colab).
    """
    if not grpo_dependencies_available():
        raise ImportError(
            "GRPO training needs transformers and trl. "
            "Install TSTL/requirements-r.txt on Colab."
        )

    from trl import GRPOConfig, GRPOTrainer

    apply_layer_policy(model, config.train_layer_indices)
    train_dataset = _coerce_train_dataset(dataset)
    reward = reward_funcs or gsm8k_numeric_reward

    raw_kwargs = {
        "output_dir": str(config.output_dir),
        "learning_rate": config.learning_rate,
        "max_steps": config.num_train_steps,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_completion_length": config.max_completion_length,
        "num_generations": config.num_generations,
        "seed": config.seed,
        "logging_steps": 10,
        "save_steps": max(config.num_train_steps, 1),
        "report_to": "none",
        "remove_unused_columns": False,
        **config.extra,
    }
    training_args = GRPOConfig(**_filter_grpo_config_kwargs(raw_kwargs))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "reward_funcs": reward,
        "train_dataset": train_dataset,
    }
    if tokenizer is not None:
        if "processing_class" in inspect.signature(GRPOTrainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

    return GRPOTrainer(**trainer_kwargs)


def run_grpo_train(
    model: nn.Module,
    dataset: Any,
    config: GrpoRunConfig,
) -> Path:
    """Run GRPO and return checkpoint directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_grpo_trainer(
        model,
        dataset,
        config,
        tokenizer=config.tokenizer,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    return config.output_dir
