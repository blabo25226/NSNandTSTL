"""Dataset loaders for TSTL R1 (Colab / HuggingFace)."""

from __future__ import annotations

from typing import Any

# Newer `datasets` requires namespace/name (not bare "gsm8k").
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_CONFIG = "main"


def load_gsm8k_subset(n_train: int, n_eval: int) -> tuple[Any, Any]:
    """Load GSM8K train/test subsets for R1 smoke runs."""
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET_ID, GSM8K_CONFIG)
    train = ds["train"].select(range(min(n_train, len(ds["train"]))))
    test = ds["test"].select(range(min(n_eval, len(ds["test"]))))
    return train, test


def to_grpo_rows(split: Any) -> list[dict[str, str]]:
    """Convert HF split to TRL GRPO rows with prompt + reference answer."""
    rows: list[dict[str, str]] = []
    for ex in split:
        prompt = f"Question: {ex['question']}\nAnswer:"
        rows.append({"prompt": prompt, "answer": ex["answer"]})
    return rows
