"""Lightweight math QA evaluation for TSTL R1 (accuracy as score S)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numeric_answer(text: str) -> str | None:
    """Parse GSM8K-style `#### answer` or the last number in text."""
    if "####" in text:
        tail = text.split("####")[-1].strip()
        m = _NUM_RE.search(tail.replace(",", ""))
        return m.group(0) if m else None
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def answers_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()


@dataclass
class EvalExample:
    question: str
    answer: str


def accuracy_score(predictions: list[str], gold_answers: list[str]) -> float:
    """S = exact-match accuracy on numeric answers."""
    if not predictions:
        return 0.0
    correct = 0
    for pred_text, gold_text in zip(predictions, gold_answers):
        pred = extract_numeric_answer(pred_text)
        gold = extract_numeric_answer(gold_text)
        if answers_match(pred, gold):
            correct += 1
    return correct / len(predictions)


def mean_benchmark_score(
    predictions_by_split: dict[str, list[str]],
    gold_by_split: dict[str, list[str]],
) -> float:
    """Unweighted mean accuracy across named splits."""
    keys = sorted(predictions_by_split.keys())
    if not keys:
        return 0.0
    scores = [
        accuracy_score(predictions_by_split[k], gold_by_split[k])
        for k in keys
    ]
    return sum(scores) / len(scores)


def build_prompt(example: dict[str, Any]) -> str:
    """Match the GRPO training prompt format in llm_data.to_grpo_rows."""
    return f"Question: {example['question']}\nAnswer:"


def eval_model(
    model: Any,
    split: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 64,
    answer_key: str = "answer",
) -> float:
    """
    Score S = greedy-decode accuracy on a math QA split (paper §3.1 in-domain S).

    Runs on GPU when the model is on GPU; needs transformers-style ``model.generate``
    and a tokenizer. Kept out of the notebook so the CLI and notebook share one
    implementation.
    """
    import torch

    preds: list[str] = []
    gold: list[str] = []
    model.eval()
    for ex in split:
        prompt = build_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        gold.append(ex[answer_key])
    return accuracy_score(preds, gold)
