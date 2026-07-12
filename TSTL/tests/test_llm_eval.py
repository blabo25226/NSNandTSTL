"""Tests for llm_eval."""

from llm_eval import accuracy_score, extract_numeric_answer


def test_extract_gsm8k_answer():
    text = "Reasoning...\n#### 42"
    assert extract_numeric_answer(text) == "42"


def test_accuracy_score():
    preds = ["The answer is 10", "#### 3"]
    gold = ["#### 10", "#### 3"]
    assert accuracy_score(preds, gold) == 1.0
