"""Tests for llm_grpo reward (no TRL required)."""

from llm_grpo import completion_text, gsm8k_numeric_reward


def test_completion_text_string():
    assert completion_text("hello") == "hello"


def test_completion_text_conversational():
    assert completion_text([{"content": "42"}]) == "42"


def test_gsm8k_numeric_reward_conversational():
    completions = [[{"content": "The answer is 42"}]]
    answers = ["#### 42"]
    assert gsm8k_numeric_reward(completions, answers) == [1.0]


def test_gsm8k_numeric_reward_standard_strings():
    completions = ["Reasoning...\nThe answer is 42"]
    answers = ["#### 42"]
    assert gsm8k_numeric_reward(completions, answers) == [1.0]


def test_gsm8k_numeric_reward_wrong():
    completions = ["The answer is 41"]
    answers = ["#### 42"]
    assert gsm8k_numeric_reward(completions, answers) == [0.0]
