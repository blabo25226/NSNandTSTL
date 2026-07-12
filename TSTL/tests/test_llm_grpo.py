"""Tests for llm_grpo reward (no TRL required)."""

from llm_grpo import gsm8k_numeric_reward


def test_gsm8k_numeric_reward_correct():
    completions = [[{"content": "The answer is 42"}]]
    answers = ["#### 42"]
    assert gsm8k_numeric_reward(completions, answers) == [1.0]


def test_gsm8k_numeric_reward_wrong():
    completions = [[{"content": "The answer is 41"}]]
    answers = ["#### 42"]
    assert gsm8k_numeric_reward(completions, answers) == [0.0]
