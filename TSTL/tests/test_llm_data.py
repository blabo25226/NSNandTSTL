"""Tests for llm_data."""

from llm_data import GSM8K_DATASET_ID, to_grpo_rows


def test_gsm8k_dataset_id_has_namespace():
    assert "/" in GSM8K_DATASET_ID
    assert GSM8K_DATASET_ID == "openai/gsm8k"


def test_to_grpo_rows():
    split = [{"question": "1+1?", "answer": "#### 2"}]
    rows = to_grpo_rows(split)
    assert rows[0]["prompt"].startswith("Question:")
    assert rows[0]["answer"] == "#### 2"
