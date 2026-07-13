"""Tests for Odrzywołek 4-stage pipeline."""

import torch

from leaf_softmax import LeafSoftmaxMode
from model import DNNEML
from pipeline import OdrzywolekPipelineConfig, run_odrzywolek_pipeline
from utils import set_seed


def test_pipeline_runs_all_stages():
    set_seed(0)
    x = torch.linspace(0.3, 1.5, 64).unsqueeze(1)
    y = x.squeeze() ** 2
    model = DNNEML.build(input_dim=1, feature_dim=4, head_depth=2, hidden_dim=32, num_layers=2)
    cfg = OdrzywolekPipelineConfig(
        total_steps=120,
        search_steps=60,
        harden_steps=40,
        polish_steps=20,
        lr=5e-3,
        leaf_softmax_mode=LeafSoftmaxMode.SOFTMAX,
        seed=0,
    )
    result = run_odrzywolek_pipeline(model, x, y, cfg)
    assert len(result.stages.search_losses) == 60
    assert len(result.stages.harden_losses) == 40
    assert len(result.stages.polish_losses) == 20
    assert result.expression_eml.startswith("Re[eml(")
    assert result.final_mse < result.initial_mse


def test_pipeline_trunk_only_search_runs():
    set_seed(2)
    x = torch.linspace(0.3, 1.5, 64).unsqueeze(1)
    y = x.squeeze() ** 2
    model = DNNEML.build(input_dim=1, feature_dim=4, head_depth=2, hidden_dim=32, num_layers=2)
    cfg = OdrzywolekPipelineConfig(
        total_steps=100,
        search_steps=50,
        harden_steps=30,
        polish_steps=20,
        lr=5e-3,
        seed=2,
        trunk_only_search=True,
    )
    result = run_odrzywolek_pipeline(model, x, y, cfg)
    assert result.expression_eml.startswith("Re[eml(")
    assert result.final_mse < float("inf")


def test_pipeline_gumbel_mode():
    x = torch.rand(32, 2)
    y = x[:, 0] * x[:, 1]
    model = DNNEML.build(
        input_dim=2,
        feature_dim=4,
        head_depth=2,
        hidden_dim=32,
        num_layers=2,
        leaf_softmax_mode=LeafSoftmaxMode.GUMBEL,
    )
    cfg = OdrzywolekPipelineConfig(
        total_steps=80,
        search_steps=40,
        harden_steps=30,
        polish_steps=10,
        seed=1,
        leaf_softmax_mode=LeafSoftmaxMode.GUMBEL,
    )
    result = run_odrzywolek_pipeline(model, x, y, cfg)
    assert "eml(" in result.expression_eml
