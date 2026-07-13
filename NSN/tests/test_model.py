"""End-to-end DNN-EML smoke tests."""

import torch

from model import DNNEML
from utils import set_seed


def test_dnn_eml_forward():
    model = DNNEML.build(input_dim=2, feature_dim=4, head_depth=2, hidden_dim=16, num_layers=2)
    x = torch.randn(10, 2)
    y = model(x)
    assert y.shape == (10,)


def test_dnn_eml_training_step():
    set_seed(42)
    model = DNNEML.build(
        input_dim=1,
        feature_dim=4,
        head_depth=2,
        hidden_dim=64,
        num_layers=3,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.linspace(-2, 2, 128).unsqueeze(1)
    target = x.squeeze() ** 2

    for _ in range(500):
        opt.zero_grad()
        loss = model.mse_loss(x, target)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        final = model.mse_loss(x, target).item()
    assert final < 1e-2
