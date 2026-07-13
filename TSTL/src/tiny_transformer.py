"""Tiny from-scratch causal Transformer for CPU-only TSTL layer scans.

No HuggingFace download and no GPU required. The layout mirrors a real
decoder LM (``embed_tokens`` / ``layers`` / ``lm_head``) so the existing
``llm_freeze`` utilities apply unchanged: a single-layer run trains block ``k``
plus the token embedding and LM head, exactly like the paper's ``θ_emb`` /
``θ_head`` convention.

Task: modular running-sum next-token prediction. Given a random sequence
``x_0..x_{T-1}`` over ``[0, vocab)``, the target at position ``t`` is
``(x_0 + ... + x_t) mod vocab``. Recovering it needs the model to compose
information across positions/layers, so it is a non-trivial signal for C(k).
Score ``S`` is per-token accuracy on a held-out set (base ≈ 1/vocab).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TinyConfig:
    vocab_size: int = 7
    seq_len: int = 16
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 6
    d_ff: int = 128
    dropout: float = 0.0


class _Block(nn.Module):
    """Pre-LN causal self-attention + MLP block (one TSTL 'layer')."""

    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


def _sinusoidal_positions(seq_len: int, d_model: int) -> torch.Tensor:
    """Non-learned positional encoding (frozen buffer -> no freeze concerns)."""
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class TinyCausalLM(nn.Module):
    """Minimal decoder LM with HF-like attribute names for llm_freeze reuse."""

    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pos_enc", _sinusoidal_positions(cfg.seq_len, cfg.d_model), persistent=False
        )
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        h = self.embed_tokens(idx) + self.pos_enc[:t]
        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device), diagonal=1
        )
        for block in self.layers:
            h = block(h, causal)
        return self.lm_head(self.norm(h))


def make_dataset(
    cfg: TinyConfig, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, targets) for the modular running-sum task."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (n, cfg.seq_len), generator=gen)
    y = torch.cumsum(x, dim=1) % cfg.vocab_size
    return x, y


@torch.no_grad()
def token_accuracy(model: TinyCausalLM, x: torch.Tensor, y: torch.Tensor) -> float:
    """S = per-token next-target accuracy (base ≈ 1/vocab_size)."""
    model.eval()
    logits = model(x)
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train_model(
    model: TinyCausalLM,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> None:
    """
    Cross-entropy training over trainable params only (freeze applied upstream).

    Shared by the layer scan and the ‖Δθ‖ analysis so both use the exact same
    optimization; the freeze policy is set by the caller before calling this.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        xb, yb = x[idx], y[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)
        )
        if not torch.isfinite(loss):
            raise RuntimeError("NaN loss during training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
