"""R1 Colab presets — trade accuracy for wall-clock time."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class R1Preset:
    name: str
    train_n: int
    eval_n: int
    steps: int
    num_generations: int
    max_completion_length: int
    layer_stride: int  # scan every N-th layer (1 = all layers)


QUICK = R1Preset(
    name="quick",
    train_n=32,
    eval_n=16,
    steps=25,
    num_generations=2,
    max_completion_length=64,
    layer_stride=4,
)

STANDARD = R1Preset(
    name="standard",
    train_n=128,
    eval_n=32,
    steps=80,
    num_generations=2,
    max_completion_length=128,
    layer_stride=2,
)

FULL = R1Preset(
    name="full",
    train_n=256,
    eval_n=64,
    steps=200,
    num_generations=4,
    max_completion_length=256,
    layer_stride=1,
)


def layer_indices_to_scan(num_layers: int, stride: int) -> list[int]:
    stride = max(1, stride)
    return list(range(0, num_layers, stride))


def estimate_grpo_forward_passes(preset: R1Preset, num_layers_scanned: int, *, include_full: bool = True) -> int:
    """Rough count of GRPO optimizer steps × generations (for wall-clock planning)."""
    runs = num_layers_scanned + (1 if include_full else 0)
    return runs * preset.steps * preset.num_generations
