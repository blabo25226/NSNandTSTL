"""
Compute-cost model for the EML operator (paper's hardware-efficiency claim).

The NSN paper reports a per-node cost of ~111 FLOPs for eml(x, y) = exp(x) - ln(y)
evaluated in complex arithmetic on general CPU/GPU. This module reconstructs that
figure analytically by counting the elementary and transcendental operations and
weighting each by a documented FLOP-equivalent, so the ~111 estimate is transparent
and adjustable (transcendental weights are hardware-dependent estimates, not exact).

  complex exp(a+bi) = e^a (cos b + i sin b)   -> exp, sin, cos + mults/clamps
  complex ln(z)     = ln|z| + i*atan2(Im,Re)  -> sqrt, ln, atan2 + mults/adds
  subtraction                                 -> 2 real subs

Under the default weights the transcendental operations alone sum to 111, matching
the paper; arithmetic overhead is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# FLOP-equivalent weights for double-precision software transcendentals. These are
# order-of-magnitude estimates (typical of software math libraries); the sum of the
# per-eml-node transcendental calls under these weights is 111, matching the paper.
DEFAULT_WEIGHTS: dict[str, float] = {
    "exp": 20.0,
    "sin": 18.0,
    "cos": 18.0,
    "log": 20.0,
    "atan2": 25.0,
    "sqrt": 10.0,
    "add": 1.0,
    "mul": 1.0,
    "cmp": 1.0,  # clamp min/max
}


@dataclass
class EMLNodeCost:
    transcendental_flops: float
    arithmetic_flops: float
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def total_flops(self) -> float:
        return self.transcendental_flops + self.arithmetic_flops


# Per-eml-node operation counts (complex exp + complex log + complex subtraction).
_EML_TRANSCENDENTAL_COUNTS: dict[str, int] = {
    "exp": 1,    # e^a in complex exp
    "sin": 1,    # sin b
    "cos": 1,    # cos b
    "log": 1,    # ln|z|
    "atan2": 1,  # phase of z
    "sqrt": 1,   # |z|
}
_EML_ARITHMETIC_COUNTS: dict[str, int] = {
    # complex exp: e^a * cos b, e^a * sin b (2 mul); clamp real+imag (4 cmp)
    # complex log: a^2, b^2 (2 mul), a^2+b^2 (1 add), eps shift (1 add)
    # subtraction: real, imag (2 add)
    "mul": 4,
    "add": 4,
    "cmp": 4,
}


def eml_node_flops(weights: dict[str, float] | None = None) -> EMLNodeCost:
    """Analytic FLOP cost of one eml(x, y) node in complex arithmetic."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    breakdown: dict[str, float] = {}

    transc = 0.0
    for op, n in _EML_TRANSCENDENTAL_COUNTS.items():
        c = n * w[op]
        breakdown[op] = c
        transc += c

    arith = 0.0
    for op, n in _EML_ARITHMETIC_COUNTS.items():
        c = n * w[op]
        breakdown[op] = c
        arith += c

    return EMLNodeCost(transcendental_flops=transc, arithmetic_flops=arith, breakdown=breakdown)


def leaf_affine_flops(feature_dim: int, weights: dict[str, float] | None = None) -> float:
    """Cost of a leaf l = alpha + beta^T z (+ gamma f_prev): 2*d mul/add + 2."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return feature_dim * (w["mul"] + w["add"]) + (w["mul"] + w["add"])  # gamma*f_prev + add


def head_flops(depth: int, feature_dim: int, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Total FLOPs of a depth-D EML tree head: (2^D - 1) nodes + 2^D leaves."""
    num_leaves = 2**depth
    num_internal = num_leaves - 1
    node = eml_node_flops(weights).total_flops
    leaf = leaf_affine_flops(feature_dim, weights)
    nodes_total = num_internal * node
    leaves_total = num_leaves * leaf
    return {
        "depth": depth,
        "num_internal_nodes": num_internal,
        "num_leaves": num_leaves,
        "eml_node_flops": node,
        "nodes_flops": nodes_total,
        "leaves_flops": leaves_total,
        "head_flops": nodes_total + leaves_total,
    }


def mlp_flops(layer_dims: list[int], weights: dict[str, float] | None = None) -> float:
    """
    FLOPs for an MLP forward pass over layer_dims = [in, h1, ..., out].

    Each linear layer: 2 * in * out (mul+add) MACs; hidden activations: 1 per unit.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = 0.0
    for i in range(len(layer_dims) - 1):
        din, dout = layer_dims[i], layer_dims[i + 1]
        total += dout * din * (w["mul"] + w["add"])
        if i < len(layer_dims) - 2:  # activation on hidden layers only
            total += dout * w["add"]
    return total
