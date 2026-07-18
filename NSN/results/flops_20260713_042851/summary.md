# EML FLOPs Cost Analysis

Per-eml-node transcendental FLOPs = **111** (paper: ~111), + 12 arithmetic = **123** total.

Breakdown (weighted FLOP-equivalents): exp=20, sin=18, cos=18, log=20, atan2=25, sqrt=10, mul=4, add=4, cmp=4

MLP trunk [2, 64, 64, 6]: 9344 FLOPs.

| depth | eml nodes | node FLOPs | head FLOPs | +trunk = total |
|-------|-----------|------------|------------|----------------|
| 1 | 1 | 123 | 151 | 9495 |
| 2 | 3 | 123 | 425 | 9769 |
| 3 | 7 | 123 | 973 | 10317 |
| 4 | 15 | 123 | 2069 | 11413 |

The per-node transcendental cost (~111 FLOPs) is the paper's motivation for a dedicated EML cell (FPGA/analog); on CPU/GPU each node is expensive relative to a plain MAC. FPGA/analog synthesis is out of scope here.
