# E8 aggregate (toy): ‖Δθ_k‖ vs C(k) across seeds

- seeds: 8
- Pearson mean/median (all): 0.019 / -0.122
- Spearman mean/median (all): -0.114 / -0.171
- Pearson mean (excl. under-converged seed 2): -0.077
- Spearman mean (excl. seed 2): -0.184
- ‖Δθ‖ relative spread mean: 0.463

| seed | Pearson | Spearman | Δspread |
|------|---------|----------|---------|
| 0 | 0.207 | 0.029 | 0.423 |
| 11 | -0.084 | 0.086 | 0.465 |
| 13 | -0.252 | -0.600 | 0.372 |
| 1 | -0.292 | -0.371 | 0.388 |
| 2 | 0.694 | 0.371 | 0.414 |
| 3 | -0.473 | -0.486 | 0.466 |
| 42 | 0.511 | 0.429 | 0.599 |
| 7 | -0.160 | -0.371 | 0.578 |

Interpretation: across seeds the ‖Δθ‖–C(k) correlation averages near zero with
inconsistent sign — reproducing TSTL §5 (contribution is not explained by how
much a layer's weights move). Toy Transformer, supervised task; seed 2 under-
converged (S_full=0.61) and is an outlier.