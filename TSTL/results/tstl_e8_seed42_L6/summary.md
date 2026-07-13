# E8 (toy): ‖Δθ_k‖ vs C(k) — seed 42

- Pearson(‖Δθ‖, C): 0.511
- Spearman(‖Δθ‖, C): 0.429
- ‖Δθ‖ relative spread (max-min)/mean: 0.599

| k | ‖Δθ_k‖ | C(k) |
|---|--------|------|
| 0 | 16.894 | 0.230 |
| 1 | 16.644 | 0.834 |
| 2 | 18.161 | 1.000 |
| 3 | 24.264 | 0.845 |
| 4 | 24.090 | 0.993 |
| 5 | 29.578 | 0.991 |

Interpretation: TSTL §5 expects weak |Δθ|–C(k) correlation (contribution is not explained by how much a layer moves). Toy-scale, single seed.