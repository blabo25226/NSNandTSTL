# TSTL MLP layer profile

- seed: 0
- hidden layers: 5
- S_base (R²): -0.2138
- S_full (R²): 0.3704
- best layer k: 0 (C=1.232)
- mid-k indices: [1, 2, 3]

## C(k)

| k | S_k | C(k) |
|---|-----|------|
| 0 | 0.5058 | 1.2318 |
| 1 | 0.3895 | 1.0327 |
| 2 | 0.3589 | 0.9803 |
| 3 | 0.3002 | 0.8799 |
| 4 | 0.1151 | 0.5631 |

## Strategies (R²)

| strategy | R² |
|----------|-----|
| full | 0.3704 |
| only_b3 | 0.5398 |
| mid_3 | 0.2839 |
| boost_b3 | 0.4542 |
