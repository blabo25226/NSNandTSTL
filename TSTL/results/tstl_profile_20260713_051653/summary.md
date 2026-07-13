# TSTL MLP layer profile

- seed: 42
- hidden layers: 5
- S_base (R²): -0.0290
- S_full (R²): 0.4152
- best layer k: 0 (C=1.046)
- mid-k indices: [1, 2, 3]

## C(k)

| k | S_k | C(k) |
|---|-----|------|
| 0 | 0.4357 | 1.0460 |
| 1 | 0.3470 | 0.8464 |
| 2 | 0.3849 | 0.9317 |
| 3 | 0.2204 | 0.5613 |
| 4 | 0.1579 | 0.4208 |

## Strategies (R²)

| strategy | R² |
|----------|-----|
| full | 0.4152 |
| only_b3 | 0.4449 |
| mid_3 | 0.2751 |
| boost_b3 | 0.3837 |
