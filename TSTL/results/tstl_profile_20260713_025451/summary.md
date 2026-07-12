# TSTL MLP layer profile

- seed: 0
- hidden layers: 5
- S_base (R²): -0.2138
- S_full (R²): 0.3427
- best layer k: 2 (C=0.570)
- mid-k indices: [1, 2, 3]

## C(k)

| k | S_k | C(k) |
|---|-----|------|
| 0 | 0.0517 | 0.4772 |
| 1 | 0.0640 | 0.4992 |
| 2 | 0.1033 | 0.5698 |
| 3 | 0.0589 | 0.4901 |
| 4 | 0.0176 | 0.4158 |

## Strategies (R²)

| strategy | R² |
|----------|-----|
| full | 0.3427 |
| only_b3 | 0.3303 |
| mid_3 | 0.3303 |
| boost_b3 | 0.3606 |
