# TSTL MLP layer profile

- seed: 42
- hidden layers: 5
- S_base (R²): -0.0290
- S_full (R²): 0.4068
- best layer k: 0 (C=1.076)
- mid-k indices: [1, 2, 3]

## C(k)

| k | S_k | C(k) |
|---|-----|------|
| 0 | 0.4400 | 1.0762 |
| 1 | 0.3468 | 0.8623 |
| 2 | 0.3980 | 0.9798 |
| 3 | 0.2214 | 0.5746 |
| 4 | 0.1506 | 0.4121 |

## Strategies (R²)

| strategy | R² |
|----------|-----|
| full | 0.4068 |
| only_b3 | 0.4469 |
| mid_3 | 0.2732 |
| boost_b3 | 0.3959 |
