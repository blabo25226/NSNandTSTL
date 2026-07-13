# TSTL MLP layer profile

- seed: 0
- hidden layers: 5
- S_base (R²): -0.2138
- S_full (R²): 0.3846
- best layer k: 0 (C=1.204)
- mid-k indices: [1, 2, 3]

## C(k)

| k | S_k | C(k) |
|---|-----|------|
| 0 | 0.5068 | 1.2043 |
| 1 | 0.4024 | 1.0298 |
| 2 | 0.3408 | 0.9268 |
| 3 | 0.3038 | 0.8649 |
| 4 | 0.1110 | 0.5428 |

## Strategies (R²)

| strategy | R² |
|----------|-----|
| full | 0.3846 |
| only_b3 | 0.5427 |
| mid_3 | 0.3033 |
| boost_b3 | 0.5164 |
