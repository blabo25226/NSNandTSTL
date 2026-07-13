# TSTL MLP layer profile

- seed: 1
- hidden layers: 5
- S_base (R²): -0.3866
- S_full (R²): 0.3886
- best layer k: 0 (C=1.151)
- mid-k indices: [1, 2, 3]

## C(k)

| k | S_k | C(k) |
|---|-----|------|
| 0 | 0.5059 | 1.1514 |
| 1 | 0.4930 | 1.1347 |
| 2 | 0.2751 | 0.8536 |
| 3 | 0.1501 | 0.6924 |
| 4 | 0.3849 | 0.9953 |

## Strategies (R²)

| strategy | R² |
|----------|-----|
| full | 0.3886 |
| only_b3 | 0.5092 |
| mid_3 | 0.2362 |
| boost_b3 | 0.3425 |
