# TSTL MLP layer profile

- seed: 2
- hidden layers: 5
- S_base (R²): -0.0931
- S_full (R²): 0.3957
- best layer k: 0 (C=1.186)
- mid-k indices: [1, 2, 3]

## C(k)

| k | S_k | C(k) |
|---|-----|------|
| 0 | 0.4867 | 1.1860 |
| 1 | 0.2936 | 0.7910 |
| 2 | 0.2688 | 0.7405 |
| 3 | 0.2248 | 0.6503 |
| 4 | 0.1118 | 0.4191 |

## Strategies (R²)

| strategy | R² |
|----------|-----|
| full | 0.3957 |
| only_b3 | 0.5763 |
| mid_3 | 0.3249 |
| boost_b3 | 0.4177 |
