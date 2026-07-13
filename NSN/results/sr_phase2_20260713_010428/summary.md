# SR Phase 2 Results

- numeric OK: 0/2
- symbolic OK: 0/2

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| feynman_I29_product | `x0 * x1` | False | 8.82e-01 | False | False | `sin(x0)` |
| feynman_I9_inv_square | `1 / x0^2` | True | 4.54e+00 | False | False | `sin(x0)` |
