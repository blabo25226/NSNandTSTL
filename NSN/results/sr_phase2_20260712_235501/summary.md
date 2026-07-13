# SR Phase 2 Results

- numeric OK: 1/2
- symbolic OK: 1/2

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| feynman_I29_product | `x0 * x1` | False | 5.00e-03 | True | True | `x0 * x1` |
| feynman_I9_inv_square | `1 / x0^2` | True | 4.63e+00 | False | False | `sin(x0)` |
