# SR Phase 2 Results

- numeric OK: 2/2
- symbolic OK: 2/2

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| feynman_I29_product | `x0 * x1` | False | 7.20e-04 | True | True | `x0 * x1` |
| feynman_I9_inv_square | `1 / x0^2` | True | 1.94e-04 | True | True | `1 / x0^2` |
