# SR Phase 3 Results

- numeric OK: 0/6
- symbolic OK: 0/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 7.74e+00 | False | False | `sin(x0)` |
| product | `x0 * x1` | False | 8.00e-01 | False | False | `sin(x0) + x1` |
| sum | `x0 + x1` | True | 7.26e-01 | False | False | `sin(x0) + x1` |
| exp | `exp(x0)` | True | 1.57e+00 | False | False | `sin(x0)` |
| sin | `sin(x0)` | False | 2.27e+00 | False | False | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 1.88e+10 | False | False | `x0 * x1` |
