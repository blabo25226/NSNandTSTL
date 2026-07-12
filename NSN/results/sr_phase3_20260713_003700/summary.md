# SR Phase 3 Results

- numeric OK: 0/6
- symbolic OK: 0/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 3.19e+00 | False | False | `sin(x0)` |
| product | `x0 * x1` | False | 2.48e+00 | False | False | `sin(x0)` |
| sum | `x0 + x1` | True | 1.51e+01 | False | False | `sin(x0) + x1` |
| exp | `exp(x0)` | True | 1.34e+00 | False | False | `sin(x0)` |
| sin | `sin(x0)` | False | 1.00e+00 | False | False | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 1.00e+01 | False | False | `sin(x0) + x1` |
