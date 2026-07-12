# SR Phase 3 Results

- numeric OK: 1/6
- symbolic OK: 1/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 2.40e-01 | False | False | `x0^2` |
| product | `x0 * x1` | False | 5.00e-03 | True | True | `x0 * x1` |
| sum | `x0 + x1` | True | 3.52e-01 | False | False | `sin(x0) + x1` |
| exp | `exp(x0)` | True | 1.00e+00 | False | False | `sin(x0)` |
| sin | `sin(x0)` | False | 2.59e+00 | False | False | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 4.75e-01 | False | False | `sin(x0)` |
