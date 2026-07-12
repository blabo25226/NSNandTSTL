# SR Phase 3 Results

- numeric OK: 0/6
- symbolic OK: 0/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 4.42e+00 | False | False | `sin(x0)` |
| product | `x0 * x1` | False | 6.10e+00 | False | False | `sin(x0)` |
| sum | `x0 + x1` | True | 2.99e+00 | False | False | `x0 + x1` |
| exp | `exp(x0)` | True | 8.72e-01 | False | False | `sin(x0)` |
| sin | `sin(x0)` | False | 2.54e+00 | False | False | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 6.90e+00 | False | False | `x0 * x1` |
