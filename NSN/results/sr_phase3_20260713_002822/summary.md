# SR Phase 3 Results

- numeric OK: 1/6
- symbolic OK: 1/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 2.24e+01 | False | False | `sin(x0)` |
| product | `x0 * x1` | False | 3.22e-02 | True | True | `x0 * x1` |
| sum | `x0 + x1` | True | 1.28e-01 | False | False | `sin(x0) + x1` |
| exp | `exp(x0)` | True | 9.23e-01 | False | False | `sin(x0)` |
| sin | `sin(x0)` | False | 2.58e+00 | False | False | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 9.40e+00 | False | False | `sin(x0) + x1` |
