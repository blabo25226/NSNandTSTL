# SR Phase 3 Results

- numeric OK: 0/6
- symbolic OK: 0/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 1.01e+01 | False | False | `sin(x0)` |
| product | `x0 * x1` | False | 8.82e-01 | False | False | `sin(x0)` |
| sum | `x0 + x1` | True | 2.32e+00 | False | False | `x0^2` |
| exp | `exp(x0)` | True | 2.10e-01 | False | False | `exp(x0)` |
| sin | `sin(x0)` | False | 6.93e-02 | False | False | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 2.18e+00 | False | False | `sin(x0) + x1` |
