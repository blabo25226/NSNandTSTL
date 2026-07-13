# SR Phase 3 Results

- numeric OK: 0/6
- symbolic OK: 0/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 2.54e+01 | False | False | `sin(x0)` |
| product | `x0 * x1` | False | 2.03e+01 | False | False | `sin(x0)` |
| sum | `x0 + x1` | True | 1.37e+01 | False | False | `sin(x0) + x1` |
| exp | `exp(x0)` | True | 1.41e+00 | False | False | `sin(x0)` |
| sin | `sin(x0)` | False | 2.41e+15 | False | False | `x0^2` |
| sin_plus | `sin(x0) + x1` | False | 1.70e+00 | False | False | `sin(x0) + x1` |
