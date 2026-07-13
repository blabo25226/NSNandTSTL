# SR Phase 3 Results

- numeric OK: 6/6
- symbolic OK: 6/6

| target | true | mono | holdout MSE | numeric | symbolic | guess |
|--------|------|------|-------------|---------|----------|-------|
| square | `x0^2` | False | 2.00e-04 | True | True | `x0^2` |
| product | `x0 * x1` | False | 7.20e-04 | True | True | `x0 * x1` |
| sum | `x0 + x1` | True | 2.24e-03 | True | True | `x0 + x1` |
| exp | `exp(x0)` | True | 7.69e-05 | True | True | `exp(x0)` |
| sin | `sin(x0)` | False | 6.96e-05 | True | True | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 3.68e-02 | True | True | `sin(x0) + x1` |
