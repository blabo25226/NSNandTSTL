# SR Phase 3 Results

- **numeric OK** = soft (trained) model holdout MSE within threshold.
- **symbolic OK** = *snapped* closed-form EML expression holdout MSE within
  threshold (i.e. snapping is faithful — the paper's snapping-success claim).

- numeric OK: 5/6
- symbolic OK: 5/6

| target | true | mono | holdout MSE | snapped MSE | degrade | numeric | symbolic | guess |
|--------|------|------|-------------|-------------|---------|---------|----------|-------|
| square | `x0^2` | False | 1.88e-04 | 1.88e-04 | x1.0 | True | True | `x0^2` |
| product | `x0 * x1` | False | 1.30e-04 | 1.30e-04 | x1.0 | True | True | `x0 * x1` |
| sum | `x0 + x1` | True | 3.09e-04 | 3.09e-04 | x1.0 | True | True | `x0 + x1` |
| exp | `exp(x0)` | True | 1.25e-04 | 1.25e-04 | x1.0 | True | True | `exp(x0)` |
| sin | `sin(x0)` | False | 6.79e-04 | 6.79e-04 | x1.0 | True | True | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 2.61e+00 | 2.61e+00 | x1.0 | False | False | `sin(x0) + x1` |
