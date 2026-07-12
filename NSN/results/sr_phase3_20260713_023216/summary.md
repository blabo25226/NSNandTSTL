# SR Phase 3 Results

- **numeric OK** = soft (trained) model holdout MSE within threshold.
- **symbolic OK** = *snapped* closed-form EML expression holdout MSE within
  threshold (i.e. snapping is faithful — the paper's snapping-success claim).

- numeric OK: 5/6
- symbolic OK: 5/6

| target | true | mono | holdout MSE | snapped MSE | degrade | numeric | symbolic | guess |
|--------|------|------|-------------|-------------|---------|---------|----------|-------|
| square | `x0^2` | False | 4.77e-04 | 4.77e-04 | x1.0 | True | True | `x0^2` |
| product | `x0 * x1` | False | 1.42e-04 | 1.42e-04 | x1.0 | True | True | `x0 * x1` |
| sum | `x0 + x1` | True | 2.49e-02 | 2.49e-02 | x1.0 | False | False | `x0 + x1` |
| exp | `exp(x0)` | True | 1.23e-04 | 1.23e-04 | x1.0 | True | True | `exp(x0)` |
| sin | `sin(x0)` | False | 6.83e-04 | 6.83e-04 | x1.0 | True | True | `sin(x0)` |
| sin_plus | `sin(x0) + x1` | False | 3.01e-04 | 3.01e-04 | x1.0 | True | True | `sin(x0) + x1` |
