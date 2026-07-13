# SR Phase 1 Results

- **numeric OK** = soft (trained) model holdout MSE within threshold.
- **symbolic OK** = *snapped* closed-form EML expression holdout MSE within
  threshold (i.e. snapping is faithful — the paper's snapping-success claim).

- numeric OK: 0/1
- symbolic OK: 0/1

| target | true | mono | holdout MSE | snapped MSE | degrade | numeric | symbolic | guess |
|--------|------|------|-------------|-------------|---------|---------|----------|-------|
| sin_plus | `sin(x0) + x1` | False | 7.52e-02 | 1.28e-01 | x1.7 | False | False | `sin(x0) + x1` |
