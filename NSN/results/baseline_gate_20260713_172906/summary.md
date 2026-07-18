# Baseline Gate Evaluation

- protocol: `trunk_first`
- seeds: [0, 1, 7, 42, 123]
- Gate A (core 80%): **FAIL**
- Gate B (sin_plus 60%): **FAIL** (optional)
- Gate C (snap degrade ≤ 1.5): **FAIL**

## Gate A per target

| target | symbolic OK | finite | rate | degrade median |
|--------|-------------|--------|------|----------------|
| square | 0/5 | 5/5 | 0% | — |
| product | 0/5 | 5/5 | 0% | — |
| sum | 0/5 | 5/5 | 0% | — |
| sin | 0/5 | 5/5 | 0% | — |

## Gate B

| target | symbolic OK | rate |
|--------|-------------|------|
| sin_plus | 0/5 | 0% |

## Notes

- Gate B optional: sin_plus below 60% does not block TSTL if Gate A+C pass.
