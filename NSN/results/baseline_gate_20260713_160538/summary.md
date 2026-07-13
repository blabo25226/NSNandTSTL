# Baseline Gate Evaluation

- protocol: `full`
- seeds: [0, 1, 7, 42, 123]
- Gate A (core 80%): **FAIL**
- Gate B (sin_plus 60%): **FAIL** (optional)
- Gate C (snap degrade ≤ 1.5): **PASS**
- Gate C median degrade (Gate A symbolic OK): 1.000

## Gate A per target

| target | symbolic OK | finite | rate | degrade median |
|--------|-------------|--------|------|----------------|
| square | 3/5 | 5/5 | 60% | 1.00 |
| product | 3/5 | 5/5 | 60% | 1.00 |
| sum | 1/5 | 5/5 | 20% | 1.00 |
| exp | 5/5 | 5/5 | 100% | 1.00 |
| sin | 2/5 | 5/5 | 40% | 1.00 |

## Gate B

| target | symbolic OK | rate |
|--------|-------------|------|
| sin_plus | 1/5 | 20% |

## Notes

- Gate B optional: sin_plus below 60% does not block TSTL if Gate A+C pass.
