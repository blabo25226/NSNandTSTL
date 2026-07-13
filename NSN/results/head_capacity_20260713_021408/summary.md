# Head-Capacity Study

Small snapped/soft gap ⇒ the exported closed-form head is faithful (genuine symbolic recovery). Large gap ⇒ the trunk MLP is doing the work.

| target | config | soft MSE | snapped MSE | degrade |
|--------|--------|----------|-------------|---------|
| square | full | 1.13e-01 | 1.88e+00 | x16.7 |
| square | small | 2.75e-04 | 9.21e-04 | x3.3 |
| square | freeze | 8.56e-01 | 2.39e+02 | x279.5 |
| square | small+parent | 3.21e-04 | 2.36e+01 | x73379.5 |
| sin | full | 1.79e+00 | 1.86e+00 | x1.0 |
| sin | small | 8.26e-05 | 2.11e+00 | x25522.9 |
| sin | freeze | 1.79e+00 | 1.80e+00 | x1.0 |
| sin | small+parent | 1.26e-04 | 1.14e+01 | x90718.0 |
| exp | full | 8.77e-01 | 9.23e-01 | x1.1 |
| exp | small | 8.62e-05 | 1.96e-01 | x2279.5 |
| exp | freeze | 3.18e-04 | 4.02e+00 | x12672.4 |
| exp | small+parent | 9.14e-05 | 1.68e+01 | x183932.3 |
