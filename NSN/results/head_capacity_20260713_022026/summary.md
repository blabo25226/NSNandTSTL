# Head-Capacity Study

Small snapped/soft gap ⇒ the exported closed-form head is faithful (genuine symbolic recovery). Large gap ⇒ the trunk MLP is doing the work.

| target | config | soft MSE | snapped MSE | degrade |
|--------|--------|----------|-------------|---------|
| square | full | 1.66e+00 | 1.66e+00 | x1.0 |
| square | small | 2.60e-04 | 2.60e-04 | x1.0 |
| square | freeze | 6.47e+01 | 6.47e+01 | x1.0 |
| square | small+parent | 1.99e+01 | 1.99e+01 | x1.0 |
| sin | full | 1.84e+00 | 1.84e+00 | x1.0 |
| sin | small | 2.14e-03 | 2.14e-03 | x1.0 |
| sin | freeze | 1.80e+00 | 1.80e+00 | x1.0 |
| sin | small+parent | 8.87e+00 | 8.87e+00 | x1.0 |
| exp | full | 9.22e-01 | 9.22e-01 | x1.0 |
| exp | small | 9.49e-05 | 9.49e-05 | x1.0 |
| exp | freeze | 3.42e+00 | 3.42e+00 | x1.0 |
| exp | small+parent | 1.04e+01 | 1.04e+01 | x1.0 |
