# Head-Capacity Study

Small snapped/soft gap ⇒ the exported closed-form head is faithful (genuine symbolic recovery). Large gap ⇒ the trunk MLP is doing the work.

| target | config | soft MSE | snapped MSE | degrade |
|--------|--------|----------|-------------|---------|
| square | full | 2.57e-01 | 6.13e+00 | x23.8 |
| square | small | 1.06e-03 | 3.04e+00 | x2859.3 |
| square | freeze | 1.58e+00 | 4.58e+00 | x2.9 |
| square | small+parent | 2.03e-03 | 2.71e+01 | x13344.8 |
