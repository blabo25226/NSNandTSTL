# Trunk Interpretability Study

Two-layer white-box explanation: the linear readout recovers `z_j = w_j^T x + b_j`
(exact for a linear trunk, else distilled with per-component R^2), and the head
export is composed into a single closed form `yhat(x)`.

## product — `x0 * x1` (trunk=mlp)

- readout R^2 (per z): [0.9329, 0.9442, 0.9457, 0.9416] (exact=False)
- soft MSE: 7.162e-04 | snapped MSE: 7.162e-04
- head:     `yhat = Re[eml(eml(-0.084606, (0.0691231*z0 + -0.0646047*z1 + 0.0727423*z2 + -0.0720768*z3)), eml(-0.020813, (-0.0679485*z0 + 0.048867*z1 + -0.0568982*z2 + 0.055011*z3)))]`
- composed: `yhat = Re[eml(eml(-0.084606, (0.0691231*(-1.5327*x0 + -1.37673*x1 + 6.3299) + -0.0646047*(1.78944*x0 + 1.66011*x1 + -8.14501) + 0.0727423*(-2.07304*x0 + -1.91868*x1 + 9.15355) + -0.0720768*(1.96105*x0 + 1.86753*x1 + -8.6754))), eml(-0.020813, (-0.0679485*(-1.5327*x0 + -1.37673*x1 + 6.3299) + 0.048867*(1.78944*x0 + 1.66011*x1 + -8.14501) + -0.0568982*(-2.07304*x0 + -1.91868*x1 + 9.15355) + 0.055011*(1.96105*x0 + 1.86753*x1 + -8.6754))))]`
  - z0 <- x0 (1.6151), x1 (1.4954)
  - z1 <- x0 (1.8702), x1 (1.7716)
  - z2 <- x0 (2.1678), x1 (2.047)
  - z3 <- x0 (2.0527), x1 (1.9944)

