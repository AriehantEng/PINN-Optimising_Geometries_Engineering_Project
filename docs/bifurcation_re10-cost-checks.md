## Checks on the cost benchmark

Layout of `bifurcation/runs/parametric_re10_3090_equal_corners`; finite elements as in `bifurcation/cost.py`.

### The equal-accuracy baseline across the range

Coarse mesh 0.08/2, reference 0.002/6, finest 0.001/7. Finite-element errors are against the finest mesh; the PINNs' errors are their stored ones, against the reference mesh.

| split | J coarse | J reference | J finest | coarse vs finest | coarse vs reference | reference vs finest | PINN original vs reference | PINN corner modes vs reference |
|---|---|---|---|---|---|---|---|---|
| 0.35 | 5.459968 | 5.464808 | 5.464821 | -0.0888% | -0.0886% | -0.0002% | +0.206% | -0.578% |
| 0.5 | 4.048909 | 4.051576 | 4.051601 | -0.0664% | -0.0658% | -0.0006% | +0.111% | -0.076% |
| 0.65 | 6.530979 | 6.536078 | 6.536099 | -0.0783% | -0.0780% | -0.0003% | -2.052% | -1.149% |

The reference drop recomputed here matches the one the PINNs were scored against to 0.0e+00 relative.

### Curvature of the objective at the finite-element optimum

s* = 0.48475, J* = 4.035831 (reference mesh).

| h | J(s* - h) | J(s* + h) | (J(s*+h) - 2J* + J(s*-h)) / h² / J* |
|---|---|---|---|
| 0.01 | 4.042661 | 4.042573 | 33.63 |
| 0.02 | 4.063050 | 4.063043 | 33.72 |

Richardson extrapolation: **J''/J = 33.60**. For the run's argmin error δ = +0.00143, (J''/J) δ²/2 predicts a regret of 0.0034%; measured 0.0035%.
