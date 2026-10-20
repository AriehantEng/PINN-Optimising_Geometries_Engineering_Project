## Finite-element answer keys for the parametric PINNs

Geometry: parent 3 long, outer walls 4, share q = 0.5. Search mesh 0.02/3, check mesh 0.005/5, reference mesh 0.002/6. J in the case's pressure units (inertial: rho U_c^2; viscous: mu U_c / h).

| case | s_FE | argmin bound ε_s | J_FE (search) | J_FE (check) | J_FE (reference, at s_FE) | L1, L2 (centreline at s_FE) | s_M | s_L | survives refinement |
|---|---|---|---|---|---|---|---|---|---|
| 0-120_Re0.1 | 0.4534 | 0.0010 | 35.7578 | 35.7728 | 35.774 | 3.0533, 4.9467 | 0.5000 | 0.4709 | yes |
| 0-120_Re10 | 0.4528 | 0.0010 | 3.62074 | 3.62265 | 3.62273 | 3.0523, 4.9477 | 0.5000 | 0.4709 | yes |

### J on the reference mesh

| case | s = 0.32 | s = 0.37 | s = 0.42 | s = 0.47 | s = 0.52 | s = 0.57 | s = 0.62 |
|---|---|---|---|---|---|---|---|
| 0-120_Re0.1 | 46.5315 | 39.4898 | 36.3033 | 35.9481 | 38.2936 | 43.9677 | 54.6628 |
| 0-120_Re10 | 4.70435 | 3.99687 | 3.67633 | 3.64 | 3.87492 | 4.44426 | 5.51794 |
