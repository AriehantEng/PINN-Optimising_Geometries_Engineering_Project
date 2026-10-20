## Finite-element answer keys for the parametric PINNs

Geometry: parent 3 long, outer walls 4, share q = 0.2. Search mesh 0.02/3, check mesh 0.005/5, reference mesh 0.002/6. J in the case's pressure units (inertial: rho U_c^2; viscous: mu U_c / h).

| case | s_FE | argmin bound ε_s | J_FE (search) | J_FE (check) | J_FE (reference, at s_FE) | L1, L2 (centreline at s_FE) | s_M | s_L | survives refinement |
|---|---|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 0.3067 | 0.0010 | 33.583 | 33.5916 | 33.5922 | 3.3766, 5.0461 | 0.3333 | 0.3116 | yes |
| 30-90_Re10 | 0.3058 | 0.0010 | 3.41196 | 3.41326 | 3.41331 | 3.3749, 5.0477 | 0.3333 | 0.3116 | yes |

### J on the reference mesh

| case | s = 0.16 | s = 0.21 | s = 0.26 | s = 0.31 | s = 0.36 | s = 0.41 | s = 0.46 |
|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 64.3239 | 42.2905 | 35.0803 | 33.6104 | 35.3622 | 39.7565 | 47.1203 |
| 30-90_Re10 | 6.48276 | 4.27989 | 3.56025 | 3.41542 | 3.59366 | 4.03735 | 4.77971 |
