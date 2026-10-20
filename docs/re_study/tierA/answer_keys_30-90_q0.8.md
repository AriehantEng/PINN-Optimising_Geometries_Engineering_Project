## Finite-element answer keys for the parametric PINNs

Geometry: parent 3 long, outer walls 4, share q = 0.8. Search mesh 0.02/3, check mesh 0.005/5, reference mesh 0.002/6. J in the case's pressure units (inertial: rho U_c^2; viscous: mu U_c / h).

| case | s_FE | argmin bound ε_s | J_FE (search) | J_FE (check) | J_FE (reference, at s_FE) | L1, L2 (centreline at s_FE) | s_M | s_L | survives refinement |
|---|---|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 0.6651 | 0.0010 | 34.138 | 34.1425 | 34.1427 | 3.9973, 4.4253 | 0.6667 | 0.6608 | yes |
| 30-90_Re10 | 0.6651 | 0.0010 | 3.42534 | 3.42579 | 3.42581 | 3.9973, 4.4253 | 0.6667 | 0.6608 | yes |

### J on the reference mesh

| case | s = 0.51 | s = 0.56 | s = 0.61 | s = 0.66 | s = 0.71 | s = 0.76 | s = 0.81 |
|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 47.6106 | 40.3512 | 35.9078 | 34.1524 | 35.8546 | 43.7066 | 66.3702 |
| 30-90_Re10 | 4.79741 | 4.05972 | 3.60741 | 3.42707 | 3.5948 | 4.37896 | 6.64529 |
