## Finite-element answer keys for the parametric PINNs

Geometry: parent 3 long, outer walls 4, share q = 0.2. Search mesh 0.02/3, check mesh 0.005/5, reference mesh 0.002/6. J in the case's pressure units (inertial: rho U_c^2; viscous: mu U_c / h).

| case | s_FE | argmin bound ε_s | J_FE (search) | J_FE (check) | J_FE (reference, at s_FE) | L1, L2 (centreline at s_FE) | s_M | s_L | survives refinement |
|---|---|---|---|---|---|---|---|---|---|
| 15-15_Re0.1 | 0.3303 | 0.0010 | 33.4014 | 33.4009 | 33.4009 | 4.0204, 4.1113 | 0.3333 | 0.3321 | yes |
| 15-15_Re10 | 0.3297 | 0.0010 | 3.34842 | 3.34831 | 3.34831 | 4.0202, 4.1115 | 0.3333 | 0.3321 | yes |

### J on the reference mesh

| case | s = 0.18 | s = 0.23 | s = 0.28 | s = 0.33 | s = 0.38 | s = 0.43 | s = 0.48 |
|---|---|---|---|---|---|---|---|
| 15-15_Re0.1 | 70.2211 | 44.3499 | 35.4362 | 33.4042 | 35.2755 | 40.33 | 48.9497 |
| 15-15_Re10 | 7.03157 | 4.44244 | 3.55083 | 3.34874 | 3.53845 | 4.04851 | 4.91813 |
