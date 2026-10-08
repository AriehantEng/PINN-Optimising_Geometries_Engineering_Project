## Baselines for Network 3, and the length behind s_L

### Held-out angle pairs (never seen by Network 3)

Regret: J at the proposed split over the finite-element optimum, minus 1, on the design study's search mesh (its noise is about 1e-4).

| angles | share | optimum s* | Murray s_M | regret | length-corrected s_L | regret | Network 3 | regret |
|---|---|---|---|---|---|---|---|---|
| 45/75 | 0.20 | 0.3261 | 0.3333 | +0.115% | 0.3258 | +0.001% | 0.3241 | +0.009% |
| 45/75 | 0.50 | 0.4985 | 0.5000 | +0.011% | 0.4960 | +0.012% | 0.4955 | +0.015% |
| 45/75 | 0.80 | 0.6677 | 0.6667 | +0.003% | 0.6671 | +0.004% | 0.6692 | +0.005% |
| 15/105 | 0.20 | 0.3152 | 0.3333 | +0.696% | 0.3195 | +0.036% | 0.3118 | +0.028% |
| 15/105 | 0.50 | 0.4878 | 0.5000 | +0.342% | 0.4888 | +0.010% | 0.4838 | +0.015% |
| 15/105 | 0.80 | 0.6618 | 0.6667 | +0.063% | 0.6608 | +0.002% | 0.6603 | +0.005% |
| 75/105 | 0.20 | 0.3330 | 0.3333 | +0.026% | 0.3333 | +0.026% | 0.3305 | +0.023% |
| 75/105 | 0.50 | 0.5006 | 0.5000 | +0.005% | 0.5000 | +0.005% | 0.4992 | +0.007% |
| 75/105 | 0.80 | 0.6673 | 0.6667 | +0.003% | 0.6667 | +0.004% | 0.6649 | +0.017% |
| 45/135 | 0.20 | 0.3347 | 0.3333 | +0.087% | 0.3333 | +0.087% | 0.3280 | +0.075% |
| 45/135 | 0.50 | 0.5000 | 0.5000 | +0.000% | 0.5000 | +0.030% | 0.5005 | +0.007% |
| 45/135 | 0.80 | 0.6681 | 0.6667 | -0.001% | 0.6667 | +0.005% | 0.6623 | +0.062% |
| 30/30 | 0.20 | 0.3313 | 0.3333 | +0.009% | 0.3320 | +0.001% | 0.3313 | -0.000% |
| 30/30 | 0.50 | 0.5000 | 0.5000 | +0.000% | 0.5000 | +0.000% | 0.5000 | +0.000% |
| 30/30 | 0.80 | 0.6684 | 0.6667 | +0.008% | 0.6680 | +0.001% | 0.6687 | -0.000% |
| 0/90 | 0.20 | 0.3207 | 0.3333 | +0.354% | 0.3240 | +0.030% | 0.3219 | +0.015% |
| 0/90 | 0.50 | 0.4899 | 0.5000 | +0.189% | 0.4921 | +0.010% | 0.4904 | +0.003% |
| 0/90 | 0.80 | 0.6621 | 0.6667 | +0.048% | 0.6620 | +0.001% | 0.6630 | +0.003% |

| proposed split | all (mean, max) | a daughter beyond 90 degrees (mean, max) | no daughter beyond 90 degrees (mean, max) |
|---|---|---|---|
| Murray s_M | 0.109%, 0.696% | 0.136%, 0.696% | 0.082%, 0.354% |
| length-corrected s_L | 0.015%, 0.087% | 0.023%, 0.087% | 0.007%, 0.030% |
| Network 3 | 0.016%, 0.075% | 0.027%, 0.075% | 0.006%, 0.015% |

Network 3's splits are those of its results file, to four decimals (at full precision its regret was 0.016% mean, 0.077% max). Negative regrets (search noise) count as 0. Case counts: all 18, a daughter beyond 90 degrees 9, no daughter beyond 90 degrees 9.

### How the length is counted

Share of the squared departure from s_M that s_L removes, the largest residual |s* - s_L|, and the residuals beyond their location bound (how many widen daughter 1). Centreline = strip area / width = mean of outer and inner wall; closed = the length over which the daughter has both walls.

| layout | centreline: removed, largest, beyond (+) | inner wall: removed, largest, beyond (+) | closed: removed, largest, beyond (+) |
|---|---|---|---|
| 15/15 | 59%, 0.0016, 5/11 (2) | 91%, 0.0009, 0/11 (0) | 5%, 0.0023, 7/11 (3) |
| 0/30 | 78%, 0.0018, 5/11 (3) | 45%, 0.0030, 9/11 (9) | 78%, 0.0018, 5/11 (3) |
| 20/40 | 84%, 0.0016, 2/11 (2) | 12%, 0.0025, 9/11 (9) | 52%, 0.0019, 10/11 (4) |
| 0/60 | 92%, 0.0021, 7/11 (3) | 45%, 0.0043, 11/11 (11) | 94%, 0.0018, 7/11 (3) |
| 45/45 | 96%, 0.0006, 0/11 (0) | -77%, 0.0033, 8/11 (4) | 36%, 0.0018, 8/11 (4) |
| 10/80 | 95%, 0.0019, 6/11 (3) | 24%, 0.0078, 11/11 (11) | 95%, 0.0025, 5/11 (0) |
| 60/60 | 85%, 0.0021, 5/11 (2) | -138%, 0.0061, 10/11 (5) | 62%, 0.0030, 9/11 (5) |
| 30/90 | 96%, 0.0025, 9/11 (5) | 21%, 0.0094, 10/11 (10) | 91%, 0.0037, 8/11 (1) |
| 0/120 | 91%, 0.0080, 9/11 (0) | 78%, 0.0107, 11/11 (11) | 97%, 0.0048, 9/11 (0) |
| 75/75 | 96%, 0.0026, 6/11 (3) | 7%, 0.0108, 10/11 (5) | 80%, 0.0056, 8/11 (4) |
| 30/120 | 89%, 0.0144, 9/11 (0) | 78%, 0.0185, 11/11 (11) | 95%, 0.0094, 11/11 (0) |
| 0/150 | 78%, 0.0353, 11/11 (0) | 92%, 0.0198, 11/11 (11) | 96%, 0.0128, 11/11 (0) |

Pooled over the twelve layouts: centreline 82%, inner wall 84%, closed 96%.

### Is the optimum's drop above the friction law?

Smallest J*/J_friction - 1 over the shares, with the friction law evaluated at s* using each length (parent included). Junction and entrance losses are positive, so a negative value means those lengths over-count the friction.

| layout | outer wall (L_d) | centreline | inner wall | closed |
|---|---|---|---|---|
| 15/15 | +2.6% | +1.3% | +0.1% | +2.6% |
| 0/30 | +1.0% | +1.3% | +0.3% | +1.9% |
| 20/40 | +3.2% | +1.4% | -1.5% | +3.2% |
| 0/60 | +0.8% | +1.4% | -0.4% | +2.7% |
| 45/45 | +4.4% | +1.1% | -2.8% | +4.4% |
| 10/80 | +2.0% | +1.0% | -3.0% | +2.3% |
| 60/60 | +5.3% | +0.0% | -5.8% | +5.3% |
| 30/90 | +3.2% | -1.2% | -8.2% | +4.1% |
| 0/120 | -2.4% | -2.0% | -6.6% | +5.7% |
| 75/75 | +5.3% | -4.9% | -13.5% | +6.3% |
| 30/120 | -8.7% | -13.5% | -21.6% | +4.2% |
| 0/150 | -23.7% | -20.5% | -20.8% | +11.2% |

### Loss coefficients against closed lengths

K_i = (path drop - parent friction - daughter friction) / (1/2 U_i^2), measured at each optimum. Model: the friction law along the closed lengths plus K_i (1/2) U_i^2 with that optimum's own K, minimised over s; 'lengths only' is the same with K = 0.

| layout | negative K, centreline friction | negative K, closed friction | K1, K2 at q = 0.5 (closed) | largest |s* - s|: lengths only | with K | beyond bound, with K |
|---|---|---|---|---|---|---|
| 15/15 | 0/22 | 0/22 | 0.85, 0.85 | 0.0021 | 0.0019 | 7/11 |
| 0/30 | 0/22 | 0/22 | 0.97, 1.06 | 0.0027 | 0.0027 | 4/11 |
| 20/40 | 0/22 | 0/22 | 0.79, 1.28 | 0.0016 | 0.0013 | 4/11 |
| 0/60 | 3/22 | 0/22 | 1.37, 1.69 | 0.0034 | 0.0033 | 7/11 |
| 45/45 | 0/22 | 0/22 | 1.37, 1.37 | 0.0018 | 0.0010 | 0/11 |
| 10/80 | 3/22 | 0/22 | 1.71, 2.16 | 0.0039 | 0.0041 | 8/11 |
| 60/60 | 2/22 | 0/22 | 1.67, 1.67 | 0.0015 | 0.0022 | 3/11 |
| 30/90 | 9/22 | 0/22 | 2.03, 2.36 | 0.0048 | 0.0053 | 9/11 |
| 0/120 | 8/22 | 0/22 | 2.41, 3.43 | 0.0061 | 0.0053 | 11/11 |
| 75/75 | 12/22 | 0/22 | 1.98, 1.98 | 0.0049 | 0.0058 | 6/11 |
| 30/120 | 18/22 | 0/22 | 3.52, 3.30 | 0.0124 | 0.0106 | 11/11 |
| 0/150 | 20/22 | 0/22 | 3.77, 4.55 | 0.0142 | 0.0096 | 11/11 |
| 90/90 | 0/22 | 0/22 | 2.29, 2.29 | - | - | - |
| 60/120 | 0/22 | 0/22 | 1.47, 4.42 | - | - | - |
| 30/150 | 0/22 | 0/22 | 0.92, 17.30 | - | - | - |

### Regret over the map, below 180 degrees

J at the proposed split over the optimum, minus 1, on the search mesh (noise about 1e-4); median and maximum.

| proposed split | no daughter beyond 90 degrees (99) | a daughter turned back (33) | all below 180 degrees (132) |
|---|---|---|---|
| Murray s_M | 0.0141%, 0.423% | 1.2257%, 8.926% | 0.0309%, 8.926% |
| s_L, centreline | 0.0029%, 0.023% | 0.1346%, 2.661% | 0.0051%, 2.661% |
| s_L, closed | 0.0065%, 0.055% | 0.0871%, 0.270% | 0.0093%, 0.270% |

Largest regret by layout (Murray, centreline s_L, closed s_L): 60/60: 0.073%, 0.006%, 0.018%; 30/90: 0.412%, 0.023%, 0.037%; 0/120: 1.226%, 0.141%, 0.071%; 15/15: 0.014%, 0.006%, 0.013%; 0/30: 0.043%, 0.011%, 0.011%; 20/40: 0.033%, 0.005%, 0.010%; 0/60: 0.132%, 0.012%, 0.010%; 45/45: 0.025%, 0.002%, 0.012%; 10/80: 0.240%, 0.012%, 0.020%; 75/75: 0.423%, 0.012%, 0.055%; 30/120: 3.799%, 0.587%, 0.158%; 0/150: 8.926%, 2.661%, 0.270%
