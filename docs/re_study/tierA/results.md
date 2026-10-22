# Tier A: new shares and a symmetric control

Twelve parametric PINN runs, one seed each (21), at Re_h = 0.1 and 10: 30/90 at q = 0.2, 0.35, 0.65 and 0.8; 15/15 at q = 0.2 (the null test); 0/120 at q = 0.5. Rules and pass marks were committed before any screening solve or run (docs/re_study/tierA/preregistration.md). P1: the PINN's optimum within ±0.002 of the reference optimum (Brent on the reference mesh). P2: the reference optimum within 0.003 of the length-corrected split s_L, where no daughter turns beyond 90° (30/90 and 15/15). Both tolerances are the paper's, unchanged. Each trained range is s_L ± 0.15; all twelve cases passed the screening for an interior minimum with daughters 4 long (docs/re_study/tierA/screening.md).

Of the 12 runs reported, 2 pass P1. Of the 10 in P2's scope, 5 pass P2. Verdicts are on the unrounded values; nothing was rerun.

## What Tier A shows

- P1 passes only at q = 0.5: 0/120 at both Reynolds numbers, to within 0.0001. Every run at another share fails it, at both Reynolds numbers, including the symmetric 15/15 null test.
- At Re_h = 10 the misses are moderate and grow with the share's distance from 0.5: −0.0049 at q = 0.35, +0.0020 at 0.65 (failing by 0.00002), −0.0064 at 0.8, −0.0078 at 0.2, and −0.0068 for 15/15 at 0.2. The regret stays below 0.13%, and the pressure-drop error within ±2.6% over each off-centre range (±3.2% for 0/120).
- At Re_h = 0.1 every off-centre run fails, by +0.006 to +0.156 (regret up to 40%; the PINN's optimum always lies above the reference's). Every Tier A run there, 0/120 included, ends at a held-out loss of 1.1 to 22, against 0.12 to 0.16 for the three earlier 30/90 runs at q = 0.5, and with largest drop errors of 8 to 30%; 0/120 (1.1, −8.1% at its range's top) still placed its optimum to 0.0001, because its error was flat near the optimum.
- The 15/15 null test misses by as much as 30/90 at the same share (−0.0068 against −0.0078 at Re_h = 10), so the miss belongs to the share, not to the layout's length asymmetry.
- Confound: every off-centre case also had a smaller corner-mode cutoff (0.10 to 0.35, by the pre-registered rule, because its narrow daughter is narrow) than 0/120 (0.40) or the earlier runs (0.5). Tier A cannot separate the share from the cutoff; a run at an off-centre share with the cutoff held at 0.40 or more would, where the geometry admits it (it does not at q = 0.2 or 0.8 with this range).
- P2, a property of the reference solution: s_L lies within 0.003 of the optimum at q = 0.35 and 0.65 at Re_h = 0.1, q = 0.35 at 10, and for 15/15 at both; it does not at q = 0.2 and 0.8 for 30/90 (0.004 to 0.006), nor at q = 0.65 at Re_h = 10 (0.0031). The length correction loses accuracy as the share moves from 0.5 in a layout with unequal lengths.

## 1. Every run against the pass marks

| Layout | q | Re_h | PINN s* | Reference s* | Argmin error | P1 | s_L | s*_ref − s_L | P2 | Regret |
|---|---|---|---|---|---|---|---|---|---|---|
| 30/90 | 0.2 | 0.1 | 0.4616 | 0.3060 | +0.1556 | FAIL | 0.3116 | -0.0056 | FAIL | 40.2721% |
| 30/90 | 0.2 | 10 | 0.2978 | 0.3056 | -0.0078 | FAIL | 0.3116 | -0.0060 | FAIL | 0.1257% |
| 30/90 | 0.35 | 0.1 | 0.4394 | 0.4012 | +0.0382 | FAIL | 0.4039 | -0.0027 | pass | 2.4696% |
| 30/90 | 0.35 | 10 | 0.3963 | 0.4012 | -0.0049 | FAIL | 0.4039 | -0.0027 | pass | 0.0391% |
| 30/90 | 0.65 | 0.1 | 0.5750 | 0.5680 | +0.0070 | FAIL | 0.5655 | +0.0025 | pass | 0.0823% |
| 30/90 | 0.65 | 10 | 0.5707 | 0.5686 | +0.0020 | FAIL | 0.5655 | +0.0031 | FAIL | 0.0058% |
| 30/90 | 0.8 | 0.1 | 0.6713 | 0.6649 | +0.0064 | FAIL | 0.6608 | +0.0041 | FAIL | 0.0916% |
| 30/90 | 0.8 | 10 | 0.6590 | 0.6653 | -0.0064 | FAIL | 0.6608 | +0.0045 | FAIL | 0.0750% |
| 15/15 | 0.2 | 0.1 | 0.3859 | 0.3303 | +0.0557 | FAIL | 0.3321 | -0.0018 | pass | 6.4347% |
| 15/15 | 0.2 | 10 | 0.3233 | 0.3300 | -0.0068 | FAIL | 0.3321 | -0.0021 | pass | 0.0966% |
| 0/120 | 0.5 | 0.1 | 0.4527 | 0.4526 | +0.0001 | pass | 0.4709 | -0.0183 | — | -0.0000% |
| 0/120 | 0.5 | 10 | 0.4527 | 0.4527 | +0.0000 | pass | 0.4709 | -0.0181 | — | -0.0000% |

*P1 |argmin error| ≤ 0.002; P2 |s*_ref − s_L| ≤ 0.003, not applied to 0/120 (a daughter turned back). Regret = J_ref(s*_PINN)/J_ref(s*_ref) − 1.*

## 2. Accuracy and training

| Layout | q | Re_h | Corner cutoff | Drop error: low end / middle / high end | Largest |drop error| | u error at s* | Held-out loss | Min | Converged |
|---|---|---|---|---|---|---|---|---|---|
| 30/90 | 0.2 | 0.1 | 0.1 | -22.82% / +16.79% / -29.96% | 29.96% | 6.72e-02 | 22 | 163 | yes |
| 30/90 | 0.2 | 10 | 0.1 | -2.54% / +0.74% / -1.67% | 2.54% | 1.39e-02 | 0.0159 | 166 | yes |
| 30/90 | 0.35 | 0.1 | 0.3 | -23.81% / +3.98% / -19.52% | 23.81% | 2.10e-02 | 6.03 | 180 | no |
| 30/90 | 0.35 | 10 | 0.3 | +0.13% / +0.49% / -1.98% | 1.98% | 1.39e-02 | 0.0136 | 185 | yes |
| 30/90 | 0.65 | 0.1 | 0.35 | -5.09% / -0.78% / -8.28% | 8.28% | 7.71e-03 | 1.49 | 160 | no |
| 30/90 | 0.65 | 10 | 0.35 | +0.01% / +0.03% / -1.95% | 1.95% | 1.17e-02 | 0.0137 | 163 | yes |
| 30/90 | 0.8 | 0.1 | 0.15 | -25.54% / +3.92% / -3.62% | 25.54% | 1.32e-02 | 15 | 191 | no |
| 30/90 | 0.8 | 10 | 0.15 | -0.25% / +0.43% / -2.30% | 2.30% | 9.38e-03 | 0.019 | 166 | yes |
| 15/15 | 0.2 | 0.1 | 0.15 | -26.49% / +7.74% / -25.99% | 26.49% | 2.34e-02 | 20.8 | 193 | yes |
| 15/15 | 0.2 | 10 | 0.15 | -2.60% / +0.24% / -1.88% | 2.60% | 7.09e-03 | 0.0115 | 184 | no |
| 0/120 | 0.5 | 0.1 | 0.4 | -5.18% / +0.31% / -8.12% | 8.12% | 1.25e-02 | 1.11 | 170 | yes |
| 0/120 | 0.5 | 10 | 0.4 | +1.14% / -0.18% / -3.19% | 3.19% | 1.17e-02 | 0.00535 | 153 | yes |

*Drop error = J_PINN/J_ref − 1 at the seven evaluation splits. Held-out loss is a diagnostic only, in each run's pressure units (μU_c/h at Re_h 0.1, ρU_c² at 10).*

### Baseline: the earlier 30/90 runs at q = 0.5

Range [0.35, 0.65], corner cutoff 0.5 (the earlier protocol; not part of Tier A).

| Re_h | Seed | Argmin error | Regret | u error at s* | Held-out loss |
|---|---|---|---|---|---|
| 0.1 | 21 | +0.0026 | 0.0115% | 2.34e-03 | 0.119 |
| 0.1 | 22 | +0.0058 | 0.0563% | 2.73e-03 | 0.162 |
| 0.1 | 23 | +0.0018 | 0.0058% | 3.18e-03 | 0.15 |
| 10 | 21 | +0.0014 | 0.0035% | 1.52e-03 | 0.000915 |
| 10 | 22 | +0.0004 | 0.0002% | 2.45e-03 | 0.000931 |
| 10 | 23 | +0.0008 | 0.0011% | 1.84e-03 | 0.000915 |

## 3. The argmin error against the share

![Argmin error of every Tier A run against daughter 1's share, with the earlier q = 0.5 30/90 runs (open circles, earlier protocol). The grey band is P1's ±0.002. Each panel has its own vertical scale; Re_h = 0.1's is symmetric-logarithmic.](fig_tierA_argmin.png)

*Argmin error of every Tier A run against daughter 1's share, with the earlier q = 0.5 30/90 runs (open circles, earlier protocol). The grey band is P1's ±0.002. Each panel has its own vertical scale; Re_h = 0.1's is symmetric-logarithmic.*

## 4. P2: the reference optimum against s_L

| Layout | q | s_M | s_L | s*_ref at Re_h 0.1 (− s_L) | s*_ref at Re_h 10 (− s_L) |
|---|---|---|---|---|---|
| 30/90 | 0.2 | 0.3333 | 0.3116 | 0.3060 (-0.0056) | 0.3056 (-0.0060) |
| 30/90 | 0.35 | 0.4232 | 0.4039 | 0.4012 (-0.0027) | 0.4012 (-0.0027) |
| 30/90 | 0.65 | 0.5768 | 0.5655 | 0.5680 (+0.0025) | 0.5686 (+0.0031) |
| 30/90 | 0.8 | 0.6667 | 0.6608 | 0.6649 (+0.0041) | 0.6653 (+0.0045) |
| 15/15 | 0.2 | 0.3333 | 0.3321 | 0.3303 (-0.0018) | 0.3300 (-0.0021) |

*s_M is the planar Murray split √q/(√q + √(1 − q)). P2 passes where |s*_ref − s_L| ≤ 0.003.*
