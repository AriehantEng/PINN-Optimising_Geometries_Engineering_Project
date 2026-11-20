## P1 re-searched to a tight tolerance

Settings: {"xatol": 1e-05, "scan_step": 0.0005, "scan_points": 4, "mesh_delta": 0.01, "check_mesh": [0.005, 5.0]}; PINN xatol 1e-05. P1: |s*_PINN - s*_ref| <= 0.002; 'borderline' = within eps_s of that boundary.

### Reference cases

| case | Brent s* (solves) | vertex (fit, window) | J''/J | jitter | noise | spread | model | mesh shift | eps_s Brent | eps_s vertex |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-120_q0.5_Re0.1 | 0.452613 (15) | 0.452602 (parabola, +-0.0020) | 29.3 | 1.6e-06 | 4.7e-04 | 1.1e-05 | 8.8e-06 | -9.2e-06 | 4.9e-04 | 2.9e-05 |
| 0-120_q0.5_Re10 | 0.452751 (15) | 0.452737 (parabola, +-0.0020) | 28.2 | 2.6e-06 | 6.0e-04 | 6.3e-05 | 8.3e-06 | -1.6e-06 | 6.2e-04 | 7.3e-05 |
| 15-15_q0.2_Re0.1 | 0.330000 (13) | 0.329965 (parabola, +-0.0020) | 45.6 | 1.3e-07 | 1.1e-04 | 1.8e-06 | 4.4e-06 | +3.7e-06 | 1.2e-04 | 9.9e-06 |
| 15-15_q0.2_Re10 | 0.329709 (11) | 0.329739 (parabola, +-0.0020) | 45.7 | 8.5e-08 | 8.6e-05 | 1.5e-06 | 4.1e-06 | +5.1e-07 | 9.7e-05 | 6.1e-06 |
| 30-120_q0.5_Re100 | 0.385934 (14) | 0.385595 (cubic, +-0.0150) | 6.8 | 2.6e-05 | 3.9e-03 | 2.5e-04 | 1.1e-05 | -1.5e-04 | 4.1e-03 | 4.1e-04 |
| 30-90_q0.2_Re0.1 | 0.306349 (14) | 0.306205 (parabola, +-0.0020) | 38.3 | 1.0e-06 | 3.3e-04 | 1.3e-05 | 3.4e-06 | +8.8e-07 | 3.4e-04 | 1.7e-05 |
| 30-90_q0.2_Re10 | 0.306078 (13) | 0.305825 (parabola, +-0.0020) | 37.9 | 1.5e-06 | 3.9e-04 | 3.6e-05 | 6.9e-07 | -9.7e-06 | 4.1e-04 | 4.6e-05 |
| 30-90_q0.35_Re0.1 | 0.401238 (16) | 0.401085 (parabola, +-0.0020) | 34.8 | 1.1e-06 | 3.6e-04 | 1.2e-05 | 8.4e-06 | -1.9e-05 | 3.8e-04 | 4.0e-05 |
| 30-90_q0.35_Re10 | 0.401146 (13) | 0.401032 (parabola, +-0.0020) | 34.1 | 1.3e-06 | 4.0e-04 | 1.9e-05 | 6.6e-06 | +1.9e-06 | 4.1e-04 | 2.8e-05 |
| 30-90_q0.5_Re0.1 | 0.484338 (13); 0.484360 (15) | 0.484379 (parabola, +-0.0020) | 33.5 | 8.5e-07 | 3.2e-04 | 3.2e-06 | 4.1e-07 | -3.7e-06 | 3.3e-04 | 7.3e-06 |
| 30-90_q0.5_Re1 | 0.484344 (14) | 0.484383 (parabola, +-0.0020) | 34.3 | 7.2e-07 | 2.9e-04 | 4.7e-06 | 3.2e-06 | +8.3e-07 | 3.0e-04 | 8.8e-06 |
| 30-90_q0.5_Re10 | 0.484749 (15); 0.484848 (17) | 0.484790 (parabola, +-0.0020) | 33.5 | 1.0e-06 | 3.5e-04 | 9.7e-06 | 4.4e-06 | -4.0e-06 | 3.7e-04 | 1.8e-05 |
| 30-90_q0.5_Re100 | 0.488592 (13) | 0.488861 (parabola, +-0.0020) | 32.9 | 2.0e-06 | 4.9e-04 | 2.8e-05 | 2.4e-06 | -1.7e-05 | 5.2e-04 | 4.8e-05 |
| 30-90_q0.65_Re0.1 | 0.567955 (15) | 0.568180 (parabola, +-0.0020) | 35.2 | 6.9e-07 | 2.8e-04 | 1.1e-05 | 2.3e-06 | +1.7e-06 | 2.9e-04 | 1.5e-05 |
| 30-90_q0.65_Re10 | 0.568738 (11) | 0.568826 (parabola, +-0.0020) | 35.8 | 7.7e-07 | 2.9e-04 | 1.4e-05 | 2.2e-06 | -4.2e-06 | 3.1e-04 | 2.0e-05 |
| 30-90_q0.8_Re0.1 | 0.664651 (12) | 0.664622 (parabola, +-0.0020) | 40.2 | 4.7e-07 | 2.2e-04 | 5.4e-06 | 2.8e-06 | +5.9e-06 | 2.3e-04 | 1.4e-05 |
| 30-90_q0.8_Re10 | 0.665019 (17) | 0.665097 (parabola, +-0.0020) | 40.6 | 5.7e-07 | 2.4e-04 | 5.9e-06 | 5.3e-06 | -2.4e-06 | 2.5e-04 | 1.4e-05 |

### Runs

| run | family | stored δs* | δs* (Brent) | verdict | δs* (vertex) | verdict |
|---|---|---|---|---|---|---|
| parametric_re0.1_0120_q0.5_corners_s21 | Tier A | +0.00014 | +0.000009 | pass | +0.000021 | pass |
| parametric_re0.1_1515_q0.2_corners_s21 | Tier A | +0.05566 | +0.055746 | miss | +0.055781 | miss |
| parametric_re0.1_3090_q0.2_corners_s21 | Tier A | +0.15560 | +0.155251 | miss | +0.155395 | miss |
| parametric_re0.1_3090_q0.35_corners_s21 | Tier A | +0.03818 | +0.038144 | miss | +0.038298 | miss |
| parametric_re0.1_3090_q0.65_corners_s21 | Tier A | +0.00698 | +0.006916 | miss | +0.006691 | miss |
| parametric_re0.1_3090_q0.8_corners_s21 | Tier A | +0.00644 | +0.006916 | miss | +0.006946 | miss |
| parametric_re10_0120_q0.5_corners_s21 | Tier A | +0.00002 | -0.000153 | pass | -0.000139 | pass |
| parametric_re10_1515_q0.2_corners_s21 | Tier A | -0.00676 | -0.006559 | miss | -0.006589 | miss |
| parametric_re10_3090_q0.2_corners_s21 | Tier A | -0.00784 | -0.008463 | miss | -0.008211 | miss |
| parametric_re10_3090_q0.35_corners_s21 | Tier A | -0.00490 | -0.004625 | miss | -0.004510 | miss |
| parametric_re10_3090_q0.65_corners_s21 | Tier A | +0.00202 | +0.001672 | pass | +0.001584 | pass |
| parametric_re10_3090_q0.8_corners_s21 | Tier A | -0.00639 | -0.005833 | miss | -0.005911 | miss |
| parametric_re0.1_3090_q0.5_cut0.15_lm400_corners_s22 | amendment 3 | +0.00112 | +0.001152 | pass | +0.001134 | pass |
| parametric_re10_3090_q0.2_width_corners_s21 | amendment 3 | -0.01284 | -0.013434 | miss | -0.013182 | miss |
| parametric_re10_3090_q0.35_width_corners_s21 | amendment 3 | -0.00686 | -0.006851 | miss | -0.006736 | miss |
| parametric_re0.1_3090_q0.35_cut0.4_corners_s21 | controls | +0.03651 | +0.036155 | miss | +0.036309 | miss |
| parametric_re0.1_3090_q0.5_cut0.15_corners_s21 | controls | -0.00124 | -0.001389 | pass | -0.001408 | pass |
| parametric_re10_3090_q0.2_corners_s22 | controls | -0.00853 | -0.008994 | miss | -0.008742 | miss |
| parametric_re10_3090_q0.35_cut0.4_corners_s21 | controls | -0.00541 | -0.005443 | miss | -0.005328 | miss |
| parametric_re10_3090_q0.5_cut0.15_corners_s21 | controls | -0.00095 | -0.001025 | pass | -0.000967 | pass |
| parametric_re10_3090_corners_ease | earlier experiment | +0.00143 | +0.001211 | pass | +0.001170 | pass |
| parametric_re10_3090_corners_more | earlier experiment | +0.00076 | +0.001062 | pass | +0.001021 | pass |
| parametric_re10_3090_corners_next | earlier experiment | +0.00143 | +0.001115 | pass | +0.001074 | pass |
| parametric_re10_3090_corners_wide | earlier experiment | +0.00071 | +0.000709 | pass | +0.000668 | pass |
| parametric_re10_3090_equal | earlier experiment | -0.00027 | -0.000436 | pass | -0.000477 | pass |
| parametric_re0.1_3090_corners_s21 | q = 0.5 (earlier) | +0.00263 | +0.002749 | miss | +0.002708 | miss |
| parametric_re0.1_3090_corners_s22 | q = 0.5 (earlier) | +0.00581 | +0.005603 | miss | +0.005563 | miss |
| parametric_re0.1_3090_corners_s23 | q = 0.5 (earlier) | +0.00184 | +0.001946 | borderline | +0.001906 | pass |
| parametric_re100_30120_corners_s21 | q = 0.5 (earlier) | +0.01391 | +0.014195 | miss | +0.014534 | miss |
| parametric_re100_3090_corners_s21 | q = 0.5 (earlier) | -0.00504 | -0.004640 | miss | -0.004910 | miss |
| parametric_re100_3090_corners_s22 | q = 0.5 (earlier) | -0.00122 | -0.001204 | pass | -0.001473 | pass |
| parametric_re100_3090_corners_s23 | q = 0.5 (earlier) | +0.00062 | +0.000739 | pass | +0.000470 | pass |
| parametric_re10_30120_corners_s21 | q = 0.5 (earlier) | +0.00000 | no interior reference optimum | | | |
| parametric_re10_3090_corners_s22 | q = 0.5 (earlier) | +0.00040 | +0.000202 | pass | +0.000161 | pass |
| parametric_re10_3090_corners_s23 | q = 0.5 (earlier) | +0.00076 | +0.000992 | pass | +0.000951 | pass |
| parametric_re10_3090_equal_corners | q = 0.5 (earlier) | +0.00143 | +0.001603 | pass | +0.001562 | pass |
| parametric_re1_30120_corners_s21 | q = 0.5 (earlier) | +0.00000 | no interior reference optimum | | | |
| parametric_re1_3090_corners_s21 | q = 0.5 (earlier) | -0.00132 | -0.001604 | pass | -0.001644 | pass |

### Counts (pass / borderline / miss)

| group | runs | stored, band ±0.0010 | Brent, band ±eps_s | vertex, band ±eps_s |
|---|---|---|---|---|
| even share (q = 0.5) | 12 | 5 / 5 / 2 | 8 / 1 / 3 | 9 / 0 / 3 |
| uneven share | 10 | 0 / 1 / 9 | 1 / 0 / 9 | 1 / 0 / 9 |
| H2: Re 0.1, default cutoff, |δs*| <= 0.003 | 3 | 1 / 1 / 1 | 1 / 1 / 1 | 2 / 0 / 1 |
| controls (Amendment 1) | 5 | 1 / 1 / 3 | 2 / 0 / 3 | 2 / 0 / 3 |
| Amendment 3 | 3 | 0 / 1 / 2 | 1 / 0 / 2 | 1 / 0 / 2 |
| 30/120, Re 100 | 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 |
