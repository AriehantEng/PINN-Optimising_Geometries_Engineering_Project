## Every run

| Setup | Run | u error, developed region (rel. L2) | u error at the inlet (rel. L2) | wiggle at x* = 4 (max abs error) | wiggle at x* = 10 (max abs error) | worst flux error (relative) | RMS x-momentum residual | RMS y-momentum residual | RMS continuity residual | largest residual, any equation | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | developing_re10 | 2.43e-04 | 6.36e-04 | 1.24e-04 | 5.71e-04 | 3.16e-04 | 1.08e-03 | 8.64e-04 | 7.16e-04 | 3.94e-02 | 7.67e-06 | 40.3 |
| baseline | developing_re10_seed1 | 3.07e-04 | 9.24e-04 | 3.17e-04 | 3.32e-04 | 3.74e-04 | 9.18e-04 | 8.40e-04 | 7.62e-04 | 4.87e-02 | 1.28e-05 | 38.8 |
| baseline | developing_re10_seed2 | 3.68e-04 | 5.55e-04 | 3.84e-04 | 5.78e-04 | 4.98e-04 | 1.05e-03 | 8.37e-04 | 7.57e-04 | 4.25e-02 | 1.06e-05 | 39.4 |
| E1 24 L-BFGS cycles | developing_re10_lbfgs24 | 3.73e-04 | 3.84e-04 | 8.37e-05 | 5.73e-04 | 5.00e-04 | 8.27e-04 | 6.22e-04 | 5.27e-04 | 2.37e-02 | 3.89e-06 | 70.9 |
| E2 10k Adam cosine | developing_re10_adam10k | 4.40e-04 | 5.80e-04 | 1.72e-04 | 8.74e-04 | 6.89e-04 | 1.10e-03 | 7.28e-04 | 6.65e-04 | 3.07e-02 | 6.45e-06 | 58.5 |
| E3a inlet sharpness 4 | developing_re10_m4 | 1.54e-04 | 5.22e-04 | 1.31e-04 | 1.61e-04 | 2.46e-04 | 5.72e-04 | 4.46e-04 | 4.91e-04 | 1.59e-02 | 2.77e-06 | 39.6 |
| E3b inlet sharpness 2 | developing_re10_m2 | 9.07e-05 | 4.86e-04 | 8.92e-05 | 1.66e-04 | 8.99e-05 | 2.75e-04 | 2.36e-04 | 2.26e-04 | 6.77e-03 | 7.52e-07 | 39.8 |
| E3c fixed dense inlet grid | developing_re10_inletgrid | 4.02e-04 | 2.57e-04 | 3.61e-04 | 6.62e-04 | 6.65e-04 | 1.70e-03 | 1.37e-03 | 1.32e-03 | 1.09e-01 | 1.96e-05 | 39.4 |
| E4 proportional RAR | developing_re10_rad | 3.38e-04 | 6.90e-04 | 2.22e-04 | 6.36e-04 | 5.81e-04 | 8.90e-04 | 7.47e-04 | 6.06e-04 | 4.11e-02 | 6.71e-06 | 44.0 |
| E5 hard inlet | developing_re10_hardinlet | 1.12e-02 | 5.79e-17 | 7.01e-03 | 1.56e-02 | 1.78e-02 | 1.66e-02 | 1.70e-02 | 1.77e-02 | 2.00e+00 | 5.77e-03 | 36.5 |
| E6 SiLU | developing_re10_silu | 1.01e-03 | 1.04e-03 | 6.23e-04 | 5.08e-04 | 1.31e-03 | 1.43e-03 | 1.16e-03 | 1.22e-03 | 9.53e-02 | 2.10e-05 | 41.4 |
| E7 stream function | developing_re10_stream | 2.96e-04 | 1.95e-03 | 2.31e-04 | 2.56e-04 | 6.22e-15 | 1.64e-03 | 1.32e-03 | 1.83e-16 | 4.11e-02 | 1.95e-05 | 72.3 |
| greedy RAR (earlier) | developing_re10_rar | 4.21e-04 | 9.43e-04 | 4.81e-04 | 2.41e-04 | 6.89e-04 | 1.09e-03 | 9.32e-04 | 8.33e-04 | 4.56e-02 | 9.95e-06 | 47.0 |

## Each setup: mean, with the range over its seeds

| Setup | Seeds | u error, developed region (rel. L2) | u error at the inlet (rel. L2) | wiggle at x* = 4 (max abs error) | wiggle at x* = 10 (max abs error) | worst flux error (relative) | RMS x-momentum residual | RMS y-momentum residual | RMS continuity residual | largest residual, any equation | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 3.06e-04 (2.43e-04 to 3.68e-04) | 7.05e-04 (5.55e-04 to 9.24e-04) | 2.75e-04 (1.24e-04 to 3.84e-04) | 4.94e-04 (3.32e-04 to 5.78e-04) | 3.96e-04 (3.16e-04 to 4.98e-04) | 1.02e-03 (9.18e-04 to 1.08e-03) | 8.47e-04 (8.37e-04 to 8.64e-04) | 7.45e-04 (7.16e-04 to 7.62e-04) | 4.35e-02 (3.94e-02 to 4.87e-02) | 1.04e-05 (7.67e-06 to 1.28e-05) | 39.5 (38.8 to 40.3) |
| E1 24 L-BFGS cycles | 1 | 3.73e-04 | 3.84e-04 | 8.37e-05 | 5.73e-04 | 5.00e-04 | 8.27e-04 | 6.22e-04 | 5.27e-04 | 2.37e-02 | 3.89e-06 | 70.9 |
| E2 10k Adam cosine | 1 | 4.40e-04 | 5.80e-04 | 1.72e-04 | 8.74e-04 | 6.89e-04 | 1.10e-03 | 7.28e-04 | 6.65e-04 | 3.07e-02 | 6.45e-06 | 58.5 |
| E3a inlet sharpness 4 | 1 | 1.54e-04 | 5.22e-04 | 1.31e-04 | 1.61e-04 | 2.46e-04 | 5.72e-04 | 4.46e-04 | 4.91e-04 | 1.59e-02 | 2.77e-06 | 39.6 |
| E3b inlet sharpness 2 | 1 | 9.07e-05 | 4.86e-04 | 8.92e-05 | 1.66e-04 | 8.99e-05 | 2.75e-04 | 2.36e-04 | 2.26e-04 | 6.77e-03 | 7.52e-07 | 39.8 |
| E3c fixed dense inlet grid | 1 | 4.02e-04 | 2.57e-04 | 3.61e-04 | 6.62e-04 | 6.65e-04 | 1.70e-03 | 1.37e-03 | 1.32e-03 | 1.09e-01 | 1.96e-05 | 39.4 |
| E4 proportional RAR | 1 | 3.38e-04 | 6.90e-04 | 2.22e-04 | 6.36e-04 | 5.81e-04 | 8.90e-04 | 7.47e-04 | 6.06e-04 | 4.11e-02 | 6.71e-06 | 44.0 |
| E5 hard inlet | 1 | 1.12e-02 | 5.79e-17 | 7.01e-03 | 1.56e-02 | 1.78e-02 | 1.66e-02 | 1.70e-02 | 1.77e-02 | 2.00e+00 | 5.77e-03 | 36.5 |
| E6 SiLU | 1 | 1.01e-03 | 1.04e-03 | 6.23e-04 | 5.08e-04 | 1.31e-03 | 1.43e-03 | 1.16e-03 | 1.22e-03 | 9.53e-02 | 2.10e-05 | 41.4 |
| E7 stream function | 1 | 2.96e-04 | 1.95e-03 | 2.31e-04 | 2.56e-04 | 6.22e-15 | 1.64e-03 | 1.32e-03 | 1.83e-16 | 4.11e-02 | 1.95e-05 | 72.3 |
| greedy RAR (earlier) | 1 | 4.21e-04 | 9.43e-04 | 4.81e-04 | 2.41e-04 | 6.89e-04 | 1.09e-03 | 9.32e-04 | 8.33e-04 | 4.56e-02 | 9.95e-06 | 47.0 |

## Against the baseline seed range

A setup counts as better or worse on a metric only if its mean lies outside the range spanned by the baseline seeds.

| Setup | u error, developed region (rel. L2) | u error at the inlet (rel. L2) | wiggle at x* = 4 (max abs error) | wiggle at x* = 10 (max abs error) | worst flux error (relative) | RMS x-momentum residual | RMS y-momentum residual | RMS continuity residual | largest residual, any equation | held-out loss |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 24 L-BFGS cycles | worse | better | better | within seed range | worse | better | better | better | better | better |
| E2 10k Adam cosine | worse | within seed range | within seed range | worse | worse | worse | better | better | better | better |
| E3a inlet sharpness 4 | better | better | within seed range | better | better | better | better | better | better | better |
| E3b inlet sharpness 2 | better | better | better | better | better | better | better | better | better | better |
| E3c fixed dense inlet grid | worse | better | within seed range | worse | worse | worse | worse | worse | worse | worse |
| E4 proportional RAR | within seed range | within seed range | within seed range | worse | worse | better | better | better | within seed range | better |
| E5 hard inlet | worse | better | worse | worse | worse | worse | worse | worse | worse | worse |
| E6 SiLU | worse | worse | worse | within seed range | worse | worse | worse | worse | worse | worse |
| E7 stream function | within seed range | worse | within seed range | better | better | worse | worse | better | within seed range | worse |
| greedy RAR (earlier) | worse | worse | worse | better | worse | worse | worse | worse | within seed range | within seed range |
