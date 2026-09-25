## Every run

| Setup | Run | u error, developed region (rel. L2) | u error at the inlet (rel. L2) | wiggle at x* = 4 (max abs error) | wiggle at x* = 10 (max abs error) | worst flux error (relative) | RMS x-momentum residual | RMS y-momentum residual | RMS continuity residual | largest residual, any equation | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | developing_re10 | 2.43e-04 | 6.36e-04 | 1.24e-04 | 5.71e-04 | 3.16e-04 | 1.08e-03 | 8.64e-04 | 7.16e-04 | 3.94e-02 | 7.67e-06 | 40.3 |
| baseline | developing_re10_seed1 | 3.07e-04 | 9.24e-04 | 3.17e-04 | 3.32e-04 | 3.74e-04 | 9.18e-04 | 8.40e-04 | 7.62e-04 | 4.87e-02 | 1.28e-05 | 38.8 |
| baseline | developing_re10_seed2 | 3.68e-04 | 5.55e-04 | 3.84e-04 | 5.78e-04 | 4.98e-04 | 1.05e-03 | 8.37e-04 | 7.57e-04 | 4.25e-02 | 1.06e-05 | 39.4 |
| rar | developing_re10_rar | 4.21e-04 | 9.43e-04 | 4.81e-04 | 2.41e-04 | 6.89e-04 | 1.09e-03 | 9.32e-04 | 8.33e-04 | 4.56e-02 | 9.95e-06 | 47.0 |

## Each setup: mean, with the range over its seeds

| Setup | Seeds | u error, developed region (rel. L2) | u error at the inlet (rel. L2) | wiggle at x* = 4 (max abs error) | wiggle at x* = 10 (max abs error) | worst flux error (relative) | RMS x-momentum residual | RMS y-momentum residual | RMS continuity residual | largest residual, any equation | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 3.06e-04 (2.43e-04 to 3.68e-04) | 7.05e-04 (5.55e-04 to 9.24e-04) | 2.75e-04 (1.24e-04 to 3.84e-04) | 4.94e-04 (3.32e-04 to 5.78e-04) | 3.96e-04 (3.16e-04 to 4.98e-04) | 1.02e-03 (9.18e-04 to 1.08e-03) | 8.47e-04 (8.37e-04 to 8.64e-04) | 7.45e-04 (7.16e-04 to 7.62e-04) | 4.35e-02 (3.94e-02 to 4.87e-02) | 1.04e-05 (7.67e-06 to 1.28e-05) | 39.5 (38.8 to 40.3) |
| rar | 1 | 4.21e-04 | 9.43e-04 | 4.81e-04 | 2.41e-04 | 6.89e-04 | 1.09e-03 | 9.32e-04 | 8.33e-04 | 4.56e-02 | 9.95e-06 | 47.0 |

## Against the baseline seed range

A setup counts as better or worse on a metric only if its mean lies outside the range spanned by the baseline seeds.

| Setup | u error, developed region (rel. L2) | u error at the inlet (rel. L2) | wiggle at x* = 4 (max abs error) | wiggle at x* = 10 (max abs error) | worst flux error (relative) | RMS x-momentum residual | RMS y-momentum residual | RMS continuity residual | largest residual, any equation | held-out loss |
|---|---|---|---|---|---|---|---|---|---|---|
| rar | worse | worse | worse | better | worse | worse | worse | worse | within seed range | within seed range |
