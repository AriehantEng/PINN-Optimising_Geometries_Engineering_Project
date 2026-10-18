## 30/90 across seeds (seeds 21, 22, 23; Re_h 1 has seed 21 only)

Argmin error = PINN s* − reference-mesh s*; regret = J_ref(s*_PINN)/J_ref* − 1; drop error = PINN J / reference J − 1. Mean ± sample standard deviation over the seeds.

| Re_h | seeds | reference s* | PINN s* (each seed) | argmin error, mean ± sd | regret, mean (max) | drop error at 0.50, mean ± sd | drop error at 0.65, mean ± sd | u error at 0.50, mean |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 21, 22, 23 | 0.4843 | 0.4870, 0.4902, 0.4862 | +0.0034 ± 0.0021 | 0.0245% (0.0563%) | +0.06% ± 0.22% | -3.70% ± 0.98% | 2.76e-03 |
| 1 | 21 | 0.4843 | 0.4830 | -0.0013 ± nan | 0.0032% (0.0032%) | +0.54% ± nan% | -3.14% ± nan% | 3.32e-03 |
| 10 | 21, 22, 23 | 0.4848 | 0.4862, 0.4852, 0.4855 | +0.0009 ± 0.0005 | 0.0016% (0.0035%) | -0.04% ± 0.04% | -0.95% ± 0.19% | 1.97e-03 |
| 100 | 21, 22, 23 | 0.4889 | 0.4838, 0.4876, 0.4895 | -0.0019 ± 0.0029 | 0.0154% (0.0421%) | -0.71% ± 0.56% | -3.51% ± 0.55% | 5.88e-03 |
