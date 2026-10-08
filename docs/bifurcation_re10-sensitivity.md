## How much of the departure from the Murray split is real? Re_h = 10

s_M: Murray split. s_L: Murray split corrected for the daughters' centreline lengths (no fitted parameter). s_model: lengths plus the junction-loss model, K(alpha) fitted on the symmetric layouts only. Bound: |mesh-induced argmin shift| + search tolerance (0.001). Resolved: departures larger than their own bound. Murray regret: J(s_M)/J* - 1, both on the search mesh.

| layout | max abs(s*-s_M) | max abs(s*-s_L) | max abs(s*-s_model) | max bound | resolved vs s_M | resolved vs s_L | Murray regret: median, max |
|---|---|---|---|---|---|---|---|
| 15/15 | 0.0023 | 0.0016 | 0.0011 | 0.0010 | 7/11 | 5/11 | 0.007%, 0.014% |
| 0/30 | 0.0040 | 0.0018 | 0.0015 | 0.0010 | 8/11 | 5/11 | 0.009%, 0.043% |
| 20/40 | 0.0035 | 0.0016 | 0.0015 | 0.0010 | 10/11 | 2/11 | 0.008%, 0.033% |
| 0/60 | 0.0073 | 0.0021 | 0.0016 | 0.0010 | 9/11 | 7/11 | 0.048%, 0.132% |
| 45/45 | 0.0026 | 0.0006 | 0.0008 | 0.0010 | 8/11 | 0/11 | 0.009%, 0.025% |
| 10/80 | 0.0098 | 0.0019 | 0.0020 | 0.0010 | 9/11 | 6/11 | 0.085%, 0.240% |
| 60/60 | 0.0048 | 0.0021 | 0.0022 | 0.0010 | 9/11 | 5/11 | 0.015%, 0.073% |
| 30/90 | 0.0126 | 0.0025 | 0.0032 | 0.0010 | 10/11 | 9/11 | 0.101%, 0.412% |
| 0/120 | 0.0233 | 0.0080 | 0.0061 | 0.0010 | 11/11 | 9/11 | 0.717%, 1.226% |
| 75/75 | 0.0134 | 0.0026 | 0.0038 | 0.0010 | 8/11 | 6/11 | 0.152%, 0.423% |
| 30/120 | 0.0427 | 0.0144 | 0.0133 | 0.0010 | 11/11 | 9/11 | 1.736%, 3.799% |
| 0/150 | 0.0721 | 0.0353 | 0.0331 | 0.0010 | 11/11 | 11/11 | 5.069%, 8.926% |
| 90/90 | 0.0014 | 0.0014 | - | 0.0010 | 3/11 | 3/11 | 0.003%, 0.014% |
| 60/120 | 0.0025 | 0.0025 | - | 0.0011 | 1/11 | 1/11 | 0.000%, 0.023% |
| 30/150 | 0.0059 | 0.0059 | - | 0.0019 | 4/11 | 4/11 | 0.005%, 0.205% |

Murray regret:
- all (165 optima): median 0.0161%, max 8.926%
- bifurcation angle up to 90 degrees (66 optima): median 0.0106%, max 0.240%
- 120 and 150 degrees (66 optima): median 0.3373%, max 8.926%
- 180 degrees (33 optima): median 0.0025%, max 0.205%

Junction-loss fit on the symmetric layouts: K(alpha) = a + b (1 - cos alpha), a = 0.222, b = 0.706. Per layout: 15/15: K = 0.568 (rms 0.0002, lengths only 0.0012); 45/45: K = 0.170 (rms 0.0001, lengths only 0.0004); 60/60: K = 0.116 (rms 0.0007, lengths only 0.0008); 75/75: K = 1.143 (rms 0.0018, lengths only 0.0029)

Measured loss coefficients at q = 0.5 (path drop less centreline friction, per 1/2 U_i^2): 15/15: 0.61, 0.61; 0/30: 0.49, 0.57; 20/40: 0.73, 0.49; 0/60: 0.31, 0.65; 45/45: 0.63, 0.63; 10/80: 0.49, 0.41; 60/60: 0.63, 0.63; 30/90: 0.93, -0.21; 0/120: -0.96, 0.31; 75/75: 0.60, 0.60; 30/120: -1.29, -2.51; 0/150: -4.65, -2.17; 90/90: 2.29, 2.29; 60/120: 1.47, 4.42; 30/150: 0.92, 17.30

Reversal shares below 180 degrees (s - s_M turns from negative to positive between):

- 0/30: observed [0.7, 0.8], lengths None, lengths + junction model None
- 0/60: observed [0.9, 0.95], lengths None, lengths + junction model None
- 10/80: observed [0.9, 0.95], lengths None, lengths + junction model None
- 15/15: observed [0.5, 0.6], lengths [0.4, 0.5], lengths + junction model [0.4, 0.5]
- 20/40: observed [0.6, 0.7], lengths [0.8, 0.9], lengths + junction model [0.7, 0.8]
- 30/90: observed [0.8, 0.9], lengths [0.9, 0.95], lengths + junction model None
- 45/45: observed [0.4, 0.5], lengths [0.4, 0.5], lengths + junction model [0.4, 0.5]
- 60/60: observed [0.4, 0.5], lengths [0.4, 0.5], lengths + junction model [0.4, 0.5]
- 75/75: observed [0.4, 0.5], lengths [0.4, 0.5], lengths + junction model [0.4, 0.5]

19.0 min.
