## Table 3 inputs: the closed-length maps at every Re_h (exact)

From docs/re_study/closed-junction-all.json (bifurcation/junction_re.py). Departure = |s* − s_M|; 'resolved' = beyond its own location bound; Murray regret = J(s_M)/J* − 1. Optima next to a split with no steady solution are left out.

| Re_h | largest departure below 180° (layout) | largest up to 90° (layout) | largest at 180° (layout) | trusted | resolved | resolved in asymmetric layouts widening the more-turned daughter | median Murray regret | max Murray regret (layout) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.0080 (30/120, q = 0.9) | 0.0033 (10/80, q = 0.9) | 0.0037 (30/150, q = 0.1) | 165/165 | 83 | 13/62 | 0.0045% | 0.206% (30/150, q = 0.2) |
| 1 | 0.0080 (30/120, q = 0.9) | 0.0031 (0/60, q = 0.8) | 0.0037 (30/150, q = 0.1) | 165/165 | 81 | 13/61 | 0.0041% | 0.206% (30/150, q = 0.2) |
| 10 | 0.0082 (30/120, q = 0.9) | 0.0033 (10/80, q = 0.95) | 0.0059 (30/150, q = 0.05) | 165/165 | 91 | 16/65 | 0.0047% | 0.209% (30/120, q = 0.9) |
| 25 | 0.0086 (30/120, q = 0.9) | 0.0042 (10/80, q = 0.7) | 0.0291 (30/150, q = 0.1) | 165/165 | 104 | 22/76 | 0.0066% | 0.796% (30/150, q = 0.1) |
| 50 | 0.0094 (0/150, q = 0.1) | 0.0047 (10/80, q = 0.9) | 0.0672 (30/150, q = 0.05) | 164/165 | 130 | 28/89 | 0.0131% | 0.921% (30/150, q = 0.05) |
| 100 | 0.0131 (0/120, q = 0.1) | 0.0062 (10/80, q = 0.8) | 0.0290 (30/150, q = 0.4) | 162/165 | 132 | 25/91 | 0.0248% | 1.225% (30/150, q = 0.3) |
