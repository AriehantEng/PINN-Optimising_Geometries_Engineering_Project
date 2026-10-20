# Tier A: pass marks and rules, fixed before any run

This file was committed before the screening solves and before any Tier A
network was trained; its commit is the time stamp. Nothing in it is changed
after a result is seen. The paper's limitations section notes that the earlier
hypotheses' quantitative forms were set after the first results; for Tier A
they are not, and the two tolerances are the paper's own, unchanged.

## The runs

Twelve parametric PINN runs, one seed each (seed 21), Re_h = 0.1 and 10:

| case | q | why |
|---|---|---|
| 30/90 | 0.2, 0.35, 0.65, 0.8 | shares other than 0.5; 30/90 is not mirror-symmetric, so q and 1 − q test different things |
| 15/15 | 0.2 | null test: a symmetric layout, where any miss of the reference optimum is network bias |
| 0/120 | 0.5 | a turned-back daughter with a larger length shift, interior at daughters 8 long |

Geometry as the 13 earlier runs: parent 3, daughters' outer walls 4, minimum
width 0.2. Re_h = 0.1 in viscous pressure units (μU_c/h), Re_h = 10 in
inertial units (ρU_c²).

## The trained split range

Centred on the length-corrected split s_L of each (layout, q), worked out
without any flow solve: s/(1 − s) = (L₁/L₂)^(1/4) √(q/(1 − q)) with each
daughter's centreline length measured on the geometry at s_L itself
(bifurcation/sensitivity.length_corrected_split). Range s_L ± 0.15, kept
inside [0.1, 0.9]; the seven evaluation splits are evenly spaced over it,
ends included.

## Screening (before training)

Each (layout, q, Re_h) is first solved by the reference finite-element solver
with daughters 4 long, by the answer key's procedure (17-point scan over
[0.1, 0.9], Brent to ±0.001 on the search mesh, re-solve on the check mesh).
A case is trained only if:

1. the optimum is feasible and not on the search bound (0.1 or 0.9);
2. it is not at the geometry's feasibility limit (no refused split beside it;
   bifurcation/answer_key.feasibility_limit), so J has an interior minimum;
3. the minimum survives the check mesh;
4. it lies inside the trained range, at least 0.05 from either end.

A case that fails is recorded as SKIPPED with its reason and is not trained.

## Pass marks

- **P1 (H3: the network locates the optimum).** |s*_PINN − s*_ref| ≤ 0.002,
  the ±0.002 search tolerance of the paper. s*_ref is Brent's optimum on the
  reference mesh from the evaluation splits, as in the 13 earlier runs. This
  is also the null test's pass mark at 15/15: the comparison is against the
  reference optimum, not s_M, because even at 15/15 the resolved optimum may
  depart from s_M by about 0.002 (closed-length maps, Table 3).
- **P2 (the length rule, H1's surviving form).** |s*_ref − s_L| ≤ 0.003 where
  no daughter turns beyond 90° (30/90 and 15/15), at each Reynolds number.
  0/120 turns a daughter back and is outside P2's scope; its s*_ref − s_L is
  reported, with no pass mark.

## Reported, with no pass mark

Regret J_ref(s*_PINN)/J_ref(s*_ref) − 1; the signed argmin error (to see
whether the Re_h = 0.1 bias of 30/90 at q = 0.5 depends on the share);
pressure-drop and field errors at the seven splits and at s*_PINN;
s*_ref − s_M and s*_ref − s_L at both Reynolds numbers; training time and
the convergence verdict (the same definition as before: finished, every loss
finite, and the best held-out loss of the last 25 LM steps not more than 10%
below the best before them).

## Fixed protocol

The network, sampling, schedule (Adam 3,000; L-BFGS 4 × 500; LM 150 with
fresh points every step and a 20-iterate average) and held-out selection are
those of the 13 earlier runs. The only change made to fit a geometry is the
corner-mode cutoff: min(0.5, R − 0.2), rounded down to 0.05, where R is the
largest cutoff the geometry admits across the trained range (as the original
0.5 was 0.2 short of the nearest other edge). A failed run is recorded, not
rerun until it works.
