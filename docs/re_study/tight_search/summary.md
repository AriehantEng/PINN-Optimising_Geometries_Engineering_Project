# P1 re-searched to a tight tolerance: results and draft text

Produced by `bifurcation/tight_search.py`. Its docstring has the method and
the commands. The tables behind every number below are in `recount.md`; the
data are in `reference.json`, `pinn.json`, `pinn_check.json`, `recount.json`
and `solves.jsonl`, which holds every finite-element solve (505).

## What was done

1. **PINN search.** It was repeated on all 38 finished parametric runs,
   using the preregistered procedure: a 61-point scan, then bounded Brent
   between the best point's neighbours. The only change is xatol, 0.002 →
   1e-5. Every scan reproduced the stored curve bit for bit. The optimum was
   then checked where it was found:
   - J at s* ± 1e-5 and s* ± 3e-5 is never lower.
   - A parabola through s* ± 2e-4 puts its vertex within 1.6e-6 of s*.

   So the PINN optima are now converged to about 1e-6.

2. **Reference search.** It was repeated on the preregistered reference
   mesh (max area 0.002, 6 corner levels), using the preregistered procedure:
   bounded Brent between the best evaluation split's neighbours. The only
   change is xatol, 0.002 → 1e-5.
   - Coverage: 17 cases with 19 brackets. The two cutoff-0.15 controls have
     their own bracket.
   - 30/120 at Re_h = 1 and 10 have no interior optimum (see below).
   - Cost: 11–17 solves per search.

3. **ε_s, rebuilt from measured parts.** This was needed because the
   reference mesh is regenerated for every split, so J carries a jitter from
   one split to the next. The jitter is 1e-7 to 3e-6 of J (rms), and Brent
   cannot place a minimum more finely than that jitter allows. Measured per
   case:
   - **noise:** the noise half-width sqrt(4σ/J″) from a 9-point scan at
     ±0.0005 steps around Brent's answer.
   - **mesh shift:** the reference mesh's own discretisation shift, measured
     against the next coarser mesh as in the answer keys (`mesh_bound`).
   - **vertex:** the vertex of a least-squares fit through every solve
     within ±0.002, with two error terms: its leave-one-out spread, and how
     far it moves when the fit gains a degree.

   Two bounds follow:
   - ε_s(Brent) = xatol + noise + |mesh shift|
   - ε_s(vertex) = spread + model + |mesh shift|

## ε_s

| | old (answer keys) | ε_s(Brent) | ε_s(vertex) |
|---|---|---|---|
| 30/90, every share and Re_h | 0.0010 | 0.00023 – 0.00052 | 0.000007 – 0.000048 |
| 0/120, Re_h = 0.1 and 10 | 0.0010 | 0.00049, 0.00062 | 0.000029, 0.000073 |
| 15/15, Re_h = 0.1 and 10 | 0.0010 | 0.00012, 0.00010 | 0.000010, 0.000006 |
| 30/120, Re_h = 100 | — | 0.0041 | 0.00041 |

Every term of the new ε_s is measured, not assumed. Leaving out 30/120 at
Re_h = 100, the ranges are:
- xatol: 1e-5
- noise: 0.9e-4 to 6.0e-4
- mesh shift: at most 1.9e-5
- leave-one-out spread: at most 6.3e-5
- model term: at most 8.8e-6

**The noise dominates ε_s(Brent).** The old 0.0010 was almost entirely the
reference Brent tolerance, as the review said. Tightening that tolerance does
not take ε_s down to the mesh term, though. It takes it down to the remeshing
noise: 1.6–10 times below 0.0010 for Brent, and 14–160 times below for the
vertex.

**30/120 at Re_h = 100 is the exception.**
- Its J is five times flatter (J″/J = 6.8 against about 34) and ten times
  noisier (jitter 2.6e-5).
- So Brent can place its optimum only to ±0.004.
- A 0.002 window could not resolve its curvature at all. It reported
  J″/J = 0.7, which is noise. So the case was rescanned at ±0.015 and fitted
  with a cubic, because the curve is not symmetric that far out.
- Its stored Brent answer 0.38593 sits in a noise dip. The cubic's minimum is
  0.38560.

**Both optima moved.** The reference optima moved by up to 0.00046
(q = 0.2, Re_h = 10) and the PINN optima by up to 0.00033: the old searches
stopped within ±0.001 and ±0.002. Some cases:
- 30/90, q = 0.5: at Re_h = 0.1 and 1, the tight Brent search returned the
  stored value to six digits (0.484338, 0.484344): both searches stopped in
  the same noise dip.
- q = 0.65, Re_h = 10: 0.568632 → 0.568738 (Brent), 0.568826 (vertex). This
  is still the preregistered value, "Brent's optimum on the reference mesh",
  and not the answer key's search-mesh 0.5695.
- 30/90 at Re_h = 1: the tight value is 0.48434 (vertex 0.48438). This
  confirms B2's 0.4843 against the answer key's search-mesh 0.4851.

## Recount

Pass: |δs*| ≤ 0.002. "Borderline" means within ε_s of that boundary. For the
stored values the band is the review's ±0.0010.

| group | stored, band ±0.0010 | tight, Brent band | tight, vertex band | preregistered verdict (no band) |
|---|---|---|---|---|
| even share (12) | 5 / 5 / 2 | **8 / 1 / 3** | **9 / 0 / 3** | 9 of 12 → **9 of 12, the same nine runs** |
| uneven shares (10) | 0 / 1 / 9 | **1 / 0 / 9** | **1 / 0 / 9** | 0 of 10 → **1 of 10** |
| H2: Re_h = 0.1, default cutoff, ≤ 0.003 (3) | 1 / 1 / 1 | 1 / 1 / 1 | 2 / 0 / 1 | 2 of 3 → **2 of 3** |
| controls, Amendment 1 (5) | 1 / 1 / 3 | 2 / 0 / 3 | 2 / 0 / 3 | 2 of 5 → 2 of 5 |

The first three verdict columns read pass / borderline / miss.

**Even share** (δs* from Brent, then from the vertex):

| run | stored | Brent | vertex | verdict |
|---|---|---|---|---|
| Re_h 0.1, seed 21 | +0.00263 | +0.00275 | +0.00271 | miss (was borderline) |
| Re_h 0.1, seed 22 | +0.00581 | +0.00560 | +0.00556 | miss |
| Re_h 0.1, seed 23 | +0.00184 | +0.00195 | +0.00191 | pass; borderline under the Brent band (0.00005 from the boundary, ε_s 0.00033) |
| Re_h 1, seed 21 | −0.00132 | −0.00160 | −0.00164 | pass (was borderline) |
| Re_h 10, seed 21 | +0.00143 | +0.00160 | +0.00156 | pass (was borderline) |
| Re_h 10, seed 22 | +0.00040 | +0.00020 | +0.00016 | pass |
| Re_h 10, seed 23 | +0.00076 | +0.00099 | +0.00095 | pass |
| Re_h 100, seed 21 | −0.00504 | −0.00464 | −0.00491 | miss |
| Re_h 100, seed 22 | −0.00122 | −0.00120 | −0.00147 | pass (was borderline) |
| Re_h 100, seed 23 | +0.00062 | +0.00074 | +0.00047 | pass |
| 0/120, Re_h 0.1 | +0.00014 | +0.00001 | +0.00002 | pass |
| 0/120, Re_h 10 | +0.00002 | −0.00015 | −0.00014 | pass |

**Uneven shares.** One verdict changes: **q = 0.65 at Re_h = 10** moves from
+0.00202 to **+0.00167 (Brent) / +0.00158 (vertex), a pass**.
- 0.00024 of the 0.00035 change is the PINN's own search tightening
  (0.570653 → 0.570410). The other 0.00011 is the reference
  (0.568632 → 0.568738).
- Under the vertex band it is a clear pass.
- Under the Brent band it clears the band by only 0.00002: the band's inner
  edge is 0.002 − 0.00031 = 0.00169.

The other nine stay clear misses. Each misses the boundary by at least six
times its ε_s:

| run | δs* (Brent) |
|---|---|
| q 0.2, Re_h 10 | −0.0085 |
| q 0.35, Re_h 10 | −0.0046 |
| q 0.8, Re_h 10 | −0.0058 |
| 15/15, Re_h 10 | −0.0066 |
| q 0.65, Re_h 0.1 | +0.0069 |
| q 0.8, Re_h 0.1 | +0.0069 |
| q 0.35, Re_h 0.1 | +0.038 |
| 15/15, Re_h 0.1 | +0.056 |
| q 0.2, Re_h 0.1 | +0.155 (PINN on the range bound) |

**Controls and amendments** (δs* from Brent):

| run | δs* |
|---|---|
| cutoff 0.15, Re_h 0.1 | −0.00139 |
| the same run after 400 more steps | +0.00115 |
| cutoff 0.15, Re_h 10 | −0.00103 |
| cutoff 0.40, q 0.35, Re_h 0.1 | +0.0362 |
| cutoff 0.40, q 0.35, Re_h 10 | −0.0054 |
| q 0.2 seed 22, Re_h 10 | −0.0090 |
| width-scaled q 0.2 | −0.0134 (from −0.0085) |
| width-scaled q 0.35 | −0.0069 (from −0.0046) |

**30/120:**
- At Re_h = 1 and 10, the reference optimum is on the evaluation grid's end
  and the PINN is on the range bound. Both are 0.35, and J rises
  monotonically across 0.35–0.65 in the answer keys, so the credit holds.
- At Re_h = 100: +0.0142 (Brent) / +0.0145 (vertex), a miss under every
  band.

## Statistics (three seeds per Re_h; tight Brent values)

| Re_h | δs* per seed | mean ± SD | 95% t-interval | stored, for comparison |
|---|---|---|---|---|
| 0.1 | +0.00275, +0.00560, +0.00195 | +0.0034 ± 0.0019 | −0.0013 to +0.0082 | +0.0034 ± 0.0021, −0.0018 to +0.0087 |
| 10 | +0.00160, +0.00020, +0.00099 | +0.0009 ± 0.0007 | −0.0008 to +0.0027 | +0.0009 ± 0.0005, −0.0004 to +0.0022 |
| 100 | −0.00464, −0.00120, +0.00074 | −0.0017 ± 0.0027 | −0.0085 to +0.0051 | −0.0019 ± 0.0029, −0.0091 to +0.0053 |

The vertex values agree to within 0.0001 in every mean and interval end.
Re_h = 1 has one seed, so it has no interval.

**Clopper–Pearson, uneven shares:** 1 pass in 10 gives a 95% upper bound of
44.5% (two-sided), or 39.4% (one-sided). The stored 0 in 10 gave 30.8%.

## Which s*_ref to report

**Recommendation: report the Brent value as the reference, and the vertex
as a robustness check.**
- The Brent value is the preregistered definition, with only its tolerance
  tightened.
- The vertex is a fit. It is sharper, but it was not preregistered.
- The two give the same preregistered verdict on every one of the 22 runs.
  They differ only on whether Re_h = 0.1 seed 23 is flagged as borderline.

---

## Draft text

The paper's current abstract, 3.3 and 5.1 are not in the repository. The
repository's `docs/paper/` holds the 20 October draft, whose structure is
different. So these are replacement passages for what the new numbers change,
written to be merged into the current wording, not to replace whole sections.
Appendix labels (B2, B3, B5) follow the review's usage.

### Abstract (the P1 sentences)

> Against the preregistered reference, the finite-element optimum on the
> reference mesh, re-searched with both optima converged to 1e-5, the PINN
> located the equal-share optimum within the preregistered ±0.002 in 9 of 12
> runs (Re_h = 0.1–100, two layouts). It did so in 1 of 10 runs at
> uneven shares (q = 0.2–0.8). The uncertainty in each reference optimum,
> measured rather than assumed, is 0.0001–0.0006. It is set by the
> remeshing noise of the reference solver, not by search tolerance, and it
> leaves one verdict within reach of the boundary.

### 3.3 (results: location of the optimum)

> **Search tolerance.** The verdicts below use optima re-searched with
> bounded Brent at xatol = 1e-5, both for the PINN and for the reference.
> The reference was searched on the preregistered reference mesh and bracket.
> The original searches stopped at ±0.002 and ±0.001, which made the
> reference's own Brent tolerance almost all of the uncertainty ε_s (0.0010)
> around the ±0.002 pass mark. At the tighter tolerance, the PINN optima are
> converged to about 1e-6.
>
> **The reference optimum's uncertainty.** The reference mesh is regenerated
> at every split. J therefore carries a split-to-split jitter of 1e-7 to
> 3e-6 of its value. No search tolerance can locate a minimum more finely
> than sqrt(4σ/J″), which is 0.0001 to 0.0006 here. With the mesh's own
> discretisation shift (at most 2e-5), this gives ε_s = 0.0001–0.0006 per
> case (Table X). A least-squares vertex through the solves near the optimum
> locates the minimum to 0.00001–0.00007. It gives the same verdict on every
> run.
>
> **Even share.** P1 passes in 9 of 12 runs, the same nine as before the
> re-search. One verdict lies within ε_s of the boundary: Re_h = 0.1,
> seed 23, at +0.00195, which passes. Of the five verdicts that lay within the
> earlier ±0.0010 band, four have resolved:
> - Re_h = 1 (−0.0016), Re_h = 10 seed 21 (+0.0016) and Re_h = 100 seed 22
>   (−0.0012) are passes.
> - Re_h = 0.1 seed 21 (+0.0027) is a miss.
>
> The two clear misses remain: Re_h = 0.1 seed 22 (+0.0056) and Re_h = 100
> seed 21 (−0.0046). Re_h = 1 rests on a single seed (seed 21). The mean
> argmin errors are:
> - +0.0034 ± 0.0019 at Re_h = 0.1 (95% interval −0.0013 to +0.0082)
> - +0.0009 ± 0.0007 at Re_h = 10 (−0.0008 to +0.0027)
> - −0.0017 ± 0.0027 at Re_h = 100 (−0.0085 to +0.0051)
>
> At 30/120, Re_h = 1 and 10, both optima lie on the lower range bound,
> because J rises monotonically from s = 0.35 to 0.65. At Re_h = 100 the
> PINN misses by +0.014. There J is five times flatter than at 30/90, and the
> reference optimum itself is fixed only to ±0.004 by Brent, or to ±0.0004
> by a cubic fit.
>
> **Uneven shares.** P1 passes in 1 of 10 runs: q = 0.65 at Re_h = 10, at
> +0.0017. Before the re-search it lay at +0.0020, on the boundary. It now
> passes against the preregistered reference, 0.5687 on the reference mesh.
> It clears the reference's ε_s (0.0003) by only 0.00002. The other nine
> miss by 0.0046 to 0.155, each by at least six times its ε_s. The 95%
> Clopper–Pearson upper bound on the uneven-share pass rate is 44.5%
> (two-sided).
>
> **H2** is judged on the three default-cutoff seeds at Re_h = 0.1. Two of
> three lie within 0.003 (+0.0027 and +0.0019). Seed 22 does not (+0.0056).
> The two cutoff-0.15 runs are Amendment 1 controls, reported separately.
> Lowering the cutoff alone gives −0.0014. Continuing that run for 400 steps
> gives +0.0012. One run continues the other, so they are not independent
> results.

### 5.1 (discussion: what limits the verdicts)

> The earlier uncertainty band of ±0.0010 around the pass mark was a property
> of the search, not of the solver. It came almost entirely from the
> reference's Brent tolerance. Tightening both searches resolved four of the
> five even-share runs that lay inside it, and it changed no preregistered
> verdict at the even share. At the uneven shares it moved one run, q = 0.65
> at Re_h = 10, from the boundary to a narrow pass. That raises the
> Clopper–Pearson bound from 30.8% to 44.5% without changing the conclusion:
> at uneven shares the PINN does not locate the optimum.
>
> **What remains is a floor set by the reference solver.** Each reference
> solve meshes its geometry afresh, and a fresh mesh moves J by about 1e-6 of
> its value. Near a minimum with J″/J ≈ 34, that is enough to reorder points
> 0.0003 apart, so a minimiser on the reference mesh cannot do better however
> tight its tolerance. Fitting the minimum through many solves instead of
> picking the lowest one removes most of this floor. It gives 1e-5 here, and
> it puts every verdict at least 0.00009 from the boundary.
>
> The floor grows as the optimum flattens. At 30/120, Re_h = 100, J″/J is
> 6.8 and the jitter is 2.6e-5, so a Brent search there is uncertain by
> ±0.004, twice the pass mark itself. A geometry optimisation that compares
> optima more finely than the reference's remeshing noise must fix the mesh
> topology across designs, or locate the optimum by a fit rather than by the
> lowest sample.
