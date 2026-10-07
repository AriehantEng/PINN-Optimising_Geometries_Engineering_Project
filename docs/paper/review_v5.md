# Review of the draft: Introduction v5, Literature Review, Appendix A

Every number below was recomputed from the repository's results; the files are named where they are used.

## Verdict

The scholarship is strong. The CARS structure works, the citation audit is careful, and the two-gap framing (the junction and the tools) is clear. Appendix A is correct line by line, and its A.3 is better than the version in `methods.md`.

The draft now has three structural problems and five factual ones. Most of them come from the study having moved on since version 5:
- inserting the literature review shifted every section number;
- the hypotheses and contributions predate three results: the parametric network, the width-rule study and the energy check;
- the evidence-check appendix still describes an earlier novelty claim.

None needs new work beyond what is already in the repository. Replacement text is given for each fix.

---

## Critical: fix before anyone reads it

### C1. Section numbers are off by one everywhere

The literature review is now Section 2, but the introduction's cross-references and roadmap still use the old numbering, where Section 2 was the problem formulation. Renumber to: **2 Literature review · 3 Problem formulation · 4 Methods · 5 The resolved design map · 6 The networks as solvers and optimisers · 7 Cost and the energy–shear trade-off · 8 Conclusions.**

| In the introduction | Now says | Should say |
|---|---|---|
| geometry, "(Section 2)", twice in Step 3 | 2 | 3 |
| free-split floor 0.1–0.2%, "(Section 4)" | 4 | 5 |
| "as Section 5 shows, a PINN … lost 6.3%" | 5 | 6 |
| corner residual and FE reference, "(Section 3)", three times | 3 | 4 |
| shear factor of 2.6, "(Section 6)" | 6 | 5 (a design-map result) |
| contributions "(Section 4)", "(Section 5)", "(Section 6)" | 4, 5, 6 | 5, 6, 7 |
| roadmap list | | rewrite (replacement text below) |

The literature review's last sentence ("Section 3 sets out the design problem") is already correct under the new numbering. `methods.md` has been renumbered to Sections 3 and 4 to match.

### C2. Five statements the data contradict or qualify

1. **"the orientation of a fixed bifurcation angle changes it by up to 23%"** is true only at equal shares, where it is 22.6% at 150° and 20.3% at 180°. Over all shares it is a factor of 2.7: at 180° with q = 0.05, 30/150 needs 12.15 against 4.56 for 90/90, because 95% of the flow must turn back by 150°. Replace with: *"…changes it by up to 23% at equal shares, and by up to a factor of 2.7 when most of the flow must turn back."*
2. **Network 1, "about three times … the reference's own estimated error"**: 3.26 × 10⁻⁶ / 8.7 × 10⁻⁷ = 3.7, so *"about four times"*. The "sixty times" for the objective is right (58).
3. **"its pressure agrees to within the reference's error"**: the pressure error is 1.73 × 10⁻⁵ against the reference's own 1.5 × 10⁻⁵, a ratio of 1.15. Replace with: *"its pressure error (1.7 × 10⁻⁵) is at the level of the reference's own (1.5 × 10⁻⁵), so it cannot be resolved further."*
4. **H1's "to within the resolution of the reference"**: the search resolves s to ±0.001, but gently turned layouts depart by 0.002–0.007 (0/60 by 0.007; 10/80 by 0.010). Stated this way, H1 is falsified by your own data. See C4.
5. **"an energy-consistency check … becomes its only internal diagnostic (Appendix A.6)"** was a promise: the check had never been computed for the bifurcation networks. It now has been (`bifurcation/energy.py`, `docs/bifurcation_re10-energy.md`); report it in Section 6.

### C3. The constraint w₁ + w₂ = w₀ needs a reason, and the reason is now in the data

The introduction cites the square law, w₀² = w₁² + w₂², as the planar optimum (Stephenson et al., 2015; Zhou et al., 2024). It then imposes w₁ + w₂ = w₀ without saying why. A reviewer will ask why the study does not use the law it cites. The width-rule study (`docs/bifurcation_re10-width-rules.md`, 68 finite-element solves) answers this:
- **Pressure drop alone has no optimum in total width.** At 60/60 and 50/50, J is 7.37 at w₀, 3.06 at 1.41 w₀ and 1.57 at 2 w₀.
- **With equal lengths, w₁ + w₂ = w₀ fixes the daughters' total volume** at L_d w₀. So s_M is the volume-constrained optimum, the same proportions as A.2(a).
- **The square law also chooses that volume.** Under Murray's pumping-plus-volume cost, priced so that the parent's own width is optimal, the resolved optimum total width is 1.42–1.43 w₀ at 60/60 and 1.37–1.38 w₀ at 0/150, against the square law's 1.414. The square law beats w₁ + w₂ = w₀ by 10.6–15.4% at 50/50.

Add to Step 3, after "The daughters share the parent's width, w₁ + w₂ = w₀.": *"For daughters of equal length this fixes their total volume, so the split is the volume-constrained optimum; the square law also optimises the volume itself, and Section 5.6 shows that the resolved junction moves that optimum by at most 3%."* This turns a weakness into a result.

### C4. Rewrite the hypotheses to match what the data can test

Your bracketed note about H1 asks the right question. If the hypotheses were refined after the first results, say so in Section 3 or frame them as expectations; do not present them as fixed in advance. Either way, three findings bear on them.
- **H1 needs a number.** Proposed wording: *"where both daughters turn by similar amounts, the optimal split stays within 0.01 of s_M."* The data support it for the symmetric layouts up to 120° (at most 0.005, at every share) and at 180° (at most 0.006). It is narrowly exceeded at 150°, where 75/75 departs by 0.013 at q = 0.95.
- **H2 holds below 180°, but it is half of a pattern that A.5 already predicts in full.** In the asymmetric layouts below 180°, the turned daughter is wider at nearly all shares: at all 11 shares in 0/120, 30/120 and 0/150. The direction reverses only when most of the flow goes down the straighter daughter (20/40 for q ≥ 0.7; 0/30 and 30/90 at q = 0.95).
- **The symmetric layouts show the same mechanism with equal loss coefficients.** The higher-flow daughter gets more width than s_M. At 15/15, 45/45, 60/60 and 75/75, s* − s_M is negative for q ≤ 0.3, positive for q ≥ 0.8, and within ±0.001 in between. The largest departure grows with the bifurcation angle: 0.002, 0.003, 0.005, 0.013. Your own A.5 condition, s* > s_M ⇔ K₁ s_M³ > K₂ (1 − s_M)³, predicts exactly this with K₁ = K₂, and it predicts the reversals in the mildly asymmetric layouts with K₂ > K₁.
- **At 180° (the face septum) neither effect appears;** the departure is at most 0.006 at every share. Proposed H2: *"the optimum widens the daughter with the larger junction loss, K_i q_i³/s_i²: the more-turned daughter, and at equal turning the higher-flow one."* Then report 180° as the exception.
- **H3 now has direct support.** The parametric network's drop errors reach 0.68% at s = 0.55 and −2.05% at 0.65, yet its optimum is off by only 0.0003, because its error is nearly flat near the optimum: it changes by 0.92% per unit s between 0.45 and 0.50. A.3 predicts a shift of −0.0092/38 = −0.00024 with this geometry's J″/J = 38. The measured −0.00027 is consistent with that, but both lie inside the ±0.002 search tolerance, so present it as consistent, not as a precise confirmation. Network 3 shows the same effect: its drop prediction errs by 0.48% on average, yet its splits cost only 0.016% extra.

### C5. The contributions predate three results

Replacement for the second and third bullets:

> • Three linked networks, each scored against the reference (Section 6). Network 1 resolves the parent channel to 3 × 10⁻⁶ in velocity and 4 × 10⁻⁶ in its pressure-drop objective (mean of three seeds), about four and sixty times the reference's own error. Network 2 resolves the bifurcation with mass and shares exact. Trained on one geometry, it predicts the pressure drop to within 0.4%. Trained once over a range of width splits, with the split as an input and no flow data, it locates the optimal split to within 0.0003 of the finite-element optimum, at a cost below 0.01%, as Appendix A.3 predicts from its error slope. Network 3, trained on the 165 optima and tested on 18 unseen cases, proposes splits whose pressure drop is within 0.016% of the optimum on average (0.077% at worst), about seven times closer than the Murray split.
> • An equal-accuracy account of cost (Section 7). A physics-informed network does not repay its training for one design question: the parametric Network 2 needs 215 min against 11 min of finite elements. Network 3 answers in milliseconds, after about 5 h of finite-element data and fitting. A label-free energy-consistency check is reported for every network (Appendix A.6), alongside metrics that separate field accuracy from design accuracy: argmin error and design regret.

Add to the first bullet: *"Relaxing w₁ + w₂ = w₀, the volume-optimal total width stays within 3% of the planar square law (Section 5.6)."*

---

## Major

**M1. Length and duplication.** The introduction is about 2,000 words. Since the literature review was added, it repeats most of the review's paper-by-paper content: Emerson's Reynolds-number figures, Bejan, topology optimisation, Wechsatol, Grossmann, Chuang and Barba, and McGreivy and Hakim. Your note asks for ¾–1¾ pages. `results.md` includes a condensed introduction, version 6, of about 1,050 words. It keeps the hook, one paragraph per gap, the aim, the questions, the hypotheses and the contributions, and moves the detail to Section 2.

**M2. The friction law is not a lower bound.** At equal shares the resolved optimum costs more than the friction estimate of 7.0 in 12 layouts, by 1.5–29%, but less in the three most strongly turned: 0/120 (−1.6%), 30/120 (−4.2%) and 0/150 (−13.3%). A reader of A.5, where the junction only adds loss, would not expect this. Report it in Section 5, and do not attribute it until it is understood. The likely candidates are the daughters' effective lengths, since L_d is measured along the outer wall, and the junction region's extra flow area.

**M3. Network 1's role must match what was done.** It "establishes how far upstream of the junction the inflow is developed". It does: the 99% entrance length is 1.78 half-widths at Re_h = 10 (1.7795 in all three seeds; reference 1.7794). But every bifurcation result uses the developed parabola directly, with a parent of 3 half-widths. Say so: *"…so that the developed inflow imposed at the bifurcation's inlet is accurate whenever the junction lies more than 1.8 half-widths downstream of a nearly uniform entry."* The coupling option (`inlet.profile: network1`) exists but was not used in the reported results.

**M4. Two versions of Appendix A exist.** Yours is more precise: the exact D̂″/D̂ = 12/[s(1 − s)] in A.3, the Emerson and Hagmeijer locations in A.4, and the general condition in A.5. The one in `methods.md` adds the parent-corrected curvature (43.9 by friction law, 44.1 by finite elements, against your upper bound of 48), the measured A.5 pattern, and the caveat that A.6's identity holds only up to opening viscous terms. Keep your text and add those three, plus the measured energy inconsistencies in A.6.

**M5. The evidence-check appendix contradicts the paper.**
- "What this means for your novelty claim" says: *"Your novelty is how the optimum moves with Reynolds number."* Version 5's novelty is angles and shares at fixed Re, with Re as the Section 8 extension.
- The CARS table's Step 2 row still lists "the optimum has never been re-found at finite Re".
- "Still to fill in" lists the headline result "once the finite-element sweep is done" (it is done) and gives a length of "about 1,100 words" (it is about 2,000).

Update or delete these. The whole evidence check and the revision notes should move to a working file before submission.

**M6. Unverified claims in the literature review.** Read these against the sources before submission:
- Hennigh et al. (2021) and Gammaidoni et al. (2026) adding integral flow-rate terms to the loss;
- Hagmeijer and Venner's claim that the minimum-power and minimum-volume laws coincide only for symmetric branching;
- Stephenson et al.'s three equivalent formulations;
- the four references already marked ⚠ (Stephenson & Lockerby, 2016; Krishnapriyan et al., 2021; Li et al., 2021; Sukumar & Srivastava's DOI).

For the missing adjoint shape-optimisation reference, two standard candidates are:
- Othmer, C. (2008). A continuous adjoint formulation for the computation of topological and surface sensitivities of ducted flows. *International Journal for Numerical Methods in Fluids, 58*(8), 861–877.
- Mohammadi, B., & Pironneau, O. (2010). *Applied shape optimization for fluids* (2nd ed.). Oxford University Press.

Read either before citing it.

**M7. The comparison with Emerson et al.'s Re₀.** Your Re_Dh uses D_h = 2w₀, the infinitely deep planar limit. Their square channels have D_h = w. "Just above Re₀ ≤ 25" compares hydraulic-diameter Reynolds numbers across different cross-sections; add "on a hydraulic-diameter basis".

---

## Minor

- **Name what 2D represents** in Section 3: channels much deeper than wide. Your own "still to fill in" list asks for this.
- **"within 0.010 of the Murray split for bifurcation angles up to 90°"** is 0.0098 at 10/80. Write "within 0.01".
- **"0.1–0.2% of it"** is 0.13–0.23%. Write "about 0.1–0.2%".
- **A.3, "δs = 0.01 costs about 0.24%"** is the daughters-only bound. With the parent the figure is 0.22%, and the introduction's "about 0.2%" covers both.
- **A.3's 71.5% equal-width saving** is a daughters-only friction figure, the same at q = 0.05 for every layout. The resolved savings at q = 0.05 range from 67.9% (60/60) to 71.5% (0/150). The agreement at 0/150 is partly coincidence.
- **Units:** state once that J is in units of ρU_c², and keep Re₀ (Emerson's) distinct from Re_h (yours).
- **Hagmeijer and Venner (2018)** is an arXiv preprint; check whether it has since been published.

---

## What was added to the paper

- `docs/paper/methods.md`: renumbered to Sections 3 and 4 to sit after the literature review.
- `docs/paper/results.md`: a title and abstract, the condensed introduction (version 6), and Sections 5–8 with every number from the repository.
- `bifurcation/energy.py`, with its results in `docs/bifurcation_re10-energy.md`: the label-free energy check the introduction promises, run on all four Network 2 runs.
