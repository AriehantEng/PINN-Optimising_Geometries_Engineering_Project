# Title, Abstract, Introduction (version 6, condensed), and Sections 5–8

[Continuation of the draft, numbered to follow the Literature Review (Section 2) and `methods.md` (Sections 3–4). Every number comes from the repository: the source file is named in brackets, to be deleted before submission. ⟨Energy⟩ marks the numbers from `docs/bifurcation_re10-energy.md`.]

---

## Title (options)

1. **Beyond Murray's split: resolved optimal widths of planar channel bifurcations, and what neural networks can take over from finite elements**
2. Mapping the departure from Murray's law in planar bifurcations with finite elements, physics-informed networks and learned design rules
3. How branch angles and flow shares move the optimal width split of a planar bifurcation

## Abstract

Murray's law sizes the branches of a fluid network as if every channel carried fully developed flow and every junction were lossless. We map how the pressure-drop-optimal widths of a planar bifurcation depart from this rule when the junction flow is resolved. The study covers 165 finite-element optima over fifteen layouts, with bifurcation angles from 30° to 180° and prescribed flow shares from 5% to 95%, at a channel Reynolds number of 10 and with the daughters sharing the parent's width.

The optimal split stays within 0.01 of the planar Murray split for bifurcation angles up to 90° and at 180°. It departs by up to 0.072 at 150°, and in every case the direction follows a junction-loss model. The daughter with the larger junction loss is widened: the more-turned daughter, or, when both turn equally, the one carrying more flow. Optimal widths lower the pressure drop by up to 72% relative to equal widths. When the total width is also priced, the resolved optimum stays within 3% of the planar square law.

We then test three neural networks as substitutes for the solver:
- **Network 1**, a physics-informed network for the parent channel, matches the reference to 3 × 10⁻⁶.
- **Network 2**, a physics-informed network with the width split as an input and no flow data, locates the optimal split to within 0.0003. Its design error follows the slope of its pressure-drop error, not the error's size. It costs 19 times the finite-element search for one design question.
- **Network 3**, trained on the finite-element optima, proposes splits within 0.016% of the optimal pressure drop in milliseconds, and repays its data and training after about 900 design queries.

A reference-free energy check flags the failed network, but it overstates the errors of the good ones by a factor of three to ten ⟨Energy⟩.

[About 270 words. Section 8 keeps the extension to higher Reynolds number, so the abstract does not promise it.]

---

## 1. Introduction (version 6, condensed)

[About 1,050 words, against version 5's 2,000. The paper-by-paper detail now lives in Section 2. Citations are unchanged from version 5, and the cross-references follow the new numbering.]

Branching channel networks are among the most widespread structures in engineering and biology, from the vascular system to the manifolds that distribute fluid across a lab-on-a-chip (Emerson et al., 2006; Murray, 1926b). The classical rule for sizing them is Murray's: minimising the combined cost of pumping and of the volume a vessel holds fixes the size of every daughter relative to its parent (Murray, 1926b; Stephenson et al., 2015). The rule has since been generalised to the constant-depth channels of microfabricated devices (Emerson et al., 2006; Stephenson et al., 2015). For planar channels it becomes a square law, w₀² = w₁² + w₂² (Stephenson et al., 2015; Zhou et al., 2024), and it is used to design manifolds and porous materials (Zhou et al., 2024). Its relevance has grown with inertial microfluidics, which deliberately operates at channel Reynolds numbers between about 1 and 100 (Zhang et al., 2016). In parallel, physics-informed neural networks (PINNs) have become a mesh-free way to solve the flow equations (Raissi et al., 2019), and making a geometric parameter a network input lets one trained network represent a family of designs (Gammaidoni et al., 2026; L. Sun et al., 2020).

Both lines of work leave a gap. Murray-type rules treat every channel as fully developed and every junction as lossless (Hagmeijer & Venner, 2018; Stephenson et al., 2015), so the branch angles can enter the optimal widths only through the channel lengths they imply (Appendix A.2). Where the junction flow has been resolved, it has been for particular layouts or for a single optimal design (Section 2.2). To our knowledge, no study has mapped how the departure of the optimal widths from Murray's rule depends on the branch angles and on the share of the flow each daughter must carry. These are the variables a manifold designer usually cannot change; the width split is the one left to choose.

On the tools side, PINNs have not outperformed finite elements as forward solvers in the benchmarks tested (Grossmann et al., 2024). Most claims of beating numerical solvers rest on weak baselines (McGreivy & Hakim, 2024), and an optimum predicted by a PINN has to be checked with an independent solve (Lu et al., 2021; Mowlavi & Nabi, 2023). Inside a design loop the test is also different. Near an optimum the objective is flat (Hagmeijer & Venner, 2018), so a design can be nearly optimal in cost while visibly wrong in geometry, and the predicted minimum moves with the slope of the network's error rather than its size (Appendix A.3).

We address both gaps with one design problem that has a known answer in one limit (Section 3). A parent channel splits into two daughters of equal length, leaving at angles α₁ and α₂ anywhere from 0° to 180°. The daughters share the parent's width, w₁ + w₂ = w₀. For equal lengths this fixes their total volume, so the optimal split is the volume-constrained one; Section 5.6 relaxes the constraint. Each daughter must deliver a prescribed share of the flow, q for daughter 1, because with equal outlet pressures the optimum simply closes one branch (Appendix A.1). The only design variable is the width split s = w₁/w₀, and the objective is the flux-weighted total-pressure drop, the pumping power per unit flow (Appendix A.6). In the fully developed limit the optimum is the planar Murray split, s_M = √q/(√q + √(1 − q)) (Appendix A.2), which serves both as the ground truth to depart from and as the baseline any network must beat. We work at Re_h = 10 (Re_Dh ≈ 27 on a hydraulic-diameter basis, Appendix A.7), so the departures come from the junction geometry at a fixed, moderate level of inertia.

The approach links a finite-element reference (Section 4.1) with three networks (Sections 4.3–4.5):
- **Network 1** is a PINN for the developing flow in the parent channel.
- **Network 2** is a PINN for the bifurcation itself, built so that no-slip, mass conservation and the prescribed shares hold exactly, since a penalty on continuity lost 6.3% of the mass (Section 6.2). In its parametric form, the width split is an input.
- **Network 3** is not physics-informed. It learns, from the finite-element optima, a correction to the Murray split and to its pressure drop.

This research addresses two questions:

1. How far do the pressure-drop-optimal widths of a planar bifurcation depart from the Murray split when the junction flow is resolved, across flow shares, bifurcation angles and orientations?
2. Can neural networks replace the finite-element solver in this design loop at equal accuracy and lower cost, and which parts of the loop can they take over?

[State in Section 3 whether these hypotheses were fixed before the results or refined after the first results; see review C4.]
- **H1.** Where both daughters turn by similar amounts, the optimal split stays within 0.01 of s_M.
- **H2.** The optimum widens the daughter with the larger junction loss: the more-turned daughter, and at equal turning the one carrying more flow (Appendix A.5).
- **H3.** A network's error in the optimal split is governed by the slope of its predicted pressure-drop error, not by the error's size (Appendix A.3).

This study makes three contributions:
- **A resolved design map at Re_h = 10** (Section 5): 165 finite-element optima over 15 layouts. The optimal split stays within 0.01 of s_M up to 90° and at 180°. It departs by up to 0.072 at 150°, in the direction the junction-loss model predicts. Optimal widths lower the pressure drop by up to 72% relative to equal widths, and the orientation of a fixed bifurcation angle changes it by up to 23% at equal shares and by a factor of 2.7 when most of the flow must turn back. When the total width is also priced, the resolved optimum stays within 3% of the planar square law.
- **Three networks scored against the reference** (Section 6):
  - Network 1 reaches 3 × 10⁻⁶ in velocity.
  - The parametric Network 2, trained with no flow data, locates the optimal split to within 0.0003, as its error slope predicts.
  - Network 3 proposes splits within 0.016% of the optimal pressure drop on 18 unseen cases, seven times closer than the Murray split.
- **An equal-accuracy account of cost** (Section 7), following McGreivy and Hakim (2024). It includes a reference-free energy check for every network, and design metrics, argmin error and regret, that separate the accuracy of a design from that of a flow field.

Section 2 reviews the literature, Section 3 formulates the problem, and Section 4 describes the methods. Sections 5–7 answer the two questions and test H1–H3, and Section 8 concludes with the extension to higher Reynolds numbers.

---

## 5. The resolved design map

[Data: `docs/bifurcation_re10-shares-all.json`, `docs/bifurcation_re10-width-rules.md`.]

### 5.1 What the map contains

The map has 180 optimisations: 15 layouts × 11 prescribed shares, plus the free split for each layout. All 180 minima survive refinement to a mesh whose elements have a quarter of the area. The change in the optimal pressure drop on refinement has a median of 1.1 × 10⁻⁴, a 95th percentile of 1.3 × 10⁻³ and a maximum of 1.0 × 10⁻² (30/150 at q = 0.05). The delivered share equals the prescribed one to 3.3 × 10⁻¹⁶, and no set-share optimum lies on a width bound. Across the map, the optimal pressure drop J* ranges from 3.78 to 12.15, in units of ρU_c².

### 5.2 How far the optimum departs from the Murray split (RQ1, H1)

**Table 5.1.** Largest departure |s* − s_M| over the 11 shares, by layout.

| bifurcation angle | symmetric | mildly asymmetric | strongly asymmetric |
|---|---|---|---|
| 30° | 15/15: 0.002 | | 0/30: 0.004 |
| 60° | | 20/40: 0.004 | 0/60: 0.007 |
| 90° | 45/45: 0.003 | | 10/80: 0.010 |
| 120° | 60/60: 0.005 | 30/90: 0.013 | 0/120: 0.023 |
| 150° | 75/75: 0.013 | 30/120: 0.043 | 0/150: **0.072** |
| 180° | 90/90: 0.001 | 60/120: 0.003 | 30/150: 0.006 |

Two variables control the departure: the bifurcation angle, and how unequally the two daughters turn. Up to 90°, every layout stays within 0.010 of s_M. At 120° and 150° the departure grows with the asymmetry, to 0.072 at 0/150, where one daughter continues straight and the other turns back by 150°. At 180° the departure collapses again, to at most 0.006, in all three orientations.

H1 in its quantitative form (within 0.01 when both daughters turn by similar amounts) holds for every symmetric layout up to 120° and at 180°. It is narrowly exceeded at 150°, where 75/75 departs by 0.013 at q = 0.95. Since a split error of 0.01 costs about 0.2% in pressure drop (Appendix A.3), the Murray split is a good design rule wherever the turns are similar, at every share.

### 5.3 Which way the optimum moves (H2)

Figure 5.1 (`docs/paper/fig5_1_departure.png`) plots s* − s_M against q for every layout, one panel per bifurcation angle. It shows a sign pattern that a single mechanism explains.

**Figure 5.1.** Departure of the resolved optimal split from the Murray split against daughter 1's share, by bifurcation angle; layouts coloured by role. Each panel has its own vertical scale. The grey band is the search tolerance (±0.001). Negative values mean daughter 1 is narrower than the Murray split; in the asymmetric layouts daughter 2 is the more-turned one.
- **In the asymmetric layouts below 180°, the more-turned daughter is wider than s_M.** This holds at every share in 0/120, 30/120 and 0/150, and at most shares elsewhere. The direction reverses only when most of the flow must go down the straighter daughter: 20/40 for q ≥ 0.7, and 0/30 and 30/90 at q = 0.95.
- **In the symmetric layouts the higher-flow daughter is wider.** At 15/15, 45/45, 60/60 and 75/75, s* − s_M is negative for q ≤ 0.3 and positive for q ≥ 0.8 (within ±0.001 in between). The largest departure grows with the bifurcation angle: 0.002, 0.003, 0.005 and 0.013.

Both patterns follow from Appendix A.5. If each daughter's junction loss is Kᵢ times its dynamic pressure, the optimum moves toward a wider daughter 1 exactly when K₁s_M³ > K₂(1 − s_M)³:
- with K₁ = K₂ (symmetric turns), this is s_M > ½, i.e. q > ½: the higher-flow daughter widens;
- with K₂ > K₁ (daughter 2 turns more), daughter 2 widens unless daughter 1 carries enough of the flow for s_M³/(1 − s_M)³ to exceed K₂/K₁, which is the reversal seen at high q.

H2, stated as "the optimum widens the daughter with the larger junction loss", is therefore supported below 180°. At equal shares the asymmetric optima range from 0.4977 (0/30) to 0.4437 (0/150), against s_M = 0.5. At 180°, where the junction ends in a flat far wall rather than a septum tip (Section 3.1), neither effect appears. Every departure is within ±0.0015 except where the less-turned daughter carries little flow. There it is slightly wider: +0.0036 to +0.0059 at 30/150 for q ≤ 0.2, and +0.0025 at 60/120 for q = 0.05. This is the opposite sign to the pattern below 180°.

[The 180° result is unexplained. One candidate is that at a flat far wall the stagnation loss is set by the parent's dynamic pressure, which no split changes. It is a hypothesis, not tested here.]

A second finding qualifies the junction-loss picture. At equal shares the resolved optimum costs more than the friction-law estimate J_M = 7.0 in twelve layouts, by 1.5–29%. In the three most strongly turned layouts it costs less: 0/120 by 1.6%, 30/120 by 4.2% and 0/150 by 13.3%. The friction law is therefore not a lower bound. Daughter length is measured along the outer wall, and in turned-back layouts the junction region adds flow area; how much each contributes has not been separated.

### 5.4 What optimisation saves, and what orientation costs

Choosing the optimal split instead of equal widths lowers the pressure drop by up to 71.5% (0/150, q = 0.05). The saving grows as the shares become unequal, and at equal shares it is small: 4.8% at 0/150, 1.7% at 30/120, 0.7% at 0/120, and below 0.002% in symmetric layouts.

Orientation matters more than the bifurcation angle alone suggests (Table 5.2). Up to 90°, layouts with the same bifurcation angle differ by at most 3.3% in J*. At 150° they differ by up to 23%. At 180° they differ by up to a factor of 2.7: when 95% of the flow must turn back by 150° (30/150, q = 0.05), J* = 12.15, against 4.56 when it turns by 90° (90/90).

**Table 5.2.** Spread of J* between layouts with the same bifurcation angle.

| bifurcation angle | 30° | 60° | 90° | 120° | 150° | 180° |
|---|---|---|---|---|---|---|
| at q = 0.5 | 1.0% | 1.8% | 2.1% | 7.0% | 22.6% | 20.3% |
| largest over q | 1.5% | 2.4% | 3.3% | 7.4% | 23.1% | ×2.7 |

For a fixed total flow, an equal split costs the most to deliver in a symmetric layout: at 60/60, J* is 7.37 at q = 0.5, 6.12 at 0.2 and 4.16 at 0.05. In asymmetric layouts the cheapest shares favour the straighter daughter.

### 5.5 Why the shares must be prescribed

With equal outlet pressures (no share set), the lowest pressure drop in every layout is reached at the most unequal split the geometry allows, J = 2.76–3.50. The narrow daughter then carries 0.13–0.23% of the flow (Appendix A.1). At 30/150 that bound is 0.899 rather than 0.9, because daughter 2's entrance gap reaches w_min first. Left free, the optimum is not a bifurcation.

### 5.6 Must the daughters share the parent's width?

w₁ + w₂ = w₀ is a constraint, not a result. Pressure drop alone has no optimum in total width: at 60/60 and q = 0.5, J falls from 7.37 at w₀ to 3.06 at 1.41 w₀ and 1.57 at 2 w₀. Murray's principle prices the width: cost = pumping power + b × channel area, with b set so that the parent's own width is optimal for its flow, b = 36νQ₀²/w₀⁴. Under this cost the friction law predicts an optimal combined width of (√q + √(1 − q)) w₀, the planar square law. Scanning the total width at the Murray proportions by finite elements (Table 5.3) puts the resolved optimum within 3% of it.

**Table 5.3.** Murray cost at q = 0.5, with the daughters' widths scaled together.

| combined width / w₀ | 1.0 | 1.2 | 1.3 | 1.414 | 1.5 | 1.6 | 2.0 | optimum |
|---|---|---|---|---|---|---|---|---|
| 60/60 | 19.09 | 16.71 | 16.28 | **16.15** | 16.22 | 16.42 | 17.98 | 1.42–1.43 |
| 0/150 | 18.79 | 17.03 | 16.77 | **16.75** | 16.87 | 17.12 | 18.73 | 1.37–1.38 |

Measured by this cost, the square law beats w₁ + w₂ = w₀ by 10.6–15.4% at q = 0.5 and by 5.2–10.6% at q = 0.2, across five layouts. The round-pipe cube law, w₀³ = w₁³ + w₂³, makes planar daughters too wide and is 1.3–2.8% dearer than the square law. The design rule is therefore:
- with the footprint fixed, keep w₁ + w₂ = w₀ and split as in Sections 5.2–5.3;
- if width can be traded against pumping power, size the total by the square law and split it the same way.

### 5.7 Wall shear at the optimum

At the Murray split both daughters carry the same wall shear (Appendix A.4); the resolved optimum does not. Table 5.4 gives the largest mismatch at the optimum, max(τ₁/τ₂, τ₂/τ₁) − 1, over the 11 shares. It is always largest at the most unequal shares, where d ln(τ₁/τ₂)/ds = −2/[s(1 − s)] is steepest.

**Table 5.4.** Largest wall-shear mismatch at the resolved optimum.

| bifurcation angle | layouts (largest mismatch) |
|---|---|
| ≤ 90° | 15/15 3%, 0/30 5%, 20/40 5%, 0/60 8%, 45/45 4%, 10/80 **12%** |
| 120° | 60/60 7%, 30/90 18%, 0/120 31% |
| 150° | 75/75 20%, 30/120 71%, 0/150 **×2.6** |
| 180° | 90/90 2%, 60/120 3%, 30/150 8% |

---

## 6. The networks as solvers and optimisers (RQ2, H3)

### 6.1 Network 1: the parent channel

[Data: `docs/developing_re10-summary.md`.]

The best setup combines several elements:
- Levenberg–Marquardt with fresh points at every step;
- an exponential input map;
- a final polish with 16,384 points per step and iterate averaging.

Over three seeds it agrees with the reference to the precision shown in Table 6.1. The baseline (Adam and L-BFGS only) reached 2.83 × 10⁻⁴ in u and 1.36 × 10⁻⁴ in the objective, so the final setup improves on it 87-fold and 35-fold. The 99% entrance length is 1.7795 half-widths in all three seeds, against 1.7794 for the reference. A developed inflow is therefore accurate wherever the junction lies more than about 1.8 half-widths downstream of a nearly uniform entry. Every bifurcation result imposes that developed profile directly.

**Table 6.1.** Network 1 against the channel reference (relative L2 errors; mean and range over three seeds).

| | u | v | p | total-pressure drop | energy budget, network alone |
|---|---|---|---|---|---|
| Network 1 | 3.3 × 10⁻⁶ (2.7–4.0) | 7.1 × 10⁻⁵ | 1.7 × 10⁻⁵ | 3.9 × 10⁻⁶ (0.55–7.2) | 2.6 × 10⁻⁶ |
| reference's own error | 8.7 × 10⁻⁷ | 9.8 × 10⁻⁶ | 1.5 × 10⁻⁵ | 6.7 × 10⁻⁸ | 1.5 × 10⁻⁷ |

### 6.2 Network 2 on one geometry

[Data: `docs/bifurcation_re10-network2.md`.]

The geometry is 30/90 at s = 0.5, with daughters of length 4 and q = 0.5. The finite-element reference has 28,399 triangles and takes 37 s.

**Table 6.2.** The three formulations of Network 2.

| | velocity–pressure | stream function | + walls extended |
|---|---|---|---|
| u, v, p (rel. L2) | 2.5e-2, 1.1e-1, 1.5e-1 | 1.9e-2, 2.6e-2, 5.2e-3 | 9.0e-3, 1.3e-2, 6.3e-3 |
| pressure-drop error | −4.5% | −0.40% | +0.42% |
| mass imbalance | −6.3% | exact | exact |
| outlet 1's pressure offset (reference 0.542) | −0.023 | 0.555 | 0.556 |
| training | 70 min | 101 min | 90 min |

Imposing continuity as a penalty lost 6.3% of the mass, and with it the prescribed split. The stream function makes both exact by construction. The two stream-function runs bracket the reference drop, at −0.40% and +0.42%, which is the run-to-run band. Halving the field errors did not narrow it.

### 6.3 A reference-free check

[Data: `docs/bifurcation_re10-energy.md`.]

Network 2 conserves mass and its shares exactly, so those cannot reveal its errors. The energy balance, |Q₀J − Φ|/Φ from the network's own field, still can (Appendix A.6). Table 6.3 compares it with the true errors.

**Table 6.3.** ⟨Energy table, filled from the run.⟩

⟨Energy findings.⟩

### 6.4 Network 2, parametric: one network for every split

[Data: `docs/bifurcation_re10-parametric.md`.]

Trained once over s ∈ [0.35, 0.65] at 30/90 and q = 0.5, with no flow data, the network predicts the pressure drop:
- to within 0.06–0.11% at s = 0.45–0.50;
- with errors of 0.2–0.3% below that range;
- with errors growing to 1.4% at s = 0.60 and −2.1% at s = 0.65.

Its field errors are 1.3–1.9% in u. At s = 0.5 its pressure-drop error, +0.11%, is smaller than either fixed-geometry run's, while its velocity errors are about 40% larger.

**Table 6.4.** The optimal split.

| | s* | J* |
|---|---|---|
| parametric Network 2 | 0.4845 | 4.0385 |
| finite elements | 0.4848 | 4.0358 |
| difference | −0.0003 | +0.07% |

The argmin error is within the ±0.002 search tolerance, and the regret is below 0.01%. This is the test H3 was built for. The network's drop errors reach 0.68% at s = 0.55 and −2.05% at 0.65, yet they did not move the optimum, because near it the error is nearly flat. Between s = 0.45 and 0.50 it changes by 0.92% per unit of s, and with this geometry's J″/J = 38, Appendix A.3 predicts an argmin shift of −0.0092/38 = −0.00024. The measured −0.00027 agrees, though both lie well inside the ±0.002 tolerance of the two searches, so the agreement is consistent with A.3 rather than a precise confirmation of it.

### 6.5 Network 3: learned design rules

[Data: `docs/network3_re10_all_w16.md`.]

Trained on the 165 optima, Network 3 was tested on six angle pairs it never saw, at three shares each. Every case was optimised separately by finite elements.
- **Regret:** its proposed splits cost 0.016% more than the optimum on average and 0.077% at worst. The Murray split costs 0.109% on average and 0.70% at worst.
- **Drop prediction:** it predicts the lowest pressure drop to within 0.48% on average and 3.0% at worst.

As H3 predicts, the design is more accurate than the predicted cost: a 0.5% error in J* comes with splits that cost only 0.016% extra.

Leave-one-layout-out cross-validation shows where the network stops being reliable. For layouts inside the sampled range, its held-out split errors are at most 0.009, and better than the Murray split for most layouts. For the layouts at the edge of the angle space (0/120, 0/150, 30/120, 30/150 and 60/120), the held-out drop errors reach 26–49%. Network 3 interpolates between angle pairs; it should not be trusted to extrapolate beyond them.

---

## 7. Cost at equal accuracy, and the energy–shear trade-off

### 7.1 What each route costs

[Data: the runs' metrics files; Network 3 query timed on 4 cores.]

All timings are on four CPU cores.

**Table 7.1.** Cost of each route.

| task | finite elements | network |
|---|---|---|
| channel flow (Network 1) | 283 s (finest of four meshes) | 263 min of training |
| one bifurcation (Network 2) | 37 s | 90 min |
| one design question: best split at fixed angles and share | 11.4 min (14 solves); 76 s on the design study's coarse mesh | 215 min (parametric Network 2) |
| a map of design questions (Network 3) | 76 s per question | 3.9 core-hours of finite-element data plus a 3.8 h fit, then 0.02 s per question, or 13 µs each batched |

At equal accuracy the physics-informed networks do not compete. Each is more expensive than the solve it replaces, by 56 times for the channel, 150 times for one bifurcation and 19 times for one design question. A finite-element solve is also more accurate than any of them. A parametric network that answered every (angle, share) question would have to cost less than about 19 separate optimisations to break even, which this one, covering a single layout and share, does not.

Network 3 does compete, because its training data are finite-element results and it replaces the search, not the solve. Its 18 validation designs are within 0.016% of optimal on average. That is comparable to the coarse-mesh search's own noise of about 0.01%. With about 19 core-hours of data and fitting, it repays itself after about 900 new design queries at 76 s each. [About 15 core-hours of the fit is cross-validation used to choose its regularisation.] Its domain is the one it was trained on: Re_h = 10, these lengths, and angle pairs inside the sampled range (Section 6.5).

### 7.2 The energy–shear trade-off

The optimal widths do not equalise wall shear (Table 5.4). Up to 90° the mismatch is at most 12%, while the split stays within 0.01 of s_M, because at small shares the shear ratio is very sensitive to the split (Appendix A.4). Where one daughter turns back, the optimum trades shear for pressure drop: at 0/150 and q = 0.05 the turned daughter carries 2.6 times the other's shear. A designer of a shear-sensitive device (Emerson et al., 2006) therefore chooses between minimum pumping power, at the resolved optimum, and equal shear, at s_M. Network 3 makes the first of these available in milliseconds, and the second is a closed form.

---

## 8. Conclusions

**RQ1.** With the junction resolved at Re_h = 10, the planar Murray split remains a good design rule wherever the daughters turn by similar amounts, to within 0.01 at every share (H1, with 75/75 at 150° the narrow exception). It fails where one daughter turns much more than the other, by up to 0.072 at 0/150. The direction of the failure follows one junction-loss model, which widens the daughter with the larger junction loss: the more-turned daughter or, at equal turning, the higher-flow one (H2, below 180°). At 180° the departure vanishes, unexplained. Relaxing the width constraint, the resolved volume-optimal total width stays within 3% of the planar square law.

**RQ2.** Neural networks can take over the search, not the solve. Physics-informed networks reproduced the flow to 3 × 10⁻⁶ in a channel and 1% in a bifurcation, with mass and shares exact, but at 19–150 times the cost of the finite-element solves they replace. Trained with the split as an input, a PINN located the optimum to within the search tolerance despite pressure-drop errors of up to 2% elsewhere in its range, because its error was flat near the optimum (H3). A small network trained on finite-element optima answers design questions in milliseconds, at the accuracy of the search that produced its data, and repays its cost after about 900 queries inside its training range.

**Limitations.**
- **Flow model:** two-dimensional flow at a single Reynolds number, with daughters of equal length.
- **Search mesh:** the argmin was located on the coarse mesh, and only the value of the optimum was checked on the finer one.
- **Open mechanisms:** the 180° result and the sub-friction pressure drops of the turned-back layouts are not explained.
- **Scope of the network results:** a single parametric PINN run and a single Network 3 configuration.

**Extension.** Inertia grows with the flow rate, and Emerson et al. (2006) found Murray-based predictions failing by 6% at Re₀ = 50 and 30% at 250. The next step is the same map at Re_h = 25, 50 and 100. Each Reynolds number costs about one hour of finite elements, and Network 3 can take Re as a fourth input.
