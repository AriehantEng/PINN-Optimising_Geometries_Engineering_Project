# 3. Problem formulation, 4. Methods, and Appendix A

[Version 2 · Renumbered to follow the Literature Review (Section 2). Companion to `introduction.md` (version 5): same notation, APA 7th edition author–date citations. Every number is taken from the repository: the code it describes is named beside it, and results files are under `docs/`. Bracketed notes are for the authors; delete them before submission. Citations new to the paper are listed at the end and must be read before submission, as for the introduction.]

---

## 3. Problem formulation

### 3.1 Geometry

A straight parent channel of width w₀ divides into two straight daughter channels (Figure 1). All lengths are scaled by the parent's half-width h = w₀/2, so the parent occupies 0 ≤ x ≤ L_p, −1 ≤ y ≤ 1, and w₀ = 2. Daughter 1 leaves the parent's upper corner C = (L_p, 1) at an angle α₁ above the parent's axis, and daughter 2 leaves the lower corner B = (L_p, −1) at α₂ below it. An angle of 0° continues straight on and an angle above 90° turns back. Each daughter is a strip of uniform width wᵢ. Its outer wall runs from its corner along the unit direction d₁ = (cos α₁, sin α₁) or d₂ = (cos α₂, −sin α₂) for a length L_d. Its inner wall runs parallel at a distance wᵢ. Its outlet closes the strip perpendicular to it. The two inner walls meet at the septum tip T, which lies off the centreline whenever w₁ ≠ w₂. The fluid domain is therefore the polygon A B E₂ₒ E₂ᵢ T E₁ᵢ E₁ₒ C D, with A = (0, −1) and D = (0, 1), and the edges are labelled inlet (D–A), outlet 1 (E₁ᵢ–E₁ₒ), outlet 2 (E₂ₒ–E₂ᵢ) and wall (all others) (`bifurcation/geometry.py`).

When α₁ + α₂ = 180° the inner walls are parallel and do not meet. The junction is then closed by a flat far wall joining the points where each inner wall starts, and the polygon has ten vertices (the "face" septum). This case includes the T-junction (90°, 90°). Placing the tip at one of those points instead, as an earlier version did, bent daughter 2's inner wall by up to 5.9°; the face construction removes that distortion.

The study fixes L_p = 3 and L_d = 8 (1.5 w₀ and 4 w₀). The daughters share the parent's width,

  w₁ + w₂ = w₀,  s = w₁/w₀,

so the width split s is the single design variable. Neither daughter may be narrower than w_min = 0.2 (a tenth of the parent), so s ∈ [0.1, 0.9]. The same minimum applies to the opening through which each daughter is entered, measured from the parent's corner (C or B) to the septum. This rule matters for daughters turned far back: at (10°, 170°) the entrance gap at B is 0.17 w₂.

A geometry is refused, never silently altered, when any of the following holds:
- an angle lies outside [0°, 180°), or both angles are zero (a splitter plate of zero thickness);
- a width or an entrance gap is below w_min;
- the daughters overlap after the junction, or the septum tip lies upstream of x = L_p;
- the inner walls meet beyond an outlet;
- the polygon intersects itself.

The vertices are written with interchangeable array back-ends (NumPy or JAX) and without branching on values, so every boundary position is differentiable with respect to w₁, w₂, α₁ and α₂ (Section 4.4.5).

[Figure 1: the polygon with A, B, C, D, T, E₁ᵢ, E₁ₒ, E₂ᵢ, E₂ₒ labelled, α₁ and α₂ marked from the axis, w₁, w₂, L_p, L_d dimensioned; inset of the face septum at α₁ + α₂ = 180°.]

### 3.2 Governing equations

The flow is steady, incompressible, Newtonian and two-dimensional. With velocities scaled by the parent's inlet centreline speed U_c and pressure by ρU_c², the equations are

  (u·∇)u + ∇p − (1/Re_h) ∇²u = 0,  ∇·u = 0,

with Re_h = U_c h/ν = 10 throughout. The Reynolds number based on hydraulic diameter is Re_Dh = (8/3) Re_h ≈ 27 (Appendix A.7). The viscous term is written in Laplacian form. The natural boundary condition of its weak form is then ν ∂u/∂n − p n = 0, which is the do-nothing outlet condition used below (ν = 1/Re_h).

### 3.3 Boundary conditions

- **Inlet** (x = 0): the developed profile u = 1 − y², v = 0, which carries Q₀ = 4/3. Network 1 shows how far downstream of a nearly flat inlet this holds. At Re_h = 10 the centreline reaches 99% of its developed value 1.78 half-widths from the inlet (Section 4.3).
- **Walls**: no-slip, u = v = 0.
- **Outlet 2**: do-nothing, ν ∂u/∂n − p n = 0.
- **Outlet 1**: ν ∂u/∂n − p n = −λ n, where the constant λ is chosen so that outlet 1 carries the prescribed share of the flow, Q₁ = q Q₀. This is the flux-defective boundary condition (Formaggia et al., 2002; Heywood et al., 1996): the outlet's flux is prescribed and its pressure follows, λ being outlet 1's pressure above outlet 2's. With λ = 0 both outlets are do-nothing at equal pressure, and the flow chooses its own split. The study includes that case for reference (Section 3.4).

### 3.4 Objective and design problem

The objective is the flux-weighted total-pressure drop

  J = (E_in − E₁ − E₂) / Q₀,  E_k = | ∫_k (p + |u|²/2) u·n ds |,

the mechanical energy lost per unit of flow delivered, in units of ρU_c². For steady incompressible flow with no-slip walls it equals the viscous dissipation per unit flow, up to viscous-flux terms at the openings that vanish for developed flow there (Appendix A.6). Minimising J at fixed Q₀ therefore minimises pumping power.

For each pair of angles (α₁, α₂) and each share q ∈ (0, 1) that daughter 1 must carry, the design problem is

  s*(α₁, α₂, q) = argmin over s ∈ [0.1, 0.9] of J(s; α₁, α₂, q),

subject to every geometry rule of Section 3.1.

The shares must be prescribed. With equal outlet pressures the daughters act as parallel resistances, each conducting like wᵢ³, so J falls monotonically toward the most unequal split allowed (Appendix A.1). The resulting "optimum" closes one daughter down to w_min, where it carries 0.1–0.2% of the flow (Section 5).

### 3.5 The Murray split

When the junction is ignored and each daughter is treated as fully developed plane Poiseuille flow, the design problem has an exact answer. Minimising the daughters' dissipation under w₁ + w₂ = w₀ gives w₁/w₂ = (L₁/L₂)^(1/4) (Q₁/Q₂)^(1/2) (Appendix A.2). For daughters of equal length, as here, this is the planar Murray split

  s_M(q) = √q / (√q + √(1 − q)).

The corresponding drop, including the parent's, is

  J_M(q) = 1.5 ν L_d Q₀ (√q + √(1 − q))⁴ + 12 ν L_p Q₀ / w₀³,

which is 7.0 at q = 0.5. Neither expression contains the angles (Appendix A.2(iii)). Every departure of the resolved optimum from s_M is therefore due to the flow at the junction and in the daughters' entrance regions.

---

## 4. Methods

### 4.1 Finite-element reference

Every network result, and every point of the design dataset, is computed or judged by one finite-element solver (`bifurcation/reference.py`, built on `developing_flow/reference.py`).

**Discretisation.** Taylor–Hood P2/P1 elements are used: continuous piecewise-quadratic velocity and piecewise-linear pressure, which are inf-sup stable (Taylor & Hood, 1973). They are assembled with scikit-fem (Gustafsson & McBain, 2020) on a quality-constrained Delaunay mesh from Triangle (Shewchuk, 1996), with a minimum angle of 30° and a maximum element area A_max. The pressure is singular at the re-entrant corners B, C and T, where the fluid's interior angle is 180° or more. Around these corners the mesh is refined for a set number of rounds: round k refines every element whose centroid lies within r_c/2^k of a re-entrant corner, with r_c = 0.4.

**Boundary data.** Boundary facets are assigned to the polygon edge they lie on, by distance to that edge's segment. The inlet profile is applied only on the inlet segment D–A, not on the line x = 0. A daughter turned back beyond about 112° crosses x = 0 with its outer wall, and an earlier version that used the line applied the inlet velocity to a wall node there, producing a 152% mass error. Each solve now refuses any solution whose wall velocity values are not exactly zero.

**Prescribed share.** The share is the single linear constraint c·U = 0, with c = b₁ + q b_in, where b_k·U is the discrete outflow through opening k. It is enforced by a Lagrange multiplier λ. Because c integrates u·n over outlet 1, λc enters the momentum equation as a uniform extra pressure on that outlet. This is the flux-defective condition of Section 3.3, holding exactly on the discrete solution.

**Nonlinear solve.** A Stokes solve gives the initial iterate. Newton's method follows, with the exact Jacobian of the convective term and a sparse direct solve at each step, until the relative update falls below 10⁻¹³ (at most 30 iterations).

**Measures.** The fluxes, mechanical energy fluxes E_k and mean static pressures are boundary integrals, computed by facet quadrature of order 8. The viscous dissipation Φ = 2ν ∫ e:e dA is used for the energy-consistency check |E_in − E₁ − E₂ − Φ|/Φ (Appendix A.6). When a network is compared with the reference, both are measured by one function: 256-point Gauss–Legendre quadrature on each opening, from point values alone (`flow.opening_measures`).

**Verification.** The underlying solver is verified on quadrilaterals against Poiseuille flow (exact to round-off, since the solution lies in the Q2/Q1 space) and Kovasznay flow, an exact Navier–Stokes solution with nonzero convection. Its velocity converges at third order and its pressure at second, as Taylor–Hood theory predicts (`tests/test_reference.py`).

On the bifurcation, the test suite (`tests/test_bifurcation.py`) requires the following:
- mass conservation to 10⁻¹² (Taylor–Hood conserves mass globally) and Q_in = 4/3 to 10⁻¹²;
- energy consistency below 10⁻³ even on a coarse mesh;
- an even split (to 10⁻⁴) for a mirror-symmetric geometry;
- for a lopsided geometry, more flow through the straighter daughter;
- no change, to 10⁻¹⁰, when the share the flow chose freely is prescribed (so λ = 0);
- delivery to 10⁻¹² of a forced share, paid for by a lower outlet-1 pressure;
- convergence of J under refinement: refining from A_max = 0.02 to 0.005 changes J by less than half as much as refining from 0.08 to 0.02.

The reference used to score Network 2 (A_max = 0.002, six corner rounds; 28,399 triangles, 129,562 unknowns, 37 s per solve) has energy consistency 7.5 × 10⁻⁶.

### 4.2 The design dataset

The design dataset is the set of finite-element optima s*(α₁, α₂, q) and their pressure drops J* (`bifurcation/design_study.py`). It is the ground truth for the first research question and the training data for Network 3.

**Layouts.** Fifteen angle pairs cover bifurcation angles from 30° to 180°. Each bifurcation angle appears in two or three orientations, from symmetric to one daughter continuing straight on:

| Bifurcation angle α₁ + α₂ | Layouts (α₁ / α₂) |
|---|---|
| 30° | 15/15, 0/30 |
| 60° | 20/40, 0/60 |
| 90° | 45/45, 10/80 |
| 120° | 60/60, 30/90, 0/120 |
| 150° | 75/75, 30/120, 0/150 |
| 180° | 90/90, 60/120, 30/150 |

The 120° and 180° layouts were computed first (`bifurcation/study_re10_shares.yaml`) and the others second (`bifurcation/study_re10_angles.yaml`). The two studies share every setting except the angles, and they were merged only after the program checked that the physics, geometry, search and mesh settings were identical (`design_study.merge`). The second set deliberately excludes the six pairs held back to validate Network 3 (45/75, 15/105, 75/105, 45/135, 30/30 and 0/90), so those pairs remain unseen.

**Shares.** q ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}. Each layout also has the case with no share set (equal outlet pressures). This gives 15 × 12 = 180 optimisations: 165 with a set share and 15 equal-pressure floors.

**Search.** Each optimisation follows the same four steps:

1. *Grid scan.* J is computed at 17 splits, s = 0.10, 0.15, …, 0.90, on the coarse mesh (A_max = 0.02, three corner rounds). For 30/90 at s = 0.5 this mesh has 2,852 triangles and 13,492 unknowns. A split that violates a geometry rule counts as J = +∞, and its reason is recorded.
2. *Refinement.* A bounded Brent search (Brent, 1973; SciPy `minimize_scalar`) to a tolerance of 0.001 in s, inside the bracket formed by the grid neighbours of the best grid point. Brent's result is kept only if it is no worse than that grid point. A best grid point at an end of the range is reported as a bound and not refined.
3. *Mesh check.* The optimum and both bracketing grid points are re-solved on a finer mesh: A_max = 0.005 (a quarter of the element area) and five corner rounds, giving 12,837 triangles and 59,217 unknowns for 30/90 at s = 0.5. The minimum is said to survive refinement if the fine-mesh J at the optimum is no larger than at either neighbour. |J_fine − J_coarse|/J_fine is recorded as the mesh's error estimate for the value.
4. *Comparisons.* J is also computed on the coarse mesh at equal widths (s = 0.5) and at the Murray split s_M(q).

**Recorded quantities.** Each row records:
- the angles, the share, s* (with w₁ and w₂), and whether s* lies on a bound;
- J* on both meshes and the mesh change;
- whether the minimum survives refinement;
- J at equal widths and the saving 1 − J*/J(0.5);
- s_M and J(s_M);
- outlet 1's pressure offset λ at the optimum, and the delivered share;
- the full grid scan, every refused split with its reason, the number of solves and the time taken.

**Execution.** The 180 optimisations ran as independent tasks on four worker processes (4 CPU cores), taking 1,529 s and 2,071 s of wall time for the two studies, about one hour in all. They required 4,381 coarse-mesh solves (23–29 per optimisation, median 25) and 526 fine-mesh solves. The median time per optimisation was 76 s (maximum 140 s).

**Data quality.**
- All 180 minima survive refinement.
- The fine-mesh change in J* has median 1.1 × 10⁻⁴, 95th percentile 1.3 × 10⁻³ and maximum 1.0 × 10⁻² (30/150 at q = 0.05, the most strongly turned layout at its most unequal share).
- The delivered share equals the prescribed one to 3.3 × 10⁻¹⁶ in all 165 cases.
- No set-share optimum lies on a width bound.
- All 15 equal-pressure optima lie at the largest or smallest split the geometry rules allow (Appendix A.1). For 30/150 that is s = 0.899 rather than 0.9: at s ≥ 0.899 daughter 2's entrance gap falls below w_min, and those 13 grid and Brent evaluations were refused with that reason recorded.

**Limitation.** The argmin was located on the coarse mesh. The fine mesh confirms that the minimum lies in the same bracket and estimates the error in J*, but s* was not re-located on the fine mesh. Appendix A.3 relates the two: a mesh error in J whose relative value changes by ε across a 0.1 change in s shifts the argmin by about ε/(0.1 × 44). For example, an error that changed by 0.1% across the bracket would shift the argmin by about 2 × 10⁻⁴.

**Data products.**
- `docs/bifurcation_re10-shares-all.json`: every field above, including the grid scans and refusals.
- `docs/bifurcation_re10-shares-all.md`: tables by layout.
- `docs/bifurcation_re10-shares-all.png`: s* and J* against q.
- `docs/bifurcation_re10-optimum-widths.csv`: one row per optimisation.

The two source studies are kept beside the merged file. To reproduce the dataset:

    python -m bifurcation.design_study --config bifurcation/study_re10_shares.yaml --out docs/bifurcation_re10-shares
    python -m bifurcation.design_study --config bifurcation/study_re10_angles.yaml --out docs/bifurcation_re10-angles
    python -m bifurcation.design_study --merge docs/bifurcation_re10-shares.json docs/bifurcation_re10-angles.json --out docs/bifurcation_re10-shares-all

### 4.3 Network 1: developing flow in the parent channel

Network 1 (`developing_flow/pinn.py`) solves the same equations in a straight channel, 0 ≤ x ≤ 10, −1 ≤ y ≤ 1, at Re_h = 10. The inlet is the flat-topped profile u = U₀(1 − y¹⁶), v = 0, with U₀ = (2/3)(2m + 1)/(2m) and m = 8, so the flux is 4/3. This is a uniform inlet rounded off just enough to satisfy no-slip at the corners: a true top hat demands an unbounded network output there (CLAUDE.md, "A top-hat inlet cannot be imposed on hard no-slip walls").

**Architecture.** The network is the modified multilayer perceptron of Wang, Teng and Perdikaris (2021): two input encoders blended into every hidden layer, with tanh activations, width 40 and depth 4 (5,403 parameters), Glorot-normal initialisation (Glorot & Bengio, 2010) and float64 arithmetic throughout. The coordinate x is mapped exponentially, ξ = 2(1 − e^(−x/2))/(1 − e^(−L/2)) − 1, so that the entrance region occupies more of the network's input.

**Constraints.**
- No-slip holds exactly: u = (1 − y²)N_u, v = (1 − y²)N_v.
- The pressure gauge p(L, y) = 0 holds exactly: p = ((L − x)/L)N_p.
- Momentum, continuity, the inlet profile and the outlet condition ∂u/∂x = ∂v/∂x = 0 are least-squares terms of equal weight.

**Training.** Each draw has 4,096 interior points (16,384 in the final polish), half of them concentrated in x < 4 and toward the walls, with 256 inlet and 256 outlet points. The schedule is Adam (Kingma & Ba, 2015) for 3,000 steps, with fresh points every 250, followed by four cycles of 500 L-BFGS iterations (Liu & Nocedal, 1989). Then comes Levenberg–Marquardt (Levenberg, 1944; Marquardt, 1963) with fresh points at every step and the damping updated by Nielsen's rule (Madsen et al., 2004): 800 steps, then a 100-step polish. During the polish the mean of the last 20 iterates is also a candidate (Polyak & Juditsky, 1992).

**Selection and scoring.** The selected network is the one with the lowest loss on held-out points, drawn from an independent random stream and never trained on: 4,096 points before the polish and 32,768 during it, since 4,096 proved too few to rank iterates that differ by less than one draw's sampling noise (CLAUDE.md). A warm-started polish uses a new seed, because its parent's held-out points selected the parent. Three seeds were trained.

The reference is a Q2/Q1 finite-element solution on a tensor mesh graded toward the inlet and walls, solved on four meshes each twice as fine as the last. Its own error is estimated from the finest two and is reported beside every network error (Celik et al., 2008). Errors are integrated on the reference's own mesh by Gauss quadrature in every element. A uniform scoring grid over-weights the inlet corners, where one column held 53.5% of the summed pressure error (CLAUDE.md).

**Entrance length.** This is where the centreline first reaches 0.99 of its developed value (0.75 × the outlet mean velocity). It is located by bracketing on a grid and then Brent's method on the field itself, identically for the network and the reference. Network 1 gives 1.7795 half-widths in all three seeds; the reference gives 1.7794; Durst et al.'s (2005) flat-inlet correlation gives 1.88.

### 4.4 Network 2: the bifurcation PINN

Network 2 (`bifurcation/flow.py`) solves the problem of Section 3 for one geometry, with the share prescribed. It is built so that no-slip, mass conservation and the share hold exactly for any parameters. The equations and the outlet conditions are left to the loss.

**4.4.1 Wall distance.** For each wall segment, φᵢ is the trimmed distance of Sukumar and Srivastava (2022, Section 5.1): zero on the segment and positive everywhere else. The segments are composed by R-equivalence (Rvachev & Sheiko, 1995) of order 1,

  D = (Σᵢ φᵢ⁻¹)⁻¹,

which vanishes on every wall and only there. The walls form three chains, separated by the openings:
- the parent's lower wall with daughter 2's outer wall;
- the septum (E₂ᵢ–T–E₁ᵢ);
- daughter 1's outer wall with the parent's upper wall.

Collinear consecutive edges are merged into one segment. A straight daughter continuing the parent's wall would otherwise leave a 180° joint, at which the composed distance has a Laplacian of order 1/r, so the residual grows like 1/r although the flow is smooth. Each chain is also extended by 0.5 beyond each opening, along its end segment and outside the domain. The stream function needs third derivatives of D, which are large where a segment ends. With the walls ending at the outlets, one point near an outlet corner held 86% of a held-out loss.

**4.4.2 Stream function.** The velocity is the curl of a stream function,

  ψ = G + D² N_ψ,  u = ∂ψ/∂y,  v = −∂ψ/∂x,

with the pressure as a second network output, so continuity holds identically. G blends the constant values ψ = 0 (lower chain), (1 − q)Q₀ (septum) and Q₀ (upper chain) with Shepard-type weights (Shepard, 1968) w_k = Π_{j≠k} D_j^μ, μ = 2. Because ψ is constant along each chain, the flux through outlet 2 is exactly (1 − q)Q₀ and through outlet 1 exactly qQ₀, for any parameters. Because μ ≥ 2 and the correction carries D², the normal derivative of ψ also vanishes on the walls, so u = v = 0 there exactly. Q₀ is computed from the inlet profile by 64-point Gauss–Legendre quadrature.

**4.4.3 Loss.** The loss is the sum of squared, unweighted residuals of four kinds:
- x- and y-momentum at interior points;
- the inlet velocity against the profile;
- the do-nothing traction ν ∂u/∂n − p n at outlet 2;
- the same traction with + λn at outlet 1, where λ is a trained scalar: the network's version of the reference's multiplier.

The four term weights are 1. Near a reflex corner the exact velocity behaves like r^λ with λ < 1 (λ ≈ 0.54 at a 270° corner in Stokes flow; [cite Dean & Montagnon, 1949; Moffatt, 1964, as in the introduction]), while D behaves like r, so N_ψ cannot be smooth there. For any network, an untrained one included, the momentum residual then grows like 1/r. Its square is not integrable against a point density that rises toward the corner, so each draw's loss would be decided by its closest corner point. Interior points are therefore never placed within 0.02 of a reflex corner (B, C, T). A further 10% of the points are placed uniformly by area in the annulus 0.02 < r < 0.5 around those corners, and each interior residual is multiplied by min(1, r/0.3).

**4.4.4 Network and training.** The network is the same modified MLP (width 40, depth 4, tanh), with inputs (x − x_c)/2, where x_c is the centre of the geometry's bounding box. It has 5,363 parameters including λ.

Each draw has 8,192 interior points, 256 inlet points crowded toward the walls (y = sin(πξ/2), ξ uniform) and 256 points on each outlet. The schedule is:
- Adam for 3,000 steps (learning rate 10⁻³, decayed by 0.9 every 1,000 steps), with fresh points every 250;
- four cycles of 500 L-BFGS iterations, each on a fresh draw;
- 150 Levenberg–Marquardt steps with fresh points at every step, initial damping 10⁻⁴ × the mean diagonal of JᵀJ, and the mean of the last 20 iterates as a candidate.

The returned network has the lowest loss on 32,768 held-out points. Training is resumable: the checkpoint stores the random generator's state, and a resumed run ends bit-for-bit where an uninterrupted run does (tested).

The method was developed in three stages on one geometry: 30/90, s = 0.5, L_d = 4, q = 0.5.
- A velocity–pressure formulation with continuity as a penalty lost 6.3% of the mass.
- The stream function made mass and the share exact.
- Extending the walls removed the outlet-corner spike.

Section 6 reports all three.

**4.4.5 Parametric Network 2.** The parametric version (`bifurcation/parametric.py`) takes the split as an input, (x, y, s), over a range [s_min, s_max] at fixed angles and share. Everything that depends on the geometry is recomputed from s through the differentiable polygon of Section 3.1: the wall segments (merged and extended as in 4.4.1), the singular corners and the septum tip. No-slip, mass and the share therefore remain exact at every s. Outlet 1's offset becomes a trained Chebyshev series λ(s) of degree 4. Each draw takes 16 splits, stratified one per sixteenth of the range, with 1,024 interior points in each split's own polygon. The held-out set uses 32 splits.

Two tests check the construction:
- At every split tested, on three angle pairs, the wall segments equal the fixed-geometry construction for that geometry.
- With its s-weights set to zero, the parametric network at a fixed split reproduces the fixed-geometry network's field and loss exactly (residuals equal to 2 × 10⁻¹⁵).

The range is checked before training: a range across which the polygon changes shape, or where any geometry is invalid, is refused. Once trained, the network gives J(s) over the whole range from line integrals alone. Its argmin is located by a 61-point scan followed by Brent's method and compared with the finite-element argmin (Section 4.6).

[First run, 30/90 with s ∈ [0.35, 0.65] and q = 0.5: `docs/bifurcation_re10-parametric.md`. Its optimum was 0.4845 against the finite-element 0.4848, with regret below 0.01%. The results go to Section 6.]

### 4.5 Network 3: optimal widths from the angles and the share

Network 3 (`bifurcation/geometric.py`) is not physics-informed. It maps (α₁, α₂, q) to the optimal split and the lowest pressure drop, and it learns only what the friction law does not supply.

**Baselines.**
- For the split: s_M(q), the Murray split.
- For the drop: J_M(q), the Murray drop of Section 3.5.

**Architecture.** A modified MLP N, with inputs α₁/90 − 1, α₂/90 − 1 and 2q − 1, width 16 and depth 2 (498 parameters), corrects both baselines:

  s = s_lo + (s_hi − s_lo) σ( logit[(s_M − s_lo)/(s_hi − s_lo)] + c ),  J = J_M exp(d).

Here σ is the logistic function and [s_lo, s_hi] = [0.1, 0.9]. The correction c is the antisymmetric part of N's first output, ½[N₁(α₁, α₂, q) − N₁(α₂, α₁, 1 − q)], and d is the symmetric part of its second output. Two properties therefore hold for any parameters:
- the split never leaves [0.1, 0.9], so no proposed width is below w_min;
- swapping the daughters gives the mirrored split, 1 − s, and the same drop.

With N = 0 the network is exactly the friction law (tested).

**Fit.** The targets are s* from the coarse-mesh search and J* from the fine-mesh re-solve. Training minimises the nonlinear least-squares objective

  Σ [(s − s*)/0.001]² + Σ [(ln J − ln J*)/0.001]² + λ_r ‖θ‖²,

so a split error of 0.001 weighs like a 0.1% drop error. The minimiser is MINPACK's Levenberg–Marquardt (Moré, 1978) through SciPy's `least_squares` (Virtanen et al., 2020), with exact Jacobians from JAX, two random initialisations and at most 2,000 evaluations each.

**Regularisation.** The weight decay λ_r ∈ {100, 1,000, 10,000} is chosen by leave-one-angle-pair-out cross-validation: 15 folds, each trained on 14 layouts and scored on the 11 optima of the layout left out. The value chosen was λ_r = 1,000. The folds run in parallel and are cached as they finish, so an interrupted run resumes with identical results (tested). The final network is refitted on all 165 optima, taking 3.8 h on four cores including the cross-validation.

**Validation.** The six held-out pairs (45/75, 15/105, 75/105, 45/135, 30/30, 0/90) are each evaluated at q = 0.2, 0.5 and 0.8. For each of these 18 cases:
1. the finite-element optimum is found by the full procedure of Section 4.2;
2. the network's proposed split is solved by finite elements;
3. the Murray split is solved by finite elements.

The score is the regret of each proposal (Section 4.6), with the friction law as the baseline Network 3 must beat.

### 4.6 Accuracy and design metrics

Loss values are never used as accuracy. Configurations with different loss terms are not comparable by loss (CLAUDE.md). Every accuracy figure is measured against the finite-element reference:
- **Field error:** relative L2 error of u, v and p, ‖f_net − f_FE‖/‖f_FE‖, integrated on the reference's own mesh by its element quadrature. It is also reported outside discs of radius 0.02 around the reflex corners, where the network is not trained.
- **Objective error:** |J_net − J_FE|/J_FE, with both measured by the same 256-point line quadrature.
- **Conservation:** mass imbalance (Q₁ + Q₂ − Q₀)/Q₀ and split error |Q₁/Q₀ − q|. These are exact by construction for the stream-function networks, so they test the construction rather than the training.
- **Energy consistency:** |E_in − E₁ − E₂ − Φ|/Φ, computed from the network alone, with no reference data (Appendix A.6).
- **Argmin error:** s*_net − s*_FE.
- **Regret:** J_FE(s*_net)/J_FE(s*_FE) − 1, the extra pressure drop paid for adopting a network's split, with both evaluated by the reference. Near the optimum the regret is about (J″/J)(δs)²/2, which is 0.22% for δs = 0.01 (Appendix A.3). On the coarse mesh, regrets below about 10⁻⁴ (0.01%) are within the reference's noise and can come out slightly negative.

### 4.7 Implementation and reproducibility

**Software.**
- Python 3.11, JAX 0.10 (Bradbury et al., 2018) and Optax 0.2;
- NumPy 2.4, SciPy 1.17 (Virtanen et al., 2020);
- scikit-fem 12.0 (Gustafsson & McBain, 2020) and Triangle (Shewchuk, 1996).

**Numerical rules.**
- Double precision is enabled before any array is created, for networks and reference alike.
- Every run is determined by its configuration file and its seed. A missing configuration key is an error, never a default. Each run stores its configuration, and its checkpoint carries fingerprints of the configuration and of the code that wrote it.
- Network 1 and Network 2 each draw their model initialisation, training points and held-out points from three independent random streams derived from the seed.
- Tolerances in the test suite are never relaxed to make a test pass.

**Hardware.** All computations, finite-element and network, ran on four CPU cores. No GPU was used.

**Commands.**

    python -m bifurcation.design_study --config bifurcation/study_re10_shares.yaml --out docs/bifurcation_re10-shares
    python -m bifurcation.flow --config bifurcation/b_re10_3090_equal_stream_ext.yaml
    python -m bifurcation.parametric --config bifurcation/p_re10_3090_equal.yaml
    python -m bifurcation.geometric --config bifurcation/n3_re10_all_w16.yaml
    pytest tests/ -q

---

## Appendix A

Notation: per unit depth, a daughter of width wᵢ and length Lᵢ carrying flow Qᵢ in plane Poiseuille flow has a pressure drop of 12μLᵢQᵢ/wᵢ³ and dissipates Pᵢ = 12μLᵢQᵢ²/wᵢ³. In the non-dimensional variables of Section 3.2, μ is replaced by ν = 1/Re_h.

### A.1 Equal outlet pressures put the optimum on a width bound

With both outlets at the same pressure, the daughters are parallel resistances: Qᵢ = wᵢ³Δp/(12μLᵢ). For equal lengths, the drop at fixed Q₀ is Δp = 12μL_d Q₀/(w₁³ + w₂³). Under w₁ + w₂ = w₀, f(s) = s³ + (1 − s)³ has f″(s) = 6 > 0. It is convex, with its minimum at s = ½ and its maxima at the ends of the range. The pressure drop is therefore largest at equal widths and falls monotonically toward either width bound, whatever the junction adds, as long as the daughters dominate. At s = 0.1 the narrow daughter carries s³/(s³ + (1 − s)³) = 0.14% of the flow. The finite-element floors give 0.1–0.2% (Section 5). [Tested: `test_equal_outlet_pressures_drive_the_optimum_to_a_width_bound`.]

### A.2 The planar Murray split

(i) *At fixed total volume.* Minimise Σ Pᵢ subject to Σ Lᵢwᵢ = V. Stationarity gives −36μLᵢQᵢ²/wᵢ⁴ + βLᵢ = 0, so wᵢ⁴ ∝ Qᵢ² and wᵢ ∝ Qᵢ^(1/2), independent of the lengths. Applied to the parent too, Q₀ = Q₁ + Q₂ gives w₀² = w₁² + w₂², the planar square law (Stephenson et al., 2015; Zhou et al., 2024).

(ii) *Under w₁ + w₂ = w₀*, the constraint of this study. Stationarity gives −36μLᵢQᵢ²/wᵢ⁴ + β = 0, so wᵢ⁴ ∝ LᵢQᵢ², and

  w₁/w₂ = (L₁/L₂)^(1/4) (Q₁/Q₂)^(1/2).

For L₁ = L₂ this is s_M = √q/(√q + √(1 − q)). Substituting wᵢ = w₀√qᵢ/S, with S = √q + √(1 − q), gives Σ Qᵢ²/wᵢ³ = Q₀²S⁴/w₀³. The daughters' drop per unit flow is therefore 12νL_dQ₀S⁴/w₀³ = 1.5νL_dQ₀S⁴ for w₀ = 2. Adding the parent's 12νL_pQ₀/w₀³ gives J_M of Section 3.5. At q = 0.5, with ν = 0.1, L_d = 8, L_p = 3 and Q₀ = 4/3, J_M = 6.4 + 0.6 = 7.0. [Tested: `test_the_friction_law_split`, `test_the_friction_drop_is_parallel_poiseuille`.]

(iii) *The angles.* Neither expression contains α₁ or α₂: in the fully developed model the junction costs nothing, and the angles could enter only through the lengths, which are equal here. The model's split is therefore the same for every layout, and any dependence of s* on the angles is a junction effect.

### A.3 The optimum is flat, and its location depends on the slope of the error

Near the optimum, J(s* + δ) ≈ J* [1 + ½(J″/J)δ²], so the regret of a split error δ is ½(J″/J)δ².

For the friction model at q = 0.5, the daughters' part is D(s) ∝ ¼[s⁻³ + (1 − s)⁻³], with D(½) = 4 and D″(½) = 192, so D″/D = 48. Including the parent's constant 0.6 alongside D = 6.4 gives J″/J = 48 × 6.4/7.0 = 43.9. The finite-element value at equal shares is 44.1. A split error of 0.01 therefore costs about 44 × 10⁻⁴/2 = 0.22%.

If a network predicts Ĵ = J(1 + e(s)), its minimum satisfies J′(1 + e) + Je′ = 0. With J′ ≈ J″δ near s*, this gives δ ≈ −e′/(J″/J). An error whose relative value changes by 1% across a 0.1 change in split (e′ = 0.1) moves the predicted optimum by 0.1/44 ≈ 0.0023, whatever its mean value. This is hypothesis H3.

### A.4 Wall shear

Plane Poiseuille flow of flux Qᵢ in width wᵢ has wall shear τᵢ = 6μQᵢ/wᵢ², so

  τ₁/τ₂ = [q/(1 − q)] [(1 − s)/s]².

At s = s_M, (1 − s)/s = √((1 − q)/q) and the ratio is 1: the Murray split equalises wall shear. Away from it, d ln(τ₁/τ₂)/ds = −2/s − 2/(1 − s) = −2/[s(1 − s)]. At small shares s(1 − s) is small, so a small departure δs of the resolved optimum from s_M changes the shear ratio by a large factor.

### A.5 Why a turned daughter should be wider (mechanism for H2)

Model the junction as a minor loss in each daughter, Kᵢ ½ρUᵢ² with Uᵢ = Qᵢ/wᵢ, where Kᵢ grows with the turn. The dissipated power gains ½ρ Σ KᵢQᵢ³/wᵢ², and per unit flow

  J_j(s) = (ρQ₀²/(2w₀²)) [K₁q³/s² + K₂(1 − q)³/(1 − s)²].

At q = ½ and s = ½, dJ_j/ds = (ρQ₀²/(2w₀²)) × 2(K₂ − K₁). If daughter 2 turns more (K₂ > K₁), J_j increases with s, and the optimum moves below s_M, widening the turned daughter by about −J_j′/J″ (Appendix A.3). The model predicts only the sign of the shift. At q = 0.5 the finite-element optima have that sign in all eight layouts with α₁ + α₂ < 180° and unequal angles. The turned daughter is wider in each, by an amount that grows with the bifurcation angle and the asymmetry: s* = 0.4977 at 0/30, 0.4988 at 20/40, 0.4948 at 0/60, 0.4930 at 10/80, 0.4925 at 30/90, 0.4808 at 0/120, 0.4683 at 30/120 and 0.4437 at 0/150, against s_M = 0.5. At 180°, where the septum is a flat face (Section 3.1), no shift is detectable: 0.5000 at 60/120 and 0.5006 at 30/150, both within the search tolerance of 0.001. The Kᵢ were not fitted, so the size of the shift is not a prediction of this model. [H2 must be stated for α₁ + α₂ < 180°, or the 180° result explained.]

### A.6 Mechanical energy balance

Dot the momentum equation with u and integrate over the domain. Using ∇·u = 0, the convective and pressure terms become boundary fluxes, ∮ (p + ½|u|²) u·n ds. The viscous term becomes ν ∫ ∇u:∇u dA − ν ∮ u·∂u/∂n ds. Because ∇·u = 0, ν ∫ ∇u:∇u dA differs from the dissipation Φ = 2ν ∫ e:e dA only by boundary terms. The no-slip walls contribute nothing, so

  E_in − E₁ − E₂ = Φ + (viscous-flux terms on the inlet and outlets).

The opening terms vanish for parallel, developed flow (∂u/∂n = 0 and v = 0 there) and are small for nearly developed flow. The flux-weighted total-pressure drop of Section 3.4 is therefore Φ/Q₀ to that accuracy. For any computed field, |E_in − E₁ − E₂ − Φ|/Φ can be evaluated from the field alone, which makes it a check that needs no reference. It is 7.5 × 10⁻⁶ for the finite-element reference of Section 4.1 and below 10⁻³ in every coarse-mesh verification test. For the networks, whose mass and shares are exact by construction, it is the only internal diagnostic left. [Implemented for the channel in `developing_flow/energy.py`; confirm against a textbook statement of the mechanical energy equation before citing.]

### A.7 Reynolds numbers

With U_c the inlet centreline speed and h the half-width, Re_h = U_ch/ν. For the parabola, the mean speed is U_m = (2/3)U_c, and the hydraulic diameter of a two-dimensional channel is D_h = 2w₀ = 4h. Hence

  Re_Dh = U_mD_h/ν = (8/3) Re_h,  Re_h = 0.375 Re_Dh,

and Re_h = 10 corresponds to Re_Dh ≈ 26.7. Per unit depth, Q₀ = (4/3)U_ch, so Re_h = 3Q₀/(4ν). A daughter's own Re_Dh is (Qᵢ/wᵢ)(2wᵢ)/ν = 2Qᵢ/ν, which depends on its share of the flow but not on its width. The design variable s therefore does not change any channel's Reynolds number, and Re_h = 10 is a single flow rate per unit depth (Q₀ = 13.3ν) for every parent width.

---

## References new in this section

[As for the introduction: read each before submission and confirm the details. Murray (1926a), Stephenson et al. (2015), Zhou et al. (2024) and Celik et al. (2008) are in the introduction's list.]

Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). *JAX: Composable transformations of Python+NumPy programs* [Computer software]. http://github.com/google/jax

Brent, R. P. (1973). *Algorithms for minimization without derivatives*. Prentice-Hall.

Durst, F., Ray, S., Ünsal, B., & Bayoumi, O. A. (2005). The development lengths of laminar pipe and channel flows. *Journal of Fluids Engineering, 127*(6), 1154–1160.

Formaggia, L., Gerbeau, J.-F., Nobile, F., & Quarteroni, A. (2002). Numerical treatment of defective boundary conditions for the Navier–Stokes equations. *SIAM Journal on Numerical Analysis, 40*(1), 376–401.

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics*, 249–256.

Gustafsson, T., & McBain, G. D. (2020). scikit-fem: A Python package for finite element assembly. *Journal of Open Source Software, 5*(52), 2369.

Heywood, J. G., Rannacher, R., & Turek, S. (1996). Artificial boundaries and flux and pressure conditions for the incompressible Navier–Stokes equations. *International Journal for Numerical Methods in Fluids, 22*(5), 325–352.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*.

Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares. *Quarterly of Applied Mathematics, 2*(2), 164–168.

Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming, 45*(1–3), 503–528.

Madsen, K., Nielsen, H. B., & Tingleff, O. (2004). *Methods for non-linear least squares problems* (2nd ed.). Informatics and Mathematical Modelling, Technical University of Denmark.

Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *Journal of the Society for Industrial and Applied Mathematics, 11*(2), 431–441.

Moré, J. J. (1978). The Levenberg–Marquardt algorithm: Implementation and theory. In G. A. Watson (Ed.), *Numerical analysis* (Lecture Notes in Mathematics, Vol. 630, pp. 105–116). Springer.

Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging. *SIAM Journal on Control and Optimization, 30*(4), 838–855.

Rvachev, V. L., & Sheiko, T. I. (1995). R-functions in boundary value problems in mechanics. *Applied Mechanics Reviews, 48*(4), 151–188.

Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. *Proceedings of the 1968 23rd ACM National Conference*, 517–524.

Shewchuk, J. R. (1996). Triangle: Engineering a 2D quality mesh generator and Delaunay triangulator. In M. C. Lin & D. Manocha (Eds.), *Applied computational geometry: Towards geometric engineering* (Lecture Notes in Computer Science, Vol. 1148, pp. 203–222). Springer.

Sukumar, N., & Srivastava, A. (2022). Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks. *Computer Methods in Applied Mechanics and Engineering, 389*, 114333.

Taylor, C., & Hood, P. (1973). A numerical solution of the Navier–Stokes equations using the finite element technique. *Computers & Fluids, 1*(1), 73–100.

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261–272.

Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing, 43*(5), A3055–A3081.
