## Three questions about the split, answered from finite elements (Re_h = 10)

Sources:
- `docs/bifurcation_re10-shares-all.json`: the design study, 165 optima with w₁ + w₂ = w₀.
- `docs/bifurcation_re10-width-rules.md`: rules and total-width scans on the design study's check mesh, 68 geometries, none refused. The shared-optimum rows reproduce the design study's drops exactly.

J is the total-pressure drop per unit flow, so for a fixed inflow it is proportional to pumping power. q is the share of the flow that daughter 1 carries, and s = w₁/w₀. Parent length is 1.5 w₀ and daughter length 4 w₀.

---

### 1. Does the flow have to be split equally? Can an unequal split be more efficient?

**No, and yes.** For a fixed total flow, an equal split is the most expensive one to deliver in a symmetric bifurcation. The cost falls as the split becomes more unequal:

| layout | 5/95 | 10/90 | 20/80 | 30/70 | 40/60 | **50/50** | no share set |
|---|---|---|---|---|---|---|---|
| 60/60 | 4.16 | 4.99 | 6.12 | 6.83 | 7.24 | **7.37** | 3.05 |
| 90/90 (T) | 4.56 | 5.32 | 6.35 | 7.01 | 7.39 | **7.51** | 3.50 |
| 15/15 | 4.00 | 4.82 | 5.94 | 6.65 | 7.06 | **7.19** | 2.87 |

The values are the lowest J at each share, each with its own best widths. An equal split forces all the flow through two half-width channels. An unequal one sends most of it down one wide channel, and a channel's resistance falls like 1/w³. In the friction law the daughters' cost is proportional to (√q + √(1 − q))⁴. That is 4 at q = 0.5, 2.06 at q = 0.05 and 1 with no split at all (Appendix A.2).

When the angles differ, the cheapest shares favour the straighter daughter:
- At 30/150, sending 95% of the flow down the 30° daughter costs 4.07, while sending 95% down the 150° daughter costs 12.15.
- At 60/120 the most expensive share is not 50/50 but 40/60.

The limit is not splitting at all. With no share set, so that the outlets sit at equal pressure, the lowest J (2.76–3.50) comes from narrowing one daughter to the minimum width, after which it carries 0.1–0.2% of the flow (Appendix A.1). Efficiency alone therefore says "don't split". The share is a requirement, set by what each outlet has to deliver; the design question is the cheapest way to meet it.

**At equal flow, unequal widths can be more efficient.** When the two angles differ, the optimum at q = 0.5 makes the more-turned daughter wider, and the saving over equal widths grows with the asymmetry:

| layout | 0/150 | 30/120 | 0/120 | 30/90 | 10/80 | 0/60 | symmetric and 180° layouts |
|---|---|---|---|---|---|---|---|
| optimum s* | 0.444 | 0.468 | 0.481 | 0.493 | 0.493 | 0.495 | 0.500 |
| saving vs equal widths | **4.8%** | 1.7% | 0.7% | 0.10% | 0.09% | 0.05% | ≤ 0.002% |

### 2. What are the different types of split?

There are three separate choices, and each has its own options.

**(a) Who sets the flow split**

| type | outlets | what happens | in the code |
|---|---|---|---|
| prescribed share (including 50/50) | outlet 1's pressure adjusts until it carries q Q₀ | an interior optimum width split exists | flux-defective outlet, `outlet1_share` |
| free split | both outlets at the same pressure | the flow chooses; the "optimum" narrows one branch to the minimum width | `outlet1_share: None` |

**(b) How the widths are divided, for a fixed total width**

| rule | split s = w₁/w₀ | basis |
|---|---|---|
| equal widths | 0.5 | none: the unoptimised design |
| Murray (planar, fixed total width) | √q / (√q + √(1 − q)) | fully developed friction, junction ignored (Appendix A.2) |
| resolved optimum | the design study's s* | finite elements: friction plus the junction; departs from Murray as the angles turn |

**(c) How wide the daughters are together**

| rule | law | combined width at 50/50 | basis |
|---|---|---|---|
| area-preserving (this study's constraint) | w₁ + w₂ = w₀ | 1.00 w₀ | fixed footprint or cross-section: the flow keeps its mean speed through the junction |
| planar Murray (square law) | w₁² + w₂² = w₀² | 1.41 w₀ | minimum pumping power plus a cost on channel volume, for flat channels |
| round-pipe Murray (cube law) | w₁³ + w₂³ = w₀³ | 1.59 w₀ | the same principle for circular pipes; tested here as a comparison only, since it is the wrong law for flat channels |

The layouts themselves also come in types: symmetric Y (60/60), asymmetric Y (30/90, 0/120), T-junction (90/90), and turned back past 90° (0/150, 30/150). They are covered by the design study and Network 3.

### 3. Must the daughters' widths add up to the parent's?

**No. w₁ + w₂ = w₀ is a constraint chosen for this study, not a result.** What the best design is depends on what is held fixed.

**Nothing held fixed: there is no optimum.** Pressure drop alone always favours wider channels. At 60/60 and 50/50, with the daughters scaled together:

| combined width w₁ + w₂ | 1.0 w₀ | 1.2 w₀ | 1.41 w₀ | 1.6 w₀ | 2.0 w₀ |
|---|---|---|---|---|---|
| J | 7.37 | 4.56 | 3.06 | 2.34 | 1.57 |

Something has to pay for width, such as footprint, material, dead volume or fabrication.

**A fixed total width (area-preserving).** This is the design study: the best split is the Murray split, corrected by the junction.

**Width with a price: Murray's principle.** Add a cost for channel area to the pumping power, C = Q₀J + bA. Choose b so that the parent's own width is the optimum for its flow, b = 36νQ₀²/w₀⁴; this is the only choice consistent with the parent being sized by the same rule. The friction law then predicts an optimum combined width of (√q + √(1 − q)) w₀, the square law. Finite elements confirm it:

| layout, 50/50 | combined width with the lowest C | friction-law prediction |
|---|---|---|
| 60/60 | 1.42–1.43 w₀ | 1.414 w₀ |
| 0/150 | 1.37–1.38 w₀ | 1.414 w₀ |

The strongly turned layout prefers slightly narrower daughters. The minimum is flat: C changes by less than 1% between 1.3 and 1.5 w₀.

How much better is the square law than keeping w₁ + w₂ = w₀, in Murray's cost?

| layout | at 50/50 | at 20/80 |
|---|---|---|
| 60/60 | −15.4% | −10.6% |
| 30/90 | −15.1% | −9.5% |
| 0/120 | −14.1% | −8.8% |
| 90/90 (T) | −14.2% | −10.2% |
| 0/150 | −10.6% | −5.2% |

At 50/50 the square law cuts the pumping power by 58%, from 9.83 to 4.09 at 60/60, for 30% more channel area. The cube law, the round-pipe rule, is cheaper than w₁ + w₂ = w₀ but 1–3% dearer than the square law: it makes flat daughters too wide. The friction-law combined width is 1.41 w₀ at 50/50, 1.34 w₀ at 20/80 and 1.20 w₀ at 5/95, and it falls to 1.0 as one daughter takes all the flow.

In short:
- with a fixed footprint, keep w₁ + w₂ = w₀ and split by Murray's rule, corrected for the junction;
- if channel size has a cost that can be traded against pumping, the daughters together should be about 1.4 times the parent's width at an even split, the planar square law.

### Further findings

- **Widths do not set the flow split.** With both outlets at the same pressure, flow divides as w³, so Murray-sized daughters deliver (q/(1 − q))^1.5 instead of q/(1 − q). To deliver an unequal share, the outlets must sit at different pressures: at 60/60 and 20/80, outlet 1 must sit 4.15 lower in units of ρU_c². Downstream resistances set the share; the widths only set what it costs.
- **Equal widths are a poor default for unequal flows.** At 20/80, equal widths cost 58–98% more than the optimum, across the five layouts.
- **Shear.** Murray's split gives both daughters the same wall shear (Appendix A.4); the junction-corrected optimum does not. This is a trade-off for shear-sensitive applications.

### Limits

- Re_h = 10 only. Inertia, and with it the junction's share of the loss, grows with flow rate, so the optima may move at higher Re.
- Two-dimensional flow, and both daughters the same length.
- The Murray price b is one consistent choice. A design with a different price for size, such as a fixed chip area, gives a different combined width.
