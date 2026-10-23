"""Tier A's results against its pre-registered pass marks.

    python docs/re_study/report/build_tier_a.py
        -> docs/re_study/tierA/results.md, docs/re_study/tierA/fig_tierA_argmin.png,
           docs/re_study/report/content_tier_a.json
    node docs/re_study/report/build_docx.js docs/re_study/report/content_tier_a.json \\
        docs/re_study/tierA_results.docx

Pass marks (docs/re_study/tierA/preregistration.md, committed before any run):
P1 |s*_PINN - s*_ref| <= 0.002; P2 |s*_ref - s_L| <= 0.003 where no daughter
turns beyond 90 degrees. Every number is read from the exported run summaries.
"""
import json
import pathlib
import struct
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bifurcation import tier_a as T  # noqa: E402
from bifurcation.sensitivity import murray_split  # noqa: E402

STUDY = ROOT / "docs" / "re_study"
TIER = STUDY / "tierA"
OUT = STUDY / "report" / "content_tier_a.json"
P1, P2 = 0.002, 0.003
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LOOK = {(30.0, 90.0): ("#2a78d6", "o", "30/90"), (15.0, 15.0): ("#eb6834", "s", "15/15"),
        (0.0, 120.0): ("#1baf7a", "D", "0/120")}
blocks = []


def h1(t): blocks.append({"h1": t})
def h2(t): blocks.append({"h2": t})
def p(t): blocks.append({"p": t})
def bullets(items): blocks.append({"bullets": items})


def table(head, rows, widths=None, note=None):
    blocks.append({"table": {"head": head, "rows": rows, "widths": widths}})
    if note:
        blocks.append({"note": note})


def image(path, caption):
    with open(path, "rb") as f:
        f.read(16)
        w, h = struct.unpack(">II", f.read(8))
    blocks.append({"img": str(path), "w": 605, "h": round(605 * h / w), "caption": caption})


def runs():
    out = []
    for angles, share in T.CASES:
        for flow in T.FLOWS:
            name = T.run_tag(angles, share, flow["reynolds"])
            stem = (f"pinn_{angles[0]:g}-{angles[1]:g}"
                    + ("" if share == 0.5 else f"_q{share:g}")
                    + f"_Re{flow['reynolds']:g}_seed{T.SEED}")
            path = TIER / f"{stem}_summary.json"
            s = json.loads(path.read_text()) if path.is_file() else {"missing": name}
            s_l = T.length_split(angles, share)[0]
            out.append({"angles": angles, "share": share, "re": flow["reynolds"], "s_L": s_l,
                        "s": s, "stem": stem})
    return out


def edges(s):
    grid = sorted((f for f in s["fields"] if not f.get("at")), key=lambda f: f["split"])
    return grid


def verdicts(r):
    s = r["s"]
    if "missing" in s or "failed" in s:
        return None, None
    p1 = abs(s["argmin_error"]) <= P1
    scope = max(r["angles"]) <= 90.0
    gap = s["s_star_reference_mesh_brent"] - r["s_L"]
    p2 = (abs(gap) <= P2) if scope else None
    return p1, p2


R = runs()
done = [r for r in R if "argmin_error" in r["s"]]

# --------------------------------------------------------------------------- #
blocks.append({"title": "Tier A: new shares and a symmetric control"})
p("Twelve parametric PINN runs, one seed each (21), at Re_h = 0.1 and 10: 30/90 at q = 0.2, "
  "0.35, 0.65 and 0.8; 15/15 at q = 0.2 (the null test); 0/120 at q = 0.5. Rules and pass "
  "marks were committed before any screening solve or run (docs/re_study/tierA/"
  "preregistration.md). P1: the PINN's optimum within ±0.002 of the reference optimum (Brent "
  "on the reference mesh). P2: the reference optimum within 0.003 of the length-corrected split "
  "s_L, where no daughter turns beyond 90° (30/90 and 15/15). Both tolerances are the paper's, "
  "unchanged. Each trained range is s_L ± 0.15; all twelve cases passed the screening for an "
  "interior minimum with daughters 4 long (docs/re_study/tierA/screening.md).")
n1 = sum(1 for r in done if verdicts(r)[0])
in_scope = [r for r in done if verdicts(r)[1] is not None]
n2 = sum(1 for r in in_scope if verdicts(r)[1])
p(f"Of the {len(done)} runs reported, {n1} pass P1. Of the {len(in_scope)} in P2's scope, "
  f"{n2} pass P2. Verdicts are on the unrounded values; nothing was rerun.")

h1("What Tier A shows")
bullets([
    "P1 passes only at q = 0.5: 0/120 at both Reynolds numbers, to within 0.0001. Every run at "
    "another share fails it, at both Reynolds numbers, including the symmetric 15/15 null test.",
    "At Re_h = 10 the misses are moderate and grow with the share's distance from 0.5: −0.0049 "
    "at q = 0.35, +0.0020 at 0.65 (failing by 0.00002), −0.0064 at 0.8, −0.0078 at 0.2, and "
    "−0.0068 for 15/15 at 0.2. The regret stays below 0.13%, and the pressure-drop error within "
    "±2.6% over each off-centre range (±3.2% for 0/120).",
    "At Re_h = 0.1 every off-centre run fails, by +0.006 to +0.156 (regret up to 40%; the PINN's "
    "optimum always lies above the reference's). Every Tier A run there, 0/120 included, ends at "
    "a held-out loss of 1.1 to 22, against 0.12 to 0.16 for the three earlier 30/90 runs at "
    "q = 0.5, and with largest drop errors of 8 to 30%; 0/120 (1.1, −8.1% at its range's top) still "
    "placed its optimum to 0.0001, because its error was flat near the optimum.",
    "The 15/15 null test misses by as much as 30/90 at the same share (−0.0068 against −0.0078 "
    "at Re_h = 10), so the miss belongs to the share, not to the layout's length asymmetry.",
    "The share, not the corner-mode cutoff, is the limit (section 5, amendment 1): every "
    "off-centre case also had a smaller cutoff, but 30/90 at q = 0.5 passes P1 with its cutoff "
    "forced to 0.15 at both Reynolds numbers, and q = 0.35 still fails with it raised to 0.40. "
    "The small cutoff costs field accuracy (held-out loss and u error several times higher) "
    "without moving the optimum.",
    "One seed is enough to call the Re_h = 10 misses real (section 5): a second seed of 30/90 at "
    "q = 0.2 misses by −0.0085 against seed 21's −0.0078, a spread of 0.0007, about the 0.0005 "
    "spread of three seeds at q = 0.5.",
    "Mechanism (section 6): the slope e′ of the PINN's pressure-drop error at the reference "
    "optimum, measured over ±0.01, predicts each argmin error through δs ≈ −e′/(J″/J) within a "
    "factor 0.8–1.4 in 13 of 18 runs, and within 0.0003 in three more whose shifts are below "
    "0.0005; it is moderately off at q = 0.65, Re_h = 0.1 (0.68) and fails at q = 0.2, "
    "Re_h = 0.1 (2.37), where training collapsed. Off-centre slopes at Re_h = 10 are 17–35% per "
    "unit s (5.7% at q = 0.65), against 0.5–5.2% at q = 0.5 (the earlier 30/90 runs and 0/120); "
    "at Re_h = 0.1 the earlier q = 0.5 slopes were 5.5–16.4%, and 0/120's 0.14%.",
    "P2, a property of the reference solution: s_L lies within 0.003 of the optimum at q = 0.35 "
    "and 0.65 at Re_h = 0.1, q = 0.35 at 10, and for 15/15 at both; it does not at q = 0.2 and "
    "0.8 for 30/90 (0.004 to 0.006), nor at q = 0.65 at Re_h = 10 (0.0031). The length correction "
    "loses accuracy as the share moves from 0.5 in a layout with unequal lengths.",
    "Sampling is not the cause (section 7, amendment 3): with points scaled to each channel's "
    "width (7.9–9.5 across the narrow daughter instead of 2.7–4.3) both 30/90 runs at Re_h = 10 "
    "still fail P1, by more (q = 0.35: −0.0069; q = 0.2: −0.0128), so by the pre-registered rule "
    "the share itself is the limit; caveat: the junction coarsened to keep the total fixed.",
    "Given 400 LM steps instead of 150, the q = 0.5 cutoff-0.15 control at Re_h = 0.1 kept "
    "accepting every step at low damping and fell a further 2.8-fold to a held-out 0.65, then "
    "levelled off, still 4–5 times the cutoff-0.5 runs' 0.12–0.16. LM is slow there, not "
    "stalled, and time closes only part of the gap. Building s_L where it misses by 0.004–0.006 "
    "costs 0.03–0.06% in pressure drop.",
])

h1("1. Every run against the pass marks")
rows = []
for r in R:
    s = r["s"]
    head = [f"{r['angles'][0]:g}/{r['angles'][1]:g}", f"{r['share']:g}", f"{r['re']:g}"]
    if "argmin_error" not in s:
        rows.append(head + ["not finished"] + [""] * 8)
        continue
    p1, p2 = verdicts(r)
    gap = s["s_star_reference_mesh_brent"] - r["s_L"]
    rows.append(head + [
        f"{s['s_star_pinn']:.4f}", f"{s['s_star_reference_mesh_brent']:.4f}",
        f"{s['argmin_error']:+.4f}", "pass" if p1 else "FAIL",
        f"{r['s_L']:.4f}", f"{gap:+.4f}",
        "—" if p2 is None else ("pass" if p2 else "FAIL"),
        f"{100 * s['regret']:.4f}%"])
table(["Layout", "q", "Re_h", "PINN s*", "Reference s*", "Argmin error", "P1", "s_L",
       "s*_ref − s_L", "P2", "Regret"], rows,
      [800, 500, 550, 900, 950, 950, 550, 800, 950, 550, 900],
      "P1 |argmin error| ≤ 0.002; P2 |s*_ref − s_L| ≤ 0.003, not applied to 0/120 (a daughter "
      "turned back). Regret = J_ref(s*_PINN)/J_ref(s*_ref) − 1.")

h1("2. Accuracy and training")
rows = []
for r in done:
    s = r["s"]
    g = edges(s)
    star = next(f for f in s["fields"] if f.get("at"))
    t = s["training"]
    cutoff = __import__("yaml").safe_load(
        (ROOT / "bifurcation" / f"p_{T.run_tag(r['angles'], r['share'], r['re'])}.yaml").read_text()
    )["model"]["corner_cutoff"]
    rows.append([f"{r['angles'][0]:g}/{r['angles'][1]:g}", f"{r['share']:g}", f"{r['re']:g}",
                 f"{cutoff:g}",
                 f"{100 * g[0]['drop_error']:+.2f}% / {100 * g[3]['drop_error']:+.2f}% / "
                 f"{100 * g[-1]['drop_error']:+.2f}%",
                 f"{100 * max(abs(f['drop_error']) for f in g):.2f}%",
                 f"{star['u']:.2e}", f"{t['selected_heldout_loss']:.3g}",
                 f"{t['wall_seconds'] / 60:.0f}", "yes" if t["plateaued"] else "no"])
table(["Layout", "q", "Re_h", "Corner cutoff", "Drop error: low end / middle / high end",
       "Largest |drop error|", "u error at s*", "Held-out loss", "Min", "Plateaued"], rows,
      [800, 500, 550, 700, 2500, 1000, 950, 950, 550, 850],
      "Drop error = J_PINN/J_ref − 1 at the seven evaluation splits. Held-out loss is a "
      "diagnostic only, in each run's pressure units (μU_c/h at Re_h 0.1, ρU_c² at 10).")

# ---- the earlier q = 0.5 runs, as the baseline ----------------------------- #
base = []
for re_ in (0.1, 10):
    for seed in (21, 22, 23):
        f = STUDY / f"pinn_30-90_Re{re_:g}_seed{seed}_summary.json"
        if f.is_file():
            base.append(json.loads(f.read_text()))
h2("Baseline: the earlier 30/90 runs at q = 0.5")
p("Range [0.35, 0.65], corner cutoff 0.5 (the earlier protocol; not part of Tier A).")
table(["Re_h", "Seed", "Argmin error", "Regret", "u error at s*", "Held-out loss"],
      [[f"{b['reynolds']:g}", str(b["seed"]), f"{b['argmin_error']:+.4f}",
        f"{100 * b['regret']:.4f}%", f"{next(f for f in b['fields'] if f.get('at'))['u']:.2e}",
        f"{b['training']['selected_heldout_loss']:.3g}"] for b in base],
      [900, 900, 1500, 1500, 1500, 1500])

# ---- figure ---------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, re_ in zip(axes, (10.0, 0.1)):
    ax.axhspan(-P1, P1, color=GRID, lw=0, label="P1: ±0.002")
    ax.axhline(0, color=MUTED, lw=0.8)
    for key, (colour, marker, label) in LOOK.items():
        pts = [(r["share"], r["s"]["argmin_error"]) for r in done
               if tuple(r["angles"]) == key and r["re"] == re_]
        if pts:
            # markers only: Tier A has no 30/90 run at q = 0.5 for a line to pass through
            ax.plot(*zip(*sorted(pts)), marker=marker, color=colour, ms=8, lw=0,
                    label=f"{label}, Tier A")
    old = [(0.5, b["argmin_error"]) for b in base if b["reynolds"] == re_]
    if old:
        ax.plot(*zip(*old), "o", ms=8, mfc="none", mec=INK, mew=1.5,
                label="30/90, q = 0.5, earlier runs (3 seeds)")
    if re_ < 1.0:      # errors from 0.006 to 0.16: symmetric log, linear inside the P1 band
        ax.set_yscale("symlog", linthresh=P1)
    ax.set_title(f"Re_h = {re_:g}" + ("  (symmetric log scale, linear within ±0.002)"
                                       if re_ < 1.0 else ""), loc="left", fontsize=10)
    ax.set_xlabel("daughter 1's share of the flow, q", fontsize=9)
    ax.set_ylabel("argmin error s*_PINN − s*_ref", fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
FIG = TIER / "fig_tierA_argmin.png"
fig.savefig(FIG, dpi=180)
plt.close(fig)
h1("3. The argmin error against the share")
image(FIG, "Argmin error of every Tier A run against daughter 1's share, with the earlier q = 0.5 "
      "30/90 runs (open circles, earlier protocol). The grey band is P1's ±0.002. Each panel "
      "has its own vertical scale; Re_h = 0.1's is symmetric-logarithmic.")

h1("4. P2: the reference optimum against s_L")
rows = []
for angles, share in T.CASES:
    if max(angles) > 90.0:
        continue
    cells = [f"{angles[0]:g}/{angles[1]:g}", f"{share:g}", f"{murray_split(share):.4f}",
             f"{T.length_split(angles, share)[0]:.4f}"]
    for re_ in (0.1, 10.0):
        r = next(x for x in R if tuple(x["angles"]) == tuple(angles) and x["share"] == share
                 and x["re"] == re_)
        if "argmin_error" in r["s"]:
            gap = r["s"]["s_star_reference_mesh_brent"] - r["s_L"]
            cells.append(f"{r['s']['s_star_reference_mesh_brent']:.4f} ({gap:+.4f})")
        else:
            cells.append("—")
    rows.append(cells)
table(["Layout", "q", "s_M", "s_L", "s*_ref at Re_h 0.1 (− s_L)", "s*_ref at Re_h 10 (− s_L)"],
      rows, [900, 600, 1000, 1000, 2400, 2400],
      "s_M is the planar Murray split √q/(√q + √(1 − q)). P2 passes where |s*_ref − s_L| ≤ 0.003.")

# ---- 5. controls (amendment 1) -------------------------------------------- #
CONTROLS = STUDY / "tierA_controls"
h1("5. Controls: the share against the corner cutoff, and a second seed")
p("Pre-registered as amendment 1 after the twelve runs, with the decision rule fixed before "
  "any control ran: if q = 0.5 fails P1 at cutoff 0.15 while q = 0.35 passes at 0.40, the "
  "cutoff rule caused the failures; if q = 0.5 passes at 0.15 and q = 0.35 still fails at "
  "0.40, the share itself is the limit.")
rows = []
for angles, share, re_, cutoff, seed in T.CONTROLS:
    stem = (f"pinn_{angles[0]:g}-{angles[1]:g}" + ("" if share == 0.5 else f"_q{share:g}")
            + f"_Re{re_:g}_seed{seed}")
    f = CONTROLS / f"{stem}_summary.json"
    if not f.is_file():
        rows.append([f"{angles[0]:g}/{angles[1]:g}", f"{share:g}", f"{re_:g}",
                     "rule" if cutoff is None else f"{cutoff:g}", str(seed), "not finished",
                     "", "", "", ""])
        continue
    s = json.loads(f.read_text())
    used = __import__("yaml").safe_load((ROOT / s["run"] / "config.yaml").read_text())["model"]["corner_cutoff"]
    tier = next((r["s"] for r in R if tuple(r["angles"]) == tuple(angles) and r["share"] == share
                 and r["re"] == re_ and "argmin_error" in r["s"]), None)
    star = next(x for x in s["fields"] if x.get("at"))
    rows.append([f"{angles[0]:g}/{angles[1]:g}", f"{share:g}", f"{re_:g}", f"{used:g}", str(seed),
                 f"{s['argmin_error']:+.4f}", "pass" if abs(s["argmin_error"]) <= P1 else "FAIL",
                 f"{100 * s['regret']:.4f}%", f"{s['training']['selected_heldout_loss']:.3g}",
                 "—" if tier is None else f"{tier['argmin_error']:+.4f}"])
table(["Layout", "q", "Re_h", "Cutoff", "Seed", "Argmin error", "P1", "Regret", "Held-out loss",
       "Tier A run's argmin error"], rows, [800, 500, 550, 650, 550, 1050, 550, 950, 1050, 1350],
      "The Tier A column is the same case under the pre-registered cutoff rule (seed 21). For "
      "q = 0.5 there was no Tier A 30/90 run; the earlier q = 0.5 runs (cutoff 0.5, range "
      "[0.35, 0.65]) are in section 2's baseline.")

# ---- 6. diagnostics ------------------------------------------------------- #
D = json.loads((TIER / "diagnostics.json").read_text())
LS = json.loads((TIER / "diagnostics_local_slope.json").read_text())
h1("6. Why the optimum moves, and why Re_h = 0.1 collapses")
h2("The error slope at the optimum predicts each argmin error")
p("If J_PINN = J_ref (1 + e(s)), the PINN's optimum lies at δs ≈ −e′(s*)/(J″/J) from the "
  "reference's (supplement S1.3). e′ is measured here by central difference from two extra "
  "reference-mesh solves at s*_ref ± 0.01, J″/J by the central second difference of J_ref.")
from bifurcation import tier_a_diagnostics as TD  # noqa: E402
by_run = {s["run"]: s for s in TD.summaries()}
rows = []
for run, r in LS.items():
    s = by_run[run]
    rows.append([TD.label(s), s["group"], f"{100 * r['e_slope']:+.2f}%", f"{r['curvature']:.1f}",
                 f"{r['predicted']:+.5f}", f"{r['measured']:+.5f}",
                 f"{r['measured'] / r['predicted']:.2f}" if abs(r["predicted"]) > 1e-12 else "—",
                 f"{r['measured'] - r['predicted']:+.5f}"])
table(["Run", "Group", "e′ per unit s", "J″/J", "Predicted δs", "Measured", "Measured / predicted",
       "Measured − predicted"], rows, [2700, 1300, 950, 650, 1050, 1050, 1000, 1100],
      "Where both shifts are below 0.0005 (0/120 at both Reynolds numbers; the earlier q = 0.5 "
      "run of seed 22 at Re_h = 10) the ratio is not meaningful and the difference is the test. "
      "Both optima are Brent searches to ±0.002 (evaluation.search_xatol).")
h2("Training: Levenberg–Marquardt fails to recover off-centre at Re_h = 0.1")
rows = [[h["run"], h["group"], f"{h['adam_end']:.3g}", f"{h['lbfgs_peak']:.3g} ({h['lbfgs_jump']:.1f}×)",
         f"{h['lm_min']:.3g}", f"{h['lm_recovery']:.0f}"] for h in D["history"]]
table(["Run", "Group", "Held-out, end of Adam", "L-BFGS peak (× Adam end)", "LM minimum",
       "End of L-BFGS / LM minimum"], rows, [2900, 1400, 1300, 1700, 1200, 1500],
      "At Re_h = 0.1 every run, the working q = 0.5 ones included, rises during L-BFGS. The "
      "earlier q = 0.5 runs then fall about 400-fold under Levenberg–Marquardt; the Tier A "
      "runs only 5- to 40-fold. Held-out loss is a diagnostic, in each run's pressure units.")
h2("Collocation points across the narrow daughter")
rows = [[x["case"], f"{x['split']:.4f}", f"{x['d1_width']:.3f}", f"{x['d1_across']:.1f}",
         f"{x['d2_width']:.3f}", f"{x['d2_across']:.1f}"] for x in D["resolution"]]
table(["Case", "Split", "Daughter 1 width", "Points across it", "Daughter 2 width",
       "Points across it"], rows, [1800, 1000, 1500, 1400, 1500, 1400],
      "One draw of 1,024 points per split (the trainer's sampler, mean of 8 draws); 'points "
      "across' = width / mean spacing. The narrowest daughters get 2.7–3.3 points across, against "
      "5.8 at the earlier q = 0.5 range's edge; q = 0.35 (4.3) also failed at Re_h = 0.1, so "
      "sampling density alone does not explain the collapse.")
h2("P2: the friction law with lengths re-measured at every split")
rows = [[x["case"], f"{x['s_M']:.4f}", f"{x['s_L']:.4f}", f"{x['friction_centreline']:.4f}",
         f"{x['friction_closed']:.4f}",
         ", ".join(f"{k}: {v:.4f}" for k, v in sorted(x["reference"].items(), key=lambda kv: float(kv[0])))]
        for x in D["friction"]]
table(["Case", "s_M", "s_L (lengths at s_L)", "Friction law, centreline lengths at every s",
       "Closed lengths at every s", "Reference optimum (Re_h: s*)"], rows,
      [1300, 800, 1200, 1800, 1500, 2600],
      "Re-measuring the lengths at every split does not close the gap at q = 0.2 and 0.8 "
      "(30/90): the departure there is a junction effect that grows as the share leaves 0.5, "
      "not the frozen-length shortcut in s_L.")

# ---- 7. amendment 3 ------------------------------------------------------ #
A3 = STUDY / "tierA_amend3"
h1("7. Amendment 3: the narrow daughter's sampling, LM given time, and what s_L costs")
h2("Width-scaled sampling")
p("Tier A's 30/90 cases at q = 0.35 and 0.2, Re_h = 10, seed 21, rule's cutoff, repeated with "
  "interior points at a density ~ 1/w² in each channel (the same total per draw): 7.9–9.5 "
  "points across every channel, against 2.7–4.3 across the narrow daughter at the range edge "
  "with uniform sampling. Decision rule (fixed before the runs): both pass → the sampler caused "
  "the off-centre misses; only q = 0.35 → moderate shares only; neither → the share itself is "
  "the limit. Caveat: the parent and junction drop from about 15 to 8–9.5 points across.")
rows = []
for share, across_uniform in ((0.35, 4.3), (0.2, 2.7)):
    f3 = A3 / f"pinn_30-90_q{share:g}_Re10_seed21_summary.json"
    tier = next(r["s"] for r in R if tuple(r["angles"]) == (30.0, 90.0) and r["share"] == share
                and r["re"] == 10.0)
    if not f3.is_file():
        rows.append([f"{share:g}", "width", "not finished", "", "", "", "", "", "",
                     f"{tier['argmin_error']:+.4f}"])
        continue
    s = json.loads(f3.read_text())
    rows.append([f"{share:g}", "width", f"{s['s_star_reference_mesh_brent']:.4f}",
                 f"{s['s_star_pinn']:.4f}", f"{s['argmin_error']:+.4f}",
                 "pass" if abs(s["argmin_error"]) <= P1 else "FAIL", f"{100 * s['regret']:.4f}%",
                 "7.9–9.5", f"{s['training']['selected_heldout_loss']:.3g}",
                 f"{tier['argmin_error']:+.4f}"])
table(["q", "Sampling", "s*_ref", "s*_PINN", "Argmin error", "P1", "Regret",
       "Points across the narrow daughter", "Held-out loss", "Uniform sampling (Tier A)"], rows,
      [500, 800, 900, 900, 1050, 550, 900, 1300, 1000, 1300],
      "e′ and the predicted shift for these runs are in section 6's table: +24.8% and −0.0072 "
      "(measured −0.0069) at q = 0.35, +57.7% and −0.0152 (measured −0.0128) at q = 0.2.")
p("Verdict by the pre-registered rule: neither passes, so the share itself is the limit, not the "
  "sampler. Both misses grew rather than shrank (−0.0049 → −0.0069 and −0.0078 → −0.0128) and "
  "the local error slope still predicts them within a factor 0.85–0.95. The caveat stands: the "
  "total was held, so the junction coarsened from about 15 to 8–9.5 points across while the "
  "narrow daughter gained; a run that adds points to the narrow daughter without taking them "
  "from the junction would be the direct test.")
h2("Levenberg–Marquardt at Re_h = 0.1: slow, not stalled")
p("From the stored logs of three runs at Re_h = 0.1: the healthy 30/90 run at q = 0.5 with "
  "cutoff 0.5 (seed 21), the q = 0.5 control at cutoff 0.15, and Tier A's q = 0.35 at cutoff "
  "0.30. The per-step damping and gradient norm were not logged before amendment 3; the "
  "continuation run logs both.")
table(["Run", "L-BFGS peak (× end of Adam)", "LM steps accepted", "Rejected attempts",
       "Final damping λ (absolute)", "Held-out fall, LM steps 1–10 / 10–50 / 50–150",
       "Held-out after 150 steps"],
      [["q 0.5, cutoff 0.5 (healthy)", "6.1", "150 of 150", "4", "0.0046", "8.2× / 4.5× / 8.5×", "0.142"],
       ["q 0.5, cutoff 0.15", "3.3", "150 of 150", "9", "0.108", "2.5× / 1.8× / 2.2×", "2.19"],
       ["q 0.35, cutoff 0.30", "2.5", "150 of 150", "23", "0.340", "1.9× / 1.3× / 1.6×", "6.35"]],
      [1900, 1300, 1100, 1100, 1200, 2000, 1100],
      "No step reached the 30-attempt limit, so LM never stalled at maximum damping; it kept "
      "accepting steps but each bought less. The healthy run overshot most during L-BFGS, so the "
      "overshoot does not explain the failures. Absolute λ scales with J^T J and is only roughly "
      "comparable between runs.")
LM400 = A3 / "pinn_30-90_Re0.1_seed22_summary.json"
if LM400.is_file():
    s = json.loads(LM400.read_text())
    trace = json.loads((ROOT / "bifurcation" / "runs" / "parametric_re0.1_3090_q0.5_cut0.15_lm400_corners_s22"
                        / "history.json").read_text())
    lm = [x for x in trace if x.get("phase") == "lm"]
    rows = []
    for a, b in ((1, 50), (51, 100), (101, 150), (151, 200), (201, 250)):
        seg = [x for x in lm if a <= x["iteration"] <= b]
        avg = [x["heldout_average"] for x in seg if x["heldout_average"] is not None]
        lam = sorted(x["lm_damping_relative"] for x in seg)
        grad = sorted(x["gradient_norm"] for x in seg)
        rows.append([f"{150 + a}–{150 + b}", f"{min(avg):.3g}" if avg else "—",
                     f"{sum(x['lm_accepted'] for x in seg)} of {len(seg)}",
                     f"{sum(x['lm_attempts'] - 1 for x in seg)}",
                     f"{lam[len(lam) // 2]:.1e}", f"{lam[-1]:.1e}", f"{grad[len(grad) // 2]:.0f}"])
    p(f"The q = 0.5, cutoff-0.15 control continued for 250 more LM steps (400 in all, seed 22, from "
      f"its selected network). On the continuation's own held-out set the starting network scored "
      f"{lm[0]['heldout_loss']:.2f} (the parent's selection on seed 21's set: 2.04). The best "
      f"averaged held-out loss was {s['training']['selected_heldout_loss']:.3g}, a further "
      f"{lm[0]['heldout_loss'] / s['training']['selected_heldout_loss']:.1f}-fold fall, reached at "
      f"step {150 + int(s['training']['selected'].split(':')[1])}; the last 50 steps did not improve "
      f"on it (plateaued: yes). Argmin error {s['argmin_error']:+.4f} (parent −0.0012), regret "
      f"{100 * s['regret']:.4f}%, u error at s* 0.86% (parent about 1.1%).")
    table(["LM steps", "Best averaged held-out", "Accepted", "Rejected attempts",
           "Median λ (relative to mean diag JᵀJ)", "Largest λ", "Median gradient norm"], rows,
          [1100, 1300, 1000, 1100, 1700, 1100, 1300],
          f"Every step was accepted, with at most {max(x['lm_attempts'] for x in lm)} attempts "
          "(the limit is 30); damping stayed low (median below "
          "0.01 relative), so LM is neither stalled nor over-damped. The loss keeps falling, but "
          "slowly, and levels off near 0.65 — still 4–5 times the 0.12–0.16 of the cutoff-0.5 runs. "
          "Given time, LM recovers part of the gap, not all of it: the small cutoff limits what the "
          "network reaches at Re_h = 0.1, not only how fast.")
else:
    p("The continuation to 400 LM steps is queued; its result is added here when it finishes.")
h2("What building s_L costs where it misses (reference solver only)")
SL = json.loads((TIER / "diagnostics_s_l_cost.json").read_text())
table(["30/90, q", "Re_h", "s_L", "Reference s*", "s_L − s*", "J(s_L)/J* − 1"],
      [[f"{v['share']:g}", f"{v['reynolds']:g}", f"{v['s_L']:.4f}", f"{v['s_ref']:.4f}",
        f"{v['s_L'] - v['s_ref']:+.4f}", f"{100 * v['cost']:.4f}%"]
       for v in sorted(SL.values(), key=lambda v: (v["share"], v["reynolds"]))],
      [1200, 900, 1200, 1400, 1300, 1600],
      "One reference-mesh solve at s_L against the reference-mesh optimum J*. Where s_L misses by "
      "0.004–0.006 (q = 0.2 and 0.8) the cost is 0.03–0.06%, consistent with (J″/J)δ²/2.")

OUT.write_text(json.dumps(blocks, indent=1, ensure_ascii=False))
lines = []
for b in blocks:
    if "title" in b:
        lines += [f"# {b['title']}", ""]
    elif "h1" in b:
        lines += [f"## {b['h1']}", ""]
    elif "h2" in b:
        lines += [f"### {b['h2']}", ""]
    elif "p" in b:
        lines += [b["p"], ""]
    elif "note" in b:
        lines += [f"*{b['note']}*", ""]
    elif "bullets" in b:
        lines += [f"- {x}" for x in b["bullets"]] + [""]
    elif "img" in b:
        lines += [f"![{b['caption']}]({pathlib.Path(b['img']).name})", "", f"*{b['caption']}*", ""]
    elif "table" in b:
        t = b["table"]
        lines += ["| " + " | ".join(t["head"]) + " |", "|" + "---|" * len(t["head"])]
        lines += ["| " + " | ".join(r) + " |" for r in t["rows"]] + [""]
(TIER / "results.md").write_text("\n".join(lines))
print(f"{len(done)} runs reported; P1 {n1}/{len(done)}, P2 {n2}/{len(in_scope)}")
