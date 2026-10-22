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
    "Confound: every off-centre case also had a smaller corner-mode cutoff (0.10 to 0.35, by the "
    "pre-registered rule, because its narrow daughter is narrow) than 0/120 (0.40) or the "
    "earlier runs (0.5). Tier A cannot separate the share from the cutoff; a run at an "
    "off-centre share with the cutoff held at 0.40 or more would, where the geometry admits it "
    "(it does not at q = 0.2 or 0.8 with this range).",
    "P2, a property of the reference solution: s_L lies within 0.003 of the optimum at q = 0.35 "
    "and 0.65 at Re_h = 0.1, q = 0.35 at 10, and for 15/15 at both; it does not at q = 0.2 and "
    "0.8 for 30/90 (0.004 to 0.006), nor at q = 0.65 at Re_h = 10 (0.0031). The length correction "
    "loses accuracy as the share moves from 0.5 in a layout with unequal lengths.",
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
                 f"{t['wall_seconds'] / 60:.0f}", "yes" if t["converged"] else "no"])
table(["Layout", "q", "Re_h", "Corner cutoff", "Drop error: low end / middle / high end",
       "Largest |drop error|", "u error at s*", "Held-out loss", "Min", "Converged"], rows,
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
