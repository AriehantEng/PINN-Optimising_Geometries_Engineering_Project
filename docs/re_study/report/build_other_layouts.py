"""Every stored result for layouts other than 30/90 and 30/120, in one place.

    python docs/re_study/report/build_other_layouts.py
        -> docs/re_study/report/content_other_layouts.json and docs/re_study/other_layouts.md
    node docs/re_study/report/build_docx.js docs/re_study/report/content_other_layouts.json \\
        docs/re_study/other_layouts.docx

All of it is finite-element data (the PINNs were trained on 30/90 and 30/120 only): the Re_h = 10
optimum map over 15 layouts and 11 shares, the equal-closed-length maps at six Reynolds
numbers, and the six held-out layouts that test Network 3. Read from the stored JSON files.
"""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bifurcation.answer_key import feasibility_limit  # noqa: E402

DOCS = ROOT / "docs"
OUT = DOCS / "re_study" / "report" / "content_other_layouts.json"
MD = DOCS / "re_study" / "other_layouts.md"
blocks = []


def h1(t): blocks.append({"h1": t})
def h2(t): blocks.append({"h2": t})
def p(t): blocks.append({"p": t})
def bullets(items): blocks.append({"bullets": items})


def table(head, rows, widths=None, note=None):
    blocks.append({"table": {"head": head, "rows": rows, "widths": widths}})
    if note:
        blocks.append({"note": note})


def name(angles):
    return f"{angles[0]:g}/{angles[1]:g}"


STUDIED = {"30/90", "30/120"}

# --------------------------------------------------------------------------- #
blocks.append({"title": "Results for the other layouts"})
p("The PINNs (Networks 1 and 2, and the parametric Network 2) were trained only on 30/90 and "
  "30/120, so every result below for other layouts is finite-element data, plus Network 3's "
  "predictions on its held-out layouts. A layout a1/a2 names the two daughters' angles from the "
  "parent's axis; s = w1/w0 is the width split and q daughter 1's share of the flow. J is the "
  "total-pressure drop in ρU_c². s_M is the Murray split, q^(1/3)/(q^(1/3) + (1 − q)^(1/3)).")
table(["Data set", "Layouts", "What it holds", "File"], [
    ["Re_h = 10 optimum map", "15 (13 besides 30/90 and 30/120)",
     "Optimal split s* and J for 11 shares each (165 optima), daughters' outer walls 8 long",
     "docs/bifurcation_re10-shares-all.json"],
    ["Equal-closed-length maps", "15", "s*, departure from s_M, Murray regret for 11 shares at "
     "Re_h = 0.1, 1, 10, 25, 50, 100 (both daughters walled over 8)",
     "docs/bifurcation_closed-re*.json"],
    ["Network 3's held-out test", "6 unseen layouts", "s*, J*, Murray and length-corrected "
     "splits and regrets, Network 3's split and regret, at q = 0.2, 0.5, 0.8",
     "docs/bifurcation_re10-baselines.json"],
], [1800, 1500, 4300, 2600])

# ---- 1. the Re 10 map ------------------------------------------------------ #
h1("1. The Re_h = 10 optimum map: s* for every layout and share")
S = json.loads((DOCS / "bifurcation_re10-shares-all.json").read_text())
geo = S["config"]["geometry"]
p(f"Parent {geo['parent_length']:g} long, daughters' outer walls {geo['daughter_length']:g} long, "
  f"minimum width {geo['min_width']:g}; each optimum located to ±{S['config']['search']['xatol']:g} "
  "by Brent's method on a 17-point scan and re-solved on a finer check mesh.")
rows_by = {}
for r in S["rows"]:
    if r.get("share") is None:
        continue
    rows_by.setdefault(name(r["angles"]), {})[r["share"]] = r
shares = S["config"]["shares"]
layouts = [name(a) for a in S["config"]["angles"]]
order = sorted(layouts, key=lambda k: (sum(map(float, k.split("/"))), k))
murray = {q: q ** (1 / 3) / (q ** (1 / 3) + (1 - q) ** (1 / 3)) for q in shares}


def cell(r):
    if not r["feasible"]:
        return "—"
    mark = "‡" if feasibility_limit(r) is not None else ("†" if r["on_bound"] else "")
    return f"{r['best_split']:.3f}{mark}"


rows = [["s_M (Murray)"] + [f"{murray[q]:.3f}" for q in shares]]
for k in order:
    rows.append([k + (" (PINN layout)" if k in STUDIED else "")]
                + [cell(rows_by[k][q]) for q in shares])
table(["Layout \\ q"] + [f"{q:g}" for q in shares], rows, [1500] + [700] * len(shares),
      "Optimal split s*. † on the search range's bound (0.1 or 0.9); ‡ at the geometry's "
      "feasibility limit (the next split is refused), so a bound, not a minimum. Layouts in "
      "order of total turning angle a1 + a2.")
rows = []
for k in order:
    r = rows_by[k][0.5]
    rows.append([k, f"{r['best_split']:.4f}", f"{r['total_pressure_drop']:.4f}",
                 f"{r['equal_widths_drop']:.4f}", f"{100 * r['saving_vs_equal_widths']:.1f}%",
                 f"{r['outlet1_pressure_offset']:+.3f}"])
table(["Layout", "s* at q = 0.5", "J*", "J at equal widths", "Saving vs equal widths",
       "Outlet 1 pressure offset"], rows, [1300, 1400, 1300, 1600, 1800, 1800],
      "At q = 0.5. The pressure offset is the Lagrange multiplier that makes outlet 1 deliver its "
      "prescribed share.")

# ---- 2. closed-length maps across Re --------------------------------------- #
h1("2. The equal-closed-length maps across Reynolds numbers")
p("Here both daughters have both walls over a length of 8, so the friction law's optimum is "
  "exactly s_M and s* − s_M isolates the junction and entrance effect.")
maps = [json.loads((DOCS / f"bifurcation_closed-re{r}.json").read_text()) for r in
        ("0.1", "1", "10", "25", "50", "100")]
J = json.loads((DOCS / "re_study" / "closed-junction-all.json").read_text())
recs = {m["reynolds"]: m["records"] for m in J["maps"]}
res = sorted(recs)
rows = []
for k in order:
    row = [k]
    for re_ in res:
        r = next(x for x in recs[re_] if name(x["angles"]) == k and abs(x["share"] - 0.5) < 1e-9)
        row.append(f"{r['best_split']:.4f}" + ("" if r["trusted"] else " ⚠"))
    rows.append(row)
table(["Layout"] + [f"Re_h = {r:g}" for r in res], rows, [1500] + [1100] * len(res),
      "s* at q = 0.5 (s_M = 0.5000). ⚠ next to a split with no steady solution.")
rows = []
for k in order:
    row = [k]
    for re_ in res:
        rs = [x for x in recs[re_] if name(x["angles"]) == k and x["trusted"]]
        worst = max(rs, key=lambda x: abs(x["departure"]))
        row.append(f"{worst['departure']:+.4f} ({worst['share']:g})")
    rows.append(row)
table(["Layout"] + [f"Re_h = {r:g}" for r in res], rows, [1500] + [1100] * len(res),
      "Largest signed departure s* − s_M over the eleven shares (the share in brackets).")
rows = []
for k in order:
    row = [k]
    for re_ in res:
        rs = [x for x in recs[re_] if name(x["angles"]) == k and x["trusted"]]
        row.append(f"{100 * max(x['murray_regret'] for x in rs):.3f}%")
    rows.append(row)
table(["Layout"] + [f"Re_h = {r:g}" for r in res], rows, [1500] + [1100] * len(res),
      "Largest Murray regret, J(s_M)/J* − 1, over the eleven shares.")

# ---- 3. Network 3's held-out layouts --------------------------------------- #
h1("3. Six held-out layouts: Network 3 against the simple rules")
B = json.loads((DOCS / "bifurcation_re10-baselines.json").read_text())
p("Network 3 was trained on the 165 optima of Section 1 and tested on six layouts it never saw, "
  "each at three shares, every case optimised separately by finite elements (Re_h = 10, "
  "daughters 8 long). Regret = J(s)/J* − 1 for adopting split s instead of s*.")
rows = []
for x in sorted(B["heldout"], key=lambda x: (sum(x["angles"]), x["angles"], x["share"])):
    rows.append([name(x["angles"]), f"{x['share']:g}", f"{x['best_split']:.4f}",
                 f"{x['best_drop']:.4f}", f"{x['murray_split']:.4f}",
                 f"{100 * x['murray_regret']:.4f}%", f"{x['length_split']:.4f}",
                 f"{100 * x['length_regret']:.4f}%", f"{x['network_split']:.4f}",
                 f"{100 * x['network_regret']:.4f}%"])
table(["Layout", "q", "s* (FE)", "J*", "s_M", "Murray regret", "s_L", "s_L regret",
       "Network 3", "Network 3 regret"], rows,
      [900, 500, 1000, 900, 900, 1100, 900, 1100, 1000, 1200])
# negative regrets are search noise (the rule's split scored a hair under J*) and count as 0,
# as in docs/bifurcation_re10-baselines.md
mean = lambda key: statistics.mean(max(0.0, x[key]) for x in B["heldout"])
worst = lambda key: max(x[key] for x in B["heldout"])
table(["Rule", "Mean regret", "Worst regret"], [
    ["Murray split s_M", f"{100 * mean('murray_regret'):.4f}%", f"{100 * worst('murray_regret'):.4f}%"],
    ["Length-corrected split s_L", f"{100 * mean('length_regret'):.4f}%",
     f"{100 * worst('length_regret'):.4f}%"],
    ["Network 3", f"{100 * mean('network_regret'):.4f}%", f"{100 * worst('network_regret'):.4f}%"],
], [3000, 2000, 2000], "Over the 18 held-out cases; negative regrets (search noise: the rule's split "
   "scored a hair under J*) count as 0. Network 3's splits are stored to four decimals; at full "
   "precision its worst regret was 0.077%, the paper's figure.")

# ---- 4. what is not there -------------------------------------------------- #
h1("4. What there is no data for")
bullets([
    "No PINN of any kind was trained on a layout other than 30/90 and 30/120, so there are no PINN "
    "field errors, pressure-drop errors or PINN optima for the other 13 layouts.",
    "The Reynolds-number answer keys (s_FE with argmin bounds and reference-mesh J at s = 0.35–0.65) "
    "exist only for 30/90 and 30/120.",
    "Flow-field pictures exist for the 15 layouts at their Re_h = 10 optima at q = 0.5 and 0.2 "
    "(docs/fields/gallery_q*.png).",
])

OUT.write_text(json.dumps(blocks, indent=1, ensure_ascii=False))

# the same content as Markdown
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
    elif "table" in b:
        t = b["table"]
        lines += ["| " + " | ".join(t["head"]) + " |", "|" + "---|" * len(t["head"])]
        lines += ["| " + " | ".join(r) + " |" for r in t["rows"]] + [""]
MD.write_text("\n".join(lines))
print(f"{len(blocks)} blocks -> {OUT}; {MD}")
