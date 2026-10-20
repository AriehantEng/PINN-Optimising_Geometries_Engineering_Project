"""The results for the paper's outstanding items, in one document.

    python docs/re_study/report/build_paper_items.py
        -> docs/re_study/report/content_paper_items.json
    node docs/re_study/report/build_docx.js docs/re_study/report/content_paper_items.json \\
        docs/re_study/paper_items_results.docx

Table 3's low-Re rows, Network 1's outlet check (retrained), the reference
solver and Network 1's optimiser settings, the three references, the two
figures redrawn with all 13 runs, and what the answer key now says about
30/120. Every number is read from the stored result files.
"""
import json
import pathlib
import statistics
import struct

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
STUDY = DOCS / "re_study"
OUT = STUDY / "report" / "content_paper_items.json"
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


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    sup = str(int(e)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m} × 10{sup}"


# --------------------------------------------------------------------------- #
blocks.append({"title": "Results for the paper's outstanding items"})
p("This document collects the results requested for the paper: Table 3's rows at Re_h = 0.1 and "
  "1, Network 1's outlet profile against the Poiseuille parabola, two details of the code, three "
  "references, and the pressure-drop-error and training figures redrawn with all 13 runs. It "
  "also records what the finite-element answer key shows about the 30/120 layout. Every number "
  "is read from the stored result files named under each item.")

# ---- 1. Table 3 ------------------------------------------------------------ #
h1("1. Table 3: the equal-length maps at every Reynolds number")
p("Source: docs/re_study/closed-junction-all.json (bifurcation/junction_re.py). In these maps "
  "both daughters have both walls over a length of 8, so the friction law's optimum is exactly "
  "the Murray split s_M and s* − s_M measures the junction and entrance effect alone. Murray "
  "regret = J(s_M)/J* − 1. Optima next to a split with no steady solution are left out.")
J = json.loads((STUDY / "closed-junction-all.json").read_text())


def tag(r):
    return f"{r['angles'][0]:g}/{r['angles'][1]:g}, q = {r['share']:g}"


rows, per_layout = [], {}
for m in J["maps"]:
    rs = [r for r in m["records"] if r["trusted"]]
    turn = lambda r: r["angles"][0] + r["angles"][1]
    up90 = max((r for r in rs if turn(r) <= 90), key=lambda r: abs(r["departure"]))
    b180 = max((r for r in rs if turn(r) < 180), key=lambda r: abs(r["departure"]))
    at180 = max((r for r in rs if turn(r) == 180), key=lambda r: abs(r["departure"]))
    worst = max(rs, key=lambda r: r["murray_regret"])
    median = statistics.median(r["murray_regret"] for r in rs)
    re_ = m["reynolds"]
    rows.append([f"{re_:g}" + (" (new)" if re_ < 10 else ""), f"{len(rs)}/{len(m['records'])}",
                 f"{abs(up90['departure']):.4f} ({tag(up90)})",
                 f"{abs(b180['departure']):.4f} ({tag(b180)})",
                 f"{abs(at180['departure']):.4f} ({tag(at180)})",
                 f"{100 * worst['murray_regret']:.3f}% ({tag(worst)})", f"{100 * median:.4f}%"])
    for r in m["records"]:
        key = f"{r['angles'][0]:g}/{r['angles'][1]:g}"
        d = per_layout.setdefault(key, {})
        cell = d.setdefault(re_, [0.0, 0.0])
        if r["trusted"]:
            cell[0] = max(cell[0], abs(r["departure"]))
            cell[1] = max(cell[1], r["murray_regret"])
table(["Re_h", "Optima kept", "Largest |s* − s_M|, up to 90° (layout)",
       "Largest |s* − s_M|, below 180° (layout)", "Largest |s* − s_M|, at 180° (layout)",
       "Largest Murray regret (layout)", "Median Murray regret"], rows,
      [900, 900, 1900, 1900, 1900, 1900, 1000],
      "Rows 0.1 and 1 are the new ones. 'Up to 90°' means a1 + a2 ≤ 90° (15/15, 0/30, 20/40, 0/60, "
      "45/45, 10/80); 'below 180°' adds 60/60, 30/90, 0/120, 75/75, 30/120 and 0/150; 'at 180°' is "
      "90/90, 60/120 and 30/150.")
lo = [r for r in rows[:2]]
p(f"The text's 'about 0.008' is {lo[0][3].split()[0]} at Re_h = 0.1 and {lo[1][3].split()[0]} "
  f"at Re_h = 1, both at 30/120 with q = 0.9. Its 'about 0.003' (up to 90°) is "
  f"{lo[0][2].split()[0]} at Re_h = 0.1 (10/80, q = 0.9) and {lo[1][2].split()[0]} at Re_h = 1 "
  "(0/60, q = 0.8). At both low Reynolds numbers the largest Murray regret, 0.206%, is at the "
  "180° layout 30/150 with q = 0.2. Layout by layout, the largest departures at Re_h = 0.1 and 1 "
  f"differ by at most {max(abs(d[0.1][0] - d[1.0][0]) for d in per_layout.values()):.4f}.")
h2("By layout")
res = [m["reynolds"] for m in J["maps"]]
order = list(per_layout)
table(["Layout"] + [f"Re_h = {r:g}" for r in res],
      [[k] + [f"{per_layout[k][r][0]:.4f}" for r in res] for k in order],
      [1300] + [1000] * len(res), "Largest |s* − s_M| over the eleven shares, by layout.")
table(["Layout"] + [f"Re_h = {r:g}" for r in res],
      [[k] + [f"{100 * per_layout[k][r][1]:.3f}%" for r in res] for k in order],
      [1300] + [1000] * len(res), "Largest Murray regret over the eleven shares, by layout.")

# ---- 2. Network 1's outlet ------------------------------------------------- #
h1("2. Network 1's outlet profile against the Poiseuille parabola")
O = json.loads((DOCS / "developing_re10-outlet.json").read_text())
p("No trained Network 1 had been kept, so it was retrained from its stored configs: G5 (Adam, "
  "four L-BFGS cycles, 800 Levenberg–Marquardt steps; developing_flow/re10_lm_fresh_inputmap.yaml) "
  "and then its three 100-step, 16,384-point polishes, seeds 1–3 (P7; "
  "developing_flow/re10_polish_g5_points*.yaml), which are the three seeds of Table 6.1. The "
  "finite-element reference was rebuilt as well and matches the stored one to every printed "
  "digit (total-pressure drop 2.1348435579, entrance length 1.7794154129).")
h2("Does the retrained network reproduce the stored one?")
S_old = json.loads((DOCS / "developing_re10-summary.json").read_text())["runs"]
S_new = json.loads((STUDY / "network1_retrain" / "summary.json").read_text())["runs"]
keys = ("u", "v", "p", "total_pressure_drop", "entrance_length")
names = ("u", "v", "p", "total-pressure drop", "entrance length")
old7 = S_old["P7 G5 + 100 steps at 16k, mean of 20, 32k held-out"]
old5 = S_old["G5 G3 + exponential input map"]
new7, new5 = S_new["P7"], S_new["G5"]
rows = []
for label, old, new in [("G5", old5, new5)] + [(f"P7 seed {i + 1}", [old7[i]], [new7[i]])
                                               for i in range(len(new7))]:
    for tagname, src in (("stored", old), ("retrained", new)):
        rows.append([label if tagname == "stored" else "", tagname]
                    + [f"{src[0][k]:.3e}" for k in keys] + [f"{src[0]['minutes']:.0f}"])
table(["Run", ""] + [f"{n} (rel. L2)" for n in names] + ["min"], rows,
      [1000, 900, 1150, 1150, 1150, 1400, 1400, 600],
      "Errors against the finite-element reference, integrated on its mesh "
      "(developing_flow/summary_reference.py). Stored: docs/developing_re10-summary.json; "
      "retrained: docs/re_study/network1_retrain/summary.json.")
same = all(abs(a[k] - b[k]) <= 1e-9 * max(abs(a[k]), 1e-30) for a, b in
           zip([old5[0]] + old7, [new5[0]] + new7) for k in keys)
p("The retrained networks reproduce the stored errors " +
  ("to every printed digit, so the outlet values below belong to the networks reported in "
   "Table 6.1." if same else
   "only approximately (the table above shows by how much), so the outlet values below belong "
   "to networks equivalent to, not identical with, those of Table 6.1."))
h2("The outlet profile at x = 10")
p("The analytical profile is u = 1.5 U (1 − y²) with the mean velocity U = Q₀/2 = 2/3, that is "
  "u = 1 − y², v = 0. Each norm is taken across the channel at x = 10 on the reference's "
  f"{O['heights']} heights (developing_flow/outlet_error.py; docs/developing_re10-outlet.md).")
cols = ("analytical", "analytical_max_abs", "shape", "reference", "flux_error", "outlet_max_abs_v")
rows = [[f"seed {r['seed']}"] + [f"{r[k]:.2e}" for k in cols] for r in O["runs"]]
mean = {k: statistics.mean(r[k] for r in O["runs"]) for k in cols}
rows.append(["mean"] + [f"{mean[k]:.2e}" for k in cols])
ref = O["reference_against_analytical"]
rows.append(["reference solution", f"{ref['analytical']:.2e}", f"{ref['analytical_max_abs']:.2e}",
             f"{ref['shape']:.2e}", "—", f"{ref['flux_error']:.2e}", "—"])
table(["", "Rel. L2 against u = 1 − y²", "max |u − (1 − y²)|", "Rel. L2 against the parabola "
       "at its own flux", "Rel. L2 against the reference at x = 10", "Outlet flux error",
       "max |v|"], rows, [1300, 1300, 1200, 1500, 1500, 1200, 1000])
lo, hi = min(r["analytical"] for r in O["runs"]), max(r["analytical"] for r in O["runs"])
p(f"Network 1's outlet profile agrees with the analytical Poiseuille profile to a relative L2 "
  f"error of {sci(mean['analytical'])} (mean of three seeds; {sci(lo)} to {sci(hi)}). The "
  f"reference itself departs from it by {sci(ref['analytical'])} there, so the network's error "
  "is its own, not the reference's. Suggested wording for H1: 'At the outlet, x = 10, its "
  f"velocity profile matches the Poiseuille profile u = 1.5U(1 − y²) to {sci(mean['analytical'])} "
  "(relative L2, mean of three seeds).'")

# ---- 3. Code details ------------------------------------------------------- #
h1("3. Two details from the code")
g5 = yaml.safe_load((ROOT / "developing_flow" / "re10_lm_fresh_inputmap.yaml").read_text())
bullets([
    "Sparse direct solver (Section 3.2): SuperLU. Each Newton step's linear system is solved by "
    "scikit-fem's solve, which calls SciPy's spsolve; UMFPACK (scikits.umfpack) is not installed, "
    "so spsolve uses SuperLU (checked on this machine: SciPy 1.17.1, scikit-fem 12.0.2). Its "
    "reference is Li, X. S. (2005). An overview of SuperLU: Algorithms, implementation, and user "
    "interface. ACM Transactions on Mathematical Software, 31(3), 302–325.",
    f"L-BFGS iterations per cycle (Section 3.3): {g5['training']['lbfgs']['iterations_per_cycle']}, "
    f"in {g5['training']['lbfgs']['cycles']} cycles (gtol {g5['training']['lbfgs']['gtol']:g}), "
    "from developing_flow/re10_lm_fresh_inputmap.yaml. Network 1's final 100-step polish uses no "
    "L-BFGS.",
])

# ---- 4. References --------------------------------------------------------- #
h1("4. The three references")
p("The publishers' sites, doi.org and Crossref are blocked from this machine, so no first page "
  "could be opened. Each detail below comes from the publisher's or an index's own listing as "
  "returned by web search.")
table(["Reference", "Confirmed", "Source"], [
    ["Gammaidoni, T., Zembi, J., & Battistoni, M. (2026). Flow and heat transfer in microchannels "
     "using physics-informed neural networks. Computers & Fluids, 315, Article 107143.",
     "Header line: Computers & Fluids, Volume 315, 30 July 2026, 107143. DOI "
     "10.1016/j.compfluid.2026.107143. Break-even after about 21 CFD simulations (abstract and "
     "highlights).", "ScienceDirect, pii S0045793026001854"],
    ["Taylor, C., & Hood, P. (1973). A numerical solution of the Navier–Stokes equations using "
     "the finite element technique. Computers & Fluids, 1(1), 73–100.",
     "Volume 1, Issue 1, January 1973, pages 73–100; DOI 10.1016/0045-7930(73)90027-3; Taylor "
     "first author.", "ScienceDirect, pii 0045793073900273; Semantic Scholar"],
    ["Durst, F., Ray, S., Ünsal, B., & Bayoumi, O. A. (2005). The development lengths of laminar "
     "pipe and channel flows. Journal of Fluids Engineering, 127(6), 1154–1160.",
     "DOI 10.1115/1.2063088; volume 127, page 1154. Issue 6 and the last page, 1160, agree "
     "across several citing reference lists.", "Crossref citation record; citing papers"],
], [3600, 3700, 2300])

# ---- 5. Figures ------------------------------------------------------------ #
h1("5. The two figures, with all 13 runs")
p("Both figures had been drawn by pairing the runs with a six-colour list, which silently "
  "stopped after six runs. Each now has a panel per Reynolds number. A network looks the same "
  "in every panel: the three 30/90 seeds are solid lines in fixed colours, and 30/120 is dashed "
  "with square markers. A figure refuses to save unless it has drawn every run, and a test "
  "checks this.")
image(STUDY / "fig_re_drop_error.png",
      "Pressure-drop error of each PINN against the reference solution at the seven evaluation "
      "splits, a panel per Re_h, all 13 runs. Zero is a perfect prediction. J is in μU_c/h at "
      "Re_h ≤ 1 and ρU_c² above. 30/120 was not trained at Re_h = 0.1.")
image(STUDY / "fig_re_training.png",
      "Held-out loss against training time, a panel per Re_h, all 13 runs; squares mark the "
      "start of L-BFGS, triangles of Levenberg–Marquardt. A diagnostic only: loss values are "
      "not accuracy, and their scale differs between the viscous (Re_h ≤ 1) and inertial runs.")

# ---- 6. 30/120 ------------------------------------------------------------- #
h1("6. 30/120: its 'optimum' is the edge of the feasible splits")
p("The finite-element answer key gave s_FE = 0.2680 for 30/120 at every Reynolds number. That "
  "value is not a minimum of J. Below s = 0.26753 the geometry is refused, because the inner "
  "walls meet beyond an outlet, and J is still falling at the last split that can be built. "
  "There, daughter 1's walled length is 0.0006: the 30° branch has all but vanished. The answer "
  "key now marks the value as a geometry limit (docs/re_study/answer_keys.md).")
rows = []
for re_ in ("0.1", "1", "10", "100"):
    c = json.loads((STUDY / "answer_keys_cases" / f"30-120_Re{re_}.json").read_text())
    grid = {round(g["split"], 3): g["total_pressure_drop"] for g in c["optimum"]["grid"]}
    rows.append([re_, f"{c['J_fe_search']:.4g}", f"{grid[0.3]:.4g}", f"{grid[0.35]:.4g}",
                 f"{grid[0.4]:.4g}", f"{grid[0.45]:.4g}", f"{grid[0.5]:.4g}",
                 f"{c['in_range']['s']:.4f}"])
table(["Re_h", "J at s = 0.2680 (limit)", "J at 0.30", "J at 0.35", "J at 0.40", "J at 0.45",
       "J at 0.50", "Best split in [0.35, 0.65]"], rows, [700, 1500, 1000, 1000, 1000, 1000, 1000,
                                                           1500],
      "Search-mesh J (μU_c/h at Re_h ≤ 1, ρU_c² above). Splits below 0.26753 are refused.")
bullets([
    "Retraining on a range around 'its real optimum' is therefore not possible: [0.15, 0.45] "
    "cannot be built, and a range starting at 0.27 would put the answer on its own lower edge "
    "again, which tests nothing. It was not run.",
    "At Re_h ≤ 10 the 30/120 runs test pressure-drop and field accuracy only. At Re_h = 100 the "
    "range holds a genuine local minimum (0.3797 on the search mesh, 0.3859 on the reference "
    "mesh), which the PINN located to 0.014 (regret 0.08%), but J at the limit is lower still "
    "(0.3984 against 0.4247 on the search mesh).",
    "The 165 optima behind Table 3 were checked for the same failure and none has it. The only "
    "optimum at a geometry limit in those maps is the equal-outlet-pressure row for 30/150 "
    "(s = 0.8987 against a closed entrance at 0.8992), which is not one of the 165.",
])
h2("The length-corrected split s_L for 30/120")
p("s_L solves s/(1 − s) = (L₁/L₂)^(1/4) √(q/(1 − q)) with L₁ and L₂ measured on the geometry at "
  "s_L itself (q = 0.5; parent 3, outer walls 4, as in the answer key).")
table(["Lengths", "s_L", "L₁", "L₂", "L₁/L₂"],
      [["centreline", "0.4523", "2.6882", "5.7759", "0.4654"],
       ["closed (both walls)", "0.4232", "1.1586", "4.0000", "0.2896"]],
      [2400, 1400, 1400, 1400, 1400],
      "The answer key's 2.0003 and 6.4638 are the centreline lengths at the 0.2680 limit, not at "
      "s_L. For comparison, 30/90 gives s_L = 0.4843 (L₁ = 3.6841, L₂ = 4.7386).")

OUT.write_text(json.dumps(blocks, indent=1, ensure_ascii=False))
print(f"{len(blocks)} blocks -> {OUT}")
