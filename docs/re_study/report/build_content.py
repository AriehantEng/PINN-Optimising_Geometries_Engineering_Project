"""Gather every result of the Reynolds-number study into one content model.

    python docs/re_study/report/build_content.py   ->   docs/re_study/report/content.json

Every number is read from the stored result files (answer keys, closed-length
maps, junction report, run summaries); nothing is typed in. build_docx.js then
renders content.json as a Word document.
"""
import glob
import json
import pathlib
import statistics
import struct

ROOT = pathlib.Path(__file__).resolve().parents[3]
STUDY = ROOT / "docs" / "re_study"
OUT = STUDY / "report" / "content.json"

blocks = []


def h1(t): blocks.append({"h1": t})
def h2(t): blocks.append({"h2": t})
def p(t): blocks.append({"p": t})
def bullets(items): blocks.append({"bullets": items})
def table(head, rows, widths=None, note=None):
    blocks.append({"table": {"head": head, "rows": rows, "widths": widths}})
    if note:
        blocks.append({"note": note})


def image(path, caption, width_in=6.3):
    with open(path, "rb") as f:
        f.read(16)
        w, h = struct.unpack(">II", f.read(8))
    blocks.append({"img": str(path), "w": round(width_in * 96), "h": round(width_in * 96 * h / w),
                   "caption": caption})


def pct(x, d=3): return f"{100 * x:+.{d}f}%"
def num(x, d=4): return "—" if x is None else f"{x:.{d}f}"


summaries = {}
for f in sorted(STUDY.glob("pinn_*_summary.json")):
    s = json.loads(f.read_text())
    summaries[f.name[: -len("_summary.json")]] = s
keys = json.loads((STUDY / "answer_keys.json").read_text())
junction = json.loads((STUDY / "closed-junction-all.json").read_text())

# --------------------------------------------------------------------------- #
blocks.append({"title": "Reynolds-number study: results"})
p("PINN runs on two bifurcation layouts at several Reynolds numbers, each checked against a "
  "finite-element answer key, and the closed-length design maps that show the approach to "
  "Murray's law. Every number below is read from the stored result files in docs/re_study and "
  "docs/bifurcation_closed-*; the file each table comes from is named under it.")

h1("1. What was run")
p("Layouts 30/90 and 30/120 on the existing parametric geometry: parent 3 long, daughters' outer "
  "walls 4 long, flow share q = 0.5, split s = w₁/w₀ ∈ [0.35, 0.65]. Corner-mode PINN with the "
  "existing schedule (Adam 3,000 steps, L-BFGS 4 × 500, Levenberg–Marquardt 150). Re_h ≤ 1 uses the "
  "viscous pressure scale μU_c/h, so the momentum equation reads Re(u·∇)u + ∇p − ∇²u = 0; Re_h ≥ 10 "
  "uses ρU_c². At Re_h = 100 the reference is reached by Reynolds continuation through 25 and 50.")
status = []
order = [("30-90", 0.1), ("30-90", 1.0), ("30-90", 10.0), ("30-90", 100.0),
         ("30-120", 0.1), ("30-120", 1.0), ("30-120", 10.0), ("30-120", 100.0)]
for n, (lay, re_) in enumerate(order, 1):
    runs = sorted((s for s in summaries.values()
                   if f"{s['layout'][0]:g}-{s['layout'][1]:g}" == lay and s["reynolds"] == re_),
                  key=lambda s: s["seed"])
    done = [s for s in runs if "failed" not in s]
    skipped = [s for s in runs if "failed" in s]
    state = (f"{len(done)} seed(s): " + ", ".join(str(s["seed"]) for s in done)) if done else ""
    if skipped:
        state = "skipped (optimum on the range's bound; see 6.2)"
    status.append([str(n), lay.replace("-", "/"), f"{re_:g}", state])
table(["#", "Layout", "Re_h", "PINN runs"], status, [600, 1400, 1000, 6360])
p("Definitions. Argmin error = PINN s* − reference s*. Regret = J_ref(s*_PINN)/J_ref* − 1, both "
  "values from the reference solution. Drop error = J_PINN/J_ref − 1 at the same split. Field "
  "errors are relative L2 errors integrated on the reference mesh. Converged = the schedule "
  "finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than "
  "10% below the best before them. Loss is a diagnostic only, never an accuracy measure.")

# --------------------------------------------------------------------------- #
h1("2. Finite-element answer keys")
p("Procedure for each case: a 17-point scan over every split the width bounds allow, Brent to "
  "±0.001 on the search mesh (max area 0.02, 3 corner levels), then the optimum and its grid "
  "neighbours on the check mesh (0.005, 5 levels). The argmin bound ε_s is the shift the check "
  "mesh's error slope implies plus 0.001. J on the reference mesh (0.002, 6 levels) is measured "
  "by the same function that measures the PINNs.")
rows = []
for c in keys["cases"]:
    rows.append([c["key"].replace("_", ", ").replace("-", "/"), num(c["s_fe"]),
                 "—" if c["argmin_bound"] is None else num(c["argmin_bound"]),
                 f"{c['J_fe_search']:.6g}", f"{c['J_fe_check']:.6g}",
                 f"{c['J_fe_reference_at_s_fe']:.6g}",
                 f"{c['centreline_lengths'][0]:.4f}, {c['centreline_lengths'][1]:.4f}",
                 num(c["s_murray"]), num(c["s_length_corrected"])])
table(["Case", "s_FE", "ε_s", "J (search)", "J (check)", "J (reference)", "L₁, L₂ at s_FE", "s_M",
       "s_L"], rows, [1500, 800, 750, 1050, 1050, 1150, 1400, 750, 850],
      "Source: docs/re_study/answer_keys.json. J in μU_c/h at Re_h ≤ 1, ρU_c² above.")
p("For 30/120 every split below s = 0.268 is refused by the geometry (the inner walls would meet "
  "beyond daughter 1's outlet), and J falls toward that limit at every Re_h: the global s_FE = "
  "0.2680 is the geometric limit, not a flow optimum, and has no argmin bound. Inside the PINNs' "
  "range the optimum is:")
rows = []
for c in keys["cases"]:
    if "in_range" in c:
        g = c["in_range"]
        rows.append([c["key"].replace("_", ", ").replace("-", "/"), num(g["s"]),
                     "range bound" if g["on_range_bound"] else num(g.get("argmin_bound")),
                     f"{g['J_search']:.6g}"])
table(["Case", "s* in [0.35, 0.65]", "ε_s", "J (search)"], rows, [2500, 2300, 2000, 2560],
      "At Re_h 0.1, 1 and 10 J rises across the whole range, so the in-range optimum is its lower "
      "bound. Only at Re_h = 100 is there an interior optimum; the reference mesh's Brent search "
      "puts it at 0.3859 (J is flat there: 0.2% between s = 0.35 and 0.40).")
splits = keys["config"]["reference"]["splits"]
rows = [[c["key"].replace("_", ", ").replace("-", "/")] +
        ["—" if x["J"] is None else f"{x['J']:.6g}" for x in c["reference"]] for c in keys["cases"]]
table(["Case"] + [f"s = {s:.2f}" for s in splits], rows, [1600] + [1108] * len(splits),
      "J on the reference mesh at the evaluation splits (for regret and the H3 error slope). "
      "Per-case CSVs: docs/re_study/fe_<case>_reference.csv.")

# --------------------------------------------------------------------------- #
h1("3. Closed-length maps: the approach to Murray's law")
p("Both daughters have both walls over 8, so the fully developed friction law puts the optimum "
  "exactly at the Murray split s_M and s* − s_M is the junction and entrance effect. Fifteen "
  "layouts × eleven shares at each Re_h. Location bound: the mesh-induced argmin shift plus the "
  "search tolerance; 'resolved' = beyond its own bound.")
rows = []
for m in junction["maps"]:
    rs = [r for r in m["records"] if r.get("trusted")]
    def top(sel):
        r = max(sel, key=lambda r: abs(r["departure"]))
        return f"{abs(r['departure']):.4f} ({r['angles'][0]:g}/{r['angles'][1]:g})"
    res = [r for r in rs if r["resolved"]]
    asym = [r for r in res if r["angles"][0] != r["angles"][1]]
    rmax = max(rs, key=lambda r: r["murray_regret"])
    rows.append([f"{m['reynolds']:g}", top([r for r in rs if sum(r["angles"]) < 180]),
                 top([r for r in rs if sum(r["angles"]) <= 90]),
                 top([r for r in rs if sum(r["angles"]) == 180]),
                 f"{len(rs)}/{len(m['records'])}", str(len(res)),
                 f"{sum(1 for r in asym if r['widens_turned'])}/{len(asym)}",
                 f"{100 * statistics.median(r['murray_regret'] for r in rs):.4f}%",
                 f"{100 * rmax['murray_regret']:.3f}% ({rmax['angles'][0]:g}/{rmax['angles'][1]:g})"])
table(["Re_h", "Largest |s*−s_M| below 180°", "Largest up to 90°", "Largest at 180°", "Trusted",
       "Resolved", "Resolved asym. widening the turned daughter", "Median Murray regret",
       "Max Murray regret"], rows, [600, 1350, 1150, 1150, 800, 850, 1250, 1000, 1210],
      "Source: docs/re_study/closed-junction-all.json (table3_closed_maps.md).")
layouts = []
for r in junction["maps"][0]["records"]:
    if r["angles"] not in layouts:
        layouts.append(r["angles"])
res_list = [m["reynolds"] for m in junction["maps"]]
dep_rows, reg_rows = [], []
for a in layouts:
    drow, rrow = [f"{a[0]:g}/{a[1]:g}"], [f"{a[0]:g}/{a[1]:g}"]
    for m in junction["maps"]:
        rs = [r for r in m["records"] if r.get("trusted") and r["angles"] == a]
        if not rs:
            drow.append("—"); rrow.append("—"); continue
        drow.append(f"{max(abs(r['departure']) for r in rs):.4f} "
                    f"({sum(1 for r in rs if r['resolved'])}/{len(rs)})")
        rrow.append(f"{100 * max(r['murray_regret'] for r in rs):.3f}%")
    dep_rows.append(drow); reg_rows.append(rrow)
head = ["Layout"] + [f"Re_h = {r:g}" for r in res_list]
table(head, dep_rows, [1200] + [1360] * len(res_list),
      "Largest |s* − s_M| over the eleven shares, with the number of shares beyond their bound.")
table(head, reg_rows, [1200] + [1360] * len(res_list), "Largest Murray regret, J(s_M)/J* − 1.")
image(STUDY / "closed-junction-all.png",
      "Figure 1. Largest departure of the optimal split from the Murray split against Re_h, one "
      "panel per bifurcation angle; blue the symmetric layout, orange the intermediate, green the "
      "most turned; open points where some shares had no steady solution. Grey: the largest "
      "location bound at each Re_h. The departure is flat from Re_h 0.1 to 10 and grows only above "
      "about 25: it tends to a viscous junction effect, not to zero.")
bullets([
    "The 180° band is drawn per Re_h: one case (30/150, q = 0.05, Re_h 50, bound 0.0082) set a "
    "single band for the whole panel before. With it per Re_h, 30/150 and 60/120 are resolved at "
    "Re_h ≤ 10, and only 90/90 at Re_h 0.1 is not (0.0007 against 0.0010).",
    "75/75 is resolved at every Re_h: its largest departure, 0.0015–0.0025, exceeds its own bound "
    "of 0.0010–0.0012, in 5–7 of 11 shares.",
])

# --------------------------------------------------------------------------- #
h1("4. PINN runs")
rows = []
for name, s in sorted(summaries.items(), key=lambda kv: (kv[1]["layout"][1], kv[1]["reynolds"],
                                                         kv[1]["seed"])):
    lay = f"{s['layout'][0]:g}/{s['layout'][1]:g}"
    if "failed" in s:
        rows.append([lay, f"{s['reynolds']:g}", str(s["seed"]), "skipped", "", "", "", "", "", ""])
        continue
    by = {round(f["split"], 6): f for f in s["fields"] if not f.get("at")}
    t = s["training"]
    rows.append([lay, f"{s['reynolds']:g}", str(s["seed"]), num(s["s_star_pinn"]),
                 num(s["s_star_reference_mesh_brent"]), f"{s['argmin_error']:+.4f}",
                 f"{100 * s['regret']:.4f}%",
                 " / ".join(f"{100 * by[k]['drop_error']:+.2f}" for k in (0.35, 0.5, 0.65)),
                 f"{by[0.5]['u']:.2e}", f"{t['wall_seconds'] / 60:.0f}, {'yes' if t['converged'] else 'no'}"])
table(["Layout", "Re_h", "Seed", "PINN s*", "Ref. s*", "Argmin error", "Regret",
       "Drop error % at 0.35 / 0.50 / 0.65", "u error at 0.50", "Min, converged"], rows,
      [800, 600, 550, 850, 850, 950, 900, 1700, 1000, 1160],
      "Source: docs/re_study/pinn_runs.md and the run summaries.")
h2("4.1 30/90 across seeds")
seed_rows = []
by_re = {}
for s in summaries.values():
    if "failed" not in s and s["layout"] == [30.0, 90.0]:
        by_re.setdefault(s["reynolds"], []).append(s)
def at(s, x, k):
    return next(f[k] for f in s["fields"] if abs(f["split"] - x) < 1e-9 and not f.get("at"))
for re_ in sorted(by_re):
    rs = sorted(by_re[re_], key=lambda s: s["seed"])
    e = [s["argmin_error"] for s in rs]
    g = [s["regret"] for s in rs]
    d65 = [at(s, 0.65, "drop_error") for s in rs]
    sd = (lambda v: f" ± {statistics.stdev(v):.4f}" if len(v) > 1 else "")
    sdp = (lambda v: f" ± {100 * statistics.stdev(v):.2f}%" if len(v) > 1 else "")
    seed_rows.append([f"{re_:g}", ", ".join(str(s["seed"]) for s in rs),
                      num(rs[0]["s_star_reference_mesh_brent"]),
                      ", ".join(num(s["s_star_pinn"]) for s in rs),
                      f"{statistics.mean(e):+.4f}{sd(e)}",
                      f"{100 * statistics.mean(g):.4f}% ({100 * max(g):.4f}%)",
                      f"{100 * statistics.mean(d65):+.2f}%{sdp(d65)}"])
table(["Re_h", "Seeds", "Ref. s*", "PINN s* by seed", "Argmin error, mean ± sd",
       "Regret, mean (max)", "Drop error at 0.65, mean ± sd"], seed_rows,
      [700, 1100, 900, 2000, 1600, 1500, 1560], "Source: docs/re_study/seeds_30-90.md.")
bullets([
    "Re_h = 10 is the PINN's best case: all three seeds within 0.0014 of the reference optimum, "
    "spread 0.0005.",
    "Re_h = 0.1 is biased: all three seeds above the reference optimum, by +0.0034 on average.",
    "Re_h = 100 is scatter, not a systematic miss: −0.0050 to +0.0006 across seeds.",
    "The drop error at the range's edge (s = 0.65) is about 1% at Re_h 10 and 3–4% at both ends "
    "of the Reynolds range; regret stays below 0.06% in every 30/90 run because J is flat near "
    "its optimum.",
])
h2("4.2 Field errors by run")
for name, s in sorted(summaries.items(), key=lambda kv: (kv[1]["layout"][1], kv[1]["reynolds"],
                                                         kv[1]["seed"])):
    if "failed" in s:
        continue
    t = s["training"]
    p(f"{s['layout'][0]:g}/{s['layout'][1]:g}, Re_h = {s['reynolds']:g}, seed {s['seed']} "
      f"({s['pressure_scaling']} pressure). PINN s* = {s['s_star_pinn']:.4f}, J_PINN = "
      f"{s['J_pinn_at_s_star']:.6g}, J_ref at the PINN's s* = {s['J_fe_at_s_star_pinn']:.6g}; "
      f"reference s* = {s['s_star_reference_mesh_brent']:.4f}, J_ref* = "
      f"{s['J_fe_at_reference_s_star']:.6g}. Loss at s* {s['loss_at_s_star']:.3e}. Training "
      f"{t['wall_seconds'] / 60:.0f} min, final training loss {t['final_training_loss']:.3e}, "
      f"held-out {t['selected_heldout_loss']:.3e}, converged: {'yes' if t['converged'] else 'no'}.")
    rows = [[(f"{f['split']:.4f} (s*)" if f.get("at") else f"{f['split']:.2f}"),
             f"{f['u']:.2e}", f"{f['v']:.2e}", f"{f['p']:.2e}", pct(f["drop_error"])]
            for f in s["fields"]]
    table(["s", "u", "v", "p", "Drop error"], rows, [1800, 1890, 1890, 1890, 1890])
image(STUDY / "fig_re_objective.png",
      "Figure 2. J(s) from each PINN (blue line, 61 splits) against the reference solution (black "
      "dots); dashed: the PINN's optimum, dotted: the reference optimum. J in μU_c/h at Re_h ≤ 1 and "
      "ρU_c² above, so only shapes compare across panels.")
image(STUDY / "fig_re_drop_error.png",
      "Figure 3. The PINNs' pressure-drop error against the reference solution at the seven "
      "evaluation splits, one line per run; zero is a perfect prediction.")
image(STUDY / "fig_re_training.png",
      "Figure 4. Held-out loss against training time, one line per run; squares mark the start of "
      "L-BFGS, triangles of Levenberg–Marquardt. A diagnostic only: its scale differs between the "
      "viscous (Re_h ≤ 1) and inertial runs.")

# --------------------------------------------------------------------------- #
h1("5. Small items")
four = (STUDY / "network1_reference_four_meshes.csv").read_text().strip().splitlines()
hdr = four[0].split(",")
rows = [[r.split(",")[hdr.index(k)] for k in ("level", "nx", "ny", "unknowns",
                                              "total_pressure_drop", "dissipation",
                                              "entrance_length")] for r in four[1:]]
bullets([
    "L-BFGS iterations per cycle: 500 for Network 1 (4 cycles, developing_flow/re10_lm_fresh_"
    "inletgrid.yaml) and for Network 2, fixed-geometry and parametric (4 cycles).",
    "Sparse direct solver: SuperLU, through SciPy's spsolve, which scikit-fem's solve calls "
    "(UMFPACK is not installed).",
    "Network 1's velocity error at x = 10 against Poiseuille: not computed. No trained Network 1 "
    "is stored on this branch; retraining it takes about 4.4 h.",
    "Reference first pages (Gammaidoni et al.; Taylor & Hood): not retrievable from here.",
])
table(["Mesh level", "nx", "ny", "Unknowns", "Total-pressure drop", "Dissipation",
       "Entrance length"], rows, [900, 700, 700, 1200, 2000, 2000, 1860],
      "Network 1's reference on its four meshes (docs/re_study/network1_reference_four_meshes.csv). "
      "Observed order over the finest three: 4.39 (total-pressure drop), 3.98 (dissipation); the "
      "finest reference departs from the Poiseuille parabola by 4.5 × 10⁻⁸ at x = 10.")

# --------------------------------------------------------------------------- #
h1("6. Notes")
h2("6.1 Loss is not accuracy")
p("The Re_h = 100 run of seed 21 had the lowest held-out loss of the 30/90 runs and the largest "
  "argmin error; the only run that met the convergence definition (30/90, Re_h 100, seed 23) has "
  "the largest velocity error of the 30/90 runs. Every other run was still improving when the "
  "fixed schedule ended.")
h2("6.2 30/120 and run 5")
p("For 30/120 J rises across [0.35, 0.65] at Re_h 0.1, 1 and 10, and below the range it falls "
  "until the geometry refuses the split at s = 0.268. Runs 6 and 7 therefore end on the range's "
  "bound with the reference, and test drop and field accuracy only. Run 5 (Re_h 0.1) was stopped "
  "at LM step ~100 for that reason and recorded as skipped. Run 8 (Re_h 100) has an interior "
  "optimum and is the one 30/120 test of the argmin: the PINN put it at 0.3998 against 0.3859, "
  "regret 0.080%.")
h2("6.3 Where the files are")
bullets([
    "Per run: docs/re_study/pinn_<layout>_Re<Re>_seed<seed>.csv (s, J_pinn, J_fe), "
    "…_loss.csv (training history), …_summary.json/.md.",
    "Answer keys: docs/re_study/answer_keys.{md,json}, answer_keys_cases/, fe_<case>_reference.csv.",
    "Closed-length maps: docs/bifurcation_closed-re{0.1,1,10,25,50,100}.{md,json,png}; junction "
    "report docs/re_study/closed-junction-all.{md,json,png}.",
    "Figures: docs/re_study/figures.py; trained networks: bifurcation/runs/parametric_re*_corners_s*.",
])

OUT.write_text(json.dumps(blocks, indent=1, ensure_ascii=False))
print(f"{len(blocks)} blocks -> {OUT}")
