"""Every graph, table and result since the figure-relabelling request, in order.

    python docs/re_study/report/build_since_relabel.py
        -> docs/re_study/report/content_since_relabel.json
    node docs/re_study/report/build_docx.js docs/re_study/report/content_since_relabel.json \\
        docs/re_study/re_study_since_relabel.docx

From the request to relabel the study figures ("reference solution", "reference
optimum", and the pressure-drop error axis) to the end of the queue: the
relabelled figures, runs 8, 2 and 6, the six seed runs, and the comparisons
made as they finished. Numbers are read from the run summaries.
"""
import json
import pathlib
import statistics
import struct

ROOT = pathlib.Path(__file__).resolve().parents[3]
STUDY = ROOT / "docs" / "re_study"
OUT = STUDY / "report" / "content_since_relabel.json"
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


S = {}
for f in STUDY.glob("pinn_*_summary.json"):
    s = json.loads(f.read_text())
    if "failed" not in s:
        S[(f"{s['layout'][0]:g}/{s['layout'][1]:g}", s["reynolds"], s["seed"])] = s


def run(lay, re_, seed):
    return S[(lay, float(re_), seed)]


def by(s, x):
    return next(f for f in s["fields"] if abs(f["split"] - x) < 1e-9 and not f.get("at"))


def drops(s):
    return " / ".join(f"{100 * by(s, x)['drop_error']:+.2f}%" for x in (0.35, 0.5, 0.65))


def run_section(title, s, notes):
    h2(title)
    t = s["training"]
    bullets([
        f"PINN optimum s* = {s['s_star_pinn']:.4f} (J_PINN = {s['J_pinn_at_s_star']:.6g}); "
        f"reference optimum {s['s_star_reference_mesh_brent']:.4f} (J_ref* = "
        f"{s['J_fe_at_reference_s_star']:.6g}); reference J at the PINN's s* = "
        f"{s['J_fe_at_s_star_pinn']:.6g}.",
        f"Argmin error {s['argmin_error']:+.4f}; regret {100 * s['regret']:.4f}%.",
        f"Pressure-drop error at s = 0.35 / 0.50 / 0.65: {drops(s)}.",
        f"Training {t['wall_seconds'] / 60:.0f} min; final training loss "
        f"{t['final_training_loss']:.3e}; held-out loss {t['selected_heldout_loss']:.3e} "
        f"({t['selected']}); converged: {'yes' if t['converged'] else 'no'}.",
    ] + notes)
    rows = [[(f"{f['split']:.4f} (PINN s*)" if f.get("at") else f"{f['split']:.2f}"),
             f"{f['u']:.2e}", f"{f['v']:.2e}", f"{f['p']:.2e}", f"{100 * f['drop_error']:+.3f}%"]
            for f in s["fields"]]
    table(["s", "u error", "v error", "p error", "Pressure-drop error"], rows,
          [1900, 1800, 1800, 1800, 2100],
          "Relative L2 errors against the reference solution, integrated on the reference mesh.")


def seed_table(re_, seeds, ref):
    rows = []
    for seed in seeds:
        s = run("30/90", re_, seed)
        rows.append([str(seed), f"{s['s_star_pinn']:.4f}", f"{s['argmin_error']:+.4f}",
                     f"{100 * s['regret']:.4f}%", drops(s), f"{by(s, 0.5)['u']:.2e}"])
    table(["Seed", "PINN s*", f"Argmin error (reference {ref:.4f})", "Regret",
           "Pressure-drop error at 0.35 / 0.50 / 0.65", "u error at 0.50"], rows,
          [700, 1100, 1900, 1100, 2900, 1700])


# --------------------------------------------------------------------------- #
blocks.append({"title": "Reynolds-number study: results since the figure relabelling"})
p("Everything produced from the request to relabel the study's figures to the end of the run "
  "queue, in the order it happened: the relabelled figures, runs 8, 2 and 6, the six seed runs, "
  "and the comparisons made as each finished. J is in μU_c/h at Re_h ≤ 1 and ρU_c² above. "
  "Argmin error = PINN s* − reference s*; regret = J_ref(s*_PINN)/J_ref* − 1; pressure-drop error "
  "= J_PINN/J_ref − 1 at the same split. 'Converged' = the schedule finished, every loss finite, "
  "and the best held-out loss of the last 25 LM steps not more than 10% below the best before "
  "them.")

h1("1. The relabelled figures")
bullets([
    "Four-panel objective plot (docs/re_study/figures.py): legend 'finite elements (reference "
    "mesh)' → 'reference solution'; 'FE optimum' → 'reference optimum'.",
    "Pressure-drop error plot: y-axis 'pressure-drop error of the PINN against the reference "
    "solution (%)'.",
    "Every redraw since uses the new labels; the final versions, with all 13 runs, are in Section 7.",
])

h1("2. Run 8: 30/120, Re_h = 100, seed 21")
run_section("Result", run("30/120", 100, 21), [
    "The only 30/120 case with an interior optimum in the range, and the largest argmin error and "
    "regret of any run.",
    "u error 0.8–1.9% and v error 1.2–3.1%: the largest field errors of any run.",
    "The reference optimum depends on the mesh here: Brent on the reference mesh gives 0.3859, "
    "the answer key's search mesh 0.3797 ± 0.0013. J is very flat near it (0.2% between "
    "s = 0.35 and 0.40), so 0.3859 is the value to quote.",
])

h1("3. Run 2: 30/90, Re_h = 1, seed 21")
run_section("Result", run("30/90", 1, 21), [
    "Inside the ±0.002 search tolerance; the smallest regret of any run with an interior optimum "
    "at the time.",
])
h2("30/90, seed 21, across Reynolds numbers")
rows = []
for re_ in (0.1, 1, 10, 100):
    s = run("30/90", re_, 21)
    rows.append([f"{re_:g}", f"{s['s_star_pinn']:.4f}", f"{s['s_star_reference_mesh_brent']:.4f}",
                 f"{s['argmin_error']:+.4f}", f"{100 * s['regret']:.4f}%"])
table(["Re_h", "PINN s*", "Reference s*", "Argmin error", "Regret"], rows,
      [1400, 1900, 1900, 1900, 2260],
      "Up to Re_h 10 the argmin errors scatter on both sides, within about ±0.003; at Re_h 100 "
      "seed 21 missed by 0.005 (Section 4 shows this was mostly seed scatter).")

h1("4. Seed runs: seed 22")
run_section("30/90, Re_h = 0.1, seed 22", run("30/90", 0.1, 22), [])
p("Seed 21 against seed 22 at Re_h = 0.1:")
seed_table(0.1, (21, 22), run("30/90", 0.1, 21)["s_star_reference_mesh_brent"])
p("The argmin error moves by about 0.003 between seeds, as large as the Re_h = 100 miss of "
  "seed 21: too early to attribute that miss to inertia.")
run_section("30/90, Re_h = 100, seed 22", run("30/90", 100, 22), [])
seed_table(100, (21, 22), run("30/90", 100, 21)["s_star_reference_mesh_brent"])
p("With seed 22 the PINN lands within the ±0.002 tolerance at Re_h = 100: run 4's 0.005 miss was "
  "mostly seed scatter. Better optimum placement came with a larger drop error at s = 0.5 "
  "(−1.06% against −0.07%): drop accuracy and optimum location do not track each other.")
run_section("30/90, Re_h = 10, seed 22", run("30/90", 10, 22), [
    "The most accurate run of the study: argmin error +0.0004, edge error −0.76%."])
rows = []
for re_ in (0.1, 10, 100):
    rows.append([f"{re_:g}"] + [f"{run('30/90', re_, sd)['argmin_error']:+.4f}" for sd in (21, 22)])
table(["Re_h", "Seed 21", "Seed 22"], rows, [3000, 3180, 3180],
      "Argmin error with two seeds per Reynolds number. Re_h 10 is tightest; both Re_h 0.1 seeds "
      "land above the optimum and both Re_h 100 seeds below it, which the third seed tests.")

h1("5. Seed runs: seed 23")
run_section("30/90, Re_h = 0.1, seed 23", run("30/90", 0.1, 23), [])
seed_table(0.1, (21, 22, 23), run("30/90", 0.1, 21)["s_star_reference_mesh_brent"])
p("All three seeds place the optimum above the reference value: at Re_h = 0.1 the PINN has a "
  "systematic bias, not just scatter.")
run_section("30/90, Re_h = 100, seed 23", run("30/90", 100, 23), [
    "The first run that meets the convergence definition, yet it has the largest velocity error of "
    "the 30/90 runs (9.4 × 10⁻³) and a +1.9% drop error at s = 0.35."])
seed_table(100, (21, 22, 23), run("30/90", 100, 21)["s_star_reference_mesh_brent"])
p("No consistent bias at Re_h = 100: the three argmin errors span −0.005 to +0.0006, mostly seed "
  "scatter. The reading that the PINN misses the inertial shift of the optimum does not hold up.")
run_section("30/90, Re_h = 10, seed 23", run("30/90", 10, 23), [])

h1("6. Run 6: 30/120, Re_h = 1, seed 21")
run_section("Result", run("30/120", 1, 21), [
    "Optimum on the range's bound at s = 0.35, as the reference: J rises across the whole range, "
    "so the run tests pressure-drop and field accuracy only.",
    "The −7.6% drop error at s = 0.65, with a 12% pressure error there, is the largest edge error of "
    "the study.",
])

h1("7. Final results")
rows = []
for (lay, re_, seed), s in sorted(S.items(), key=lambda kv: (kv[0][0] == "30/120", kv[0][1], kv[0][2])):
    t = s["training"]
    rows.append([lay, f"{re_:g}", str(seed), f"{s['s_star_pinn']:.4f}",
                 f"{s['s_star_reference_mesh_brent']:.4f}", f"{s['argmin_error']:+.4f}",
                 f"{100 * s['regret']:.4f}%", drops(s), f"{by(s, 0.5)['u']:.2e}",
                 f"{t['wall_seconds'] / 60:.0f}", "yes" if t["converged"] else "no"])
rows.append(["30/120", "0.1", "21", "skipped: optimum on the range's bound", "", "", "", "", "", "", ""])
table(["Layout", "Re_h", "Seed", "PINN s*", "Ref. s*", "Argmin error", "Regret",
       "Pressure-drop error at 0.35 / 0.50 / 0.65", "u error at 0.50", "Min", "Converged"], rows,
      [750, 550, 500, 800, 800, 850, 850, 1950, 950, 500, 860],
      "All 13 runs (docs/re_study/pinn_runs.md).")
rows = []
for re_ in (0.1, 1, 10, 100):
    rs = [s for (lay, r, sd), s in S.items() if lay == "30/90" and r == re_]
    rs.sort(key=lambda s: s["seed"])
    e = [s["argmin_error"] for s in rs]
    g = [s["regret"] for s in rs]
    d65 = [by(s, 0.65)["drop_error"] for s in rs]
    sd = (lambda v: f" ± {statistics.stdev(v):.4f}" if len(v) > 1 else "")
    sdp = (lambda v: f" ± {100 * statistics.stdev(v):.2f}%" if len(v) > 1 else "")
    rows.append([f"{re_:g}", ", ".join(str(s["seed"]) for s in rs),
                 f"{rs[0]['s_star_reference_mesh_brent']:.4f}",
                 ", ".join(f"{s['s_star_pinn']:.4f}" for s in rs),
                 f"{statistics.mean(e):+.4f}{sd(e)}",
                 f"{100 * statistics.mean(g):.4f}% ({100 * max(g):.4f}%)",
                 f"{100 * statistics.mean(d65):+.2f}%{sdp(d65)}"])
table(["Re_h", "Seeds", "Reference s*", "PINN s* by seed", "Argmin error, mean ± sd",
       "Regret, mean (max)", "Drop error at 0.65, mean ± sd"], rows,
      [650, 1100, 1100, 2000, 1600, 1500, 1410],
      "30/90 over seeds 21–23; Re_h 1 has seed 21 only (docs/re_study/seeds_30-90.md).")
bullets([
    "H3, 30/90: the PINN locates the optimum within the ±0.002 search tolerance on average at "
    "Re_h 10 (+0.0009 ± 0.0005); it is biased at Re_h 0.1 (+0.0034 ± 0.0021) and scattered at "
    "Re_h 100 (−0.0019 ± 0.0029). Regret stays below 0.06% in every 30/90 run.",
    "30/120: only at Re_h 100 is there an interior optimum in the range; the PINN misses it by "
    "0.014 (regret 0.08%). At Re_h 0.1, 1 and 10 the optimum sits on the range's bound (run 5 was "
    "stopped for that reason).",
    "Accuracy against Reynolds number: the drop error at the range's edge is about 1% at Re_h 10 and "
    "grows toward both ends, to 2.6–4.6% at Re_h 0.1 and 2.9–4.0% at Re_h 100 for 30/90, and "
    "7.6% for 30/120 at Re_h 1.",
    "Convergence: only one run (30/90, Re_h 100, seed 23) met the definition within the fixed "
    "schedule; all others were still improving at the end.",
    "Code check after the last run: the fast test suite passed (805 passed, 5 skipped, 2 expected "
    "failures).",
])
image(STUDY / "fig_re_objective.png",
      "Figure 1. J(s) from each PINN (blue line, 61 splits) against the reference solution (black "
      "dots); dashed: the PINN's optimum, dotted: the reference optimum. J in μU_c/h at Re_h ≤ 1 and "
      "ρU_c² above, so only shapes compare across panels.")
image(STUDY / "fig_re_drop_error.png",
      "Figure 2. Pressure-drop error of each PINN against the reference solution at the seven "
      "evaluation splits; zero is a perfect prediction.")
image(STUDY / "fig_re_training.png",
      "Figure 3. Held-out loss against training time for every run; squares mark the start of "
      "L-BFGS, triangles of Levenberg–Marquardt. A diagnostic only: its scale differs between the "
      "viscous (Re_h ≤ 1) and inertial runs.")

OUT.write_text(json.dumps(blocks, indent=1, ensure_ascii=False))
print(f"{len(blocks)} blocks -> {OUT}")
