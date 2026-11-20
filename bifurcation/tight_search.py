"""Both optima of P1 re-searched to a tight tolerance, and the argmin bound eps_s
rebuilt from measured parts. No training.

    python -m bifurcation.tight_search reference --out docs/re_study/tight_search \\
        --xatol 1e-5 --scan-step 0.0005 --scan-points 4 --mesh-delta 0.01 \\
        --check-mesh 0.005 5 --workers 4
    python -m bifurcation.tight_search pinn --out docs/re_study/tight_search --xatol 1e-5
    python -m bifurcation.tight_search report --out docs/re_study/tight_search

Both searches repeat parametric.evaluate's procedure with only the tolerance
changed: the PINN's objective on its scan (evaluation.curve_points), then
bounded Brent between the best point's neighbours; the reference's at the
evaluation splits (the stored values pick the bracket), then bounded Brent on
the run's own reference mesh (evaluation.max_area, corner_levels,
corner_radius), measured by flow.opening_measures. That is P1's preregistered
reference, "Brent's optimum on the reference mesh from the evaluation splits".

The reference mesh is generated afresh for every split, so J on it carries a
small jitter from one split to the next. Below that jitter no tolerance can
place the minimum, so for every reference case three more things are measured:

    scan    J at s_B + k * scan_step, k = +-1 .. +-scan_points, around the
            Brent optimum s_B; a least-squares parabola through every
            reference-mesh value within that window gives the curvature J''/J,
            the jitter (rms residual sigma), the vertex, and the vertex's
            leave-one-out spread
    noise   sqrt(4 sigma / J''): two values whose jitter differs by 2 sigma
            cannot be ranked closer to the minimum than this, so Brent's
            answer is fixed only to within it
    mesh    the reference mesh's own discretisation shift, estimated as the
            answer keys estimate the search mesh's (sensitivity.mesh_bound):
            e = J_ref / J_check - 1 at s_B and s_B +- mesh_delta, and
            delta_s = -e'/(J''/J), the check mesh being the next coarser one

    eps_s (Brent)  = xatol + noise + |delta_s|
    eps_s (vertex) = leave-one-out spread + |delta_s|

Every finite-element solve is appended to solves.jsonl as it finishes and is
read back on a restart, so an interrupted search resumes where it stopped (the
solves are deterministic: a stored value reproduces bit for bit).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import pathlib
import re
import threading
import time

import numpy as np

RUNS = pathlib.Path(__file__).resolve().parent / "runs"
P1_TOLERANCE = 0.002

#: the paper's tables, by run name; anything else is an earlier experiment
FAMILIES = (
    ("controls", re.compile(r"_cut0\.(15|4)_corners_s21$|_q0\.2_corners_s22$")),
    ("amendment 3", re.compile(r"_width_corners_|_lm400_")),
    ("Tier A", re.compile(r"_q0\.(2|35|65|8)_corners_s21$|_0120_|_1515_")),
    ("q = 0.5 (earlier)", re.compile(r"_3090_corners_s2[123]$|_3090_equal_corners$|_30120_")),
)


def family(name):
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "earlier experiment"


# --------------------------------------------------------------------------- #
# Cases: one reference problem per (physics, geometry, inlet, mesh)
# --------------------------------------------------------------------------- #


def case_of(config):
    """Everything the reference J(s) depends on, as a key and a description."""
    spec = {"angles": [config.angle1, config.angle2], "share": config.outlet1_share,
            "reynolds": config.reynolds, "pressure_scaling": config.pressure_scaling,
            "continuation": list(config.fe_continuation),
            "geometry": [config.parent_length, config.daughter_length, config.min_width],
            "inlet": config.inlet_profile, "mesh": [config.fe_max_area, config.fe_corner_levels,
                                                    config.fe_corner_radius],
            "n_line": config.n_line}
    key = (f"{config.angle1:g}-{config.angle2:g}_q{config.outlet1_share:g}"
           f"_Re{config.reynolds:g}")
    return key, spec


def finished_runs():
    return [d for d in sorted(RUNS.glob("parametric_*")) if (d / "metrics.json").is_file()]


def bracket_of(metrics):
    """parametric.evaluate's bracket: the best evaluation split and its
    neighbours, from the stored values (which reproduce bit for bit)."""
    rows = sorted((r["split"], r["reference_drop"]) for r in metrics["splits"])
    k = min(range(len(rows)), key=lambda i: rows[i][1])
    if k in (0, len(rows) - 1):
        return None
    return rows[k - 1][0], rows[k + 1][0]


# --------------------------------------------------------------------------- #
# Finite-element solves (worker processes) and their cache
# --------------------------------------------------------------------------- #


def fe_drop(task):
    """J at split s for one run's reference problem, on (max_area, levels)."""
    run, s, max_area, levels = task
    import jax
    jax.config.update("jax_enable_x64", True)
    from bifurcation import flow, parametric as P, reference
    from developing_flow import pinn
    c = P.load_config(pathlib.Path(run) / "config.yaml")
    start = time.perf_counter()
    g = c.geometry(s)
    solution = reference.solve_bifurcation(
        g, c.reynolds, c.pressure_scaling, flow.inlet_function(c), c.outlet1_share,
        max_area, levels, c.fe_corner_radius, continuation=c.fe_continuation)
    a, _ = pinn.scaling_coefficients(c.reynolds, c.pressure_scaling)
    j = flow.opening_measures(solution, g, a, c.n_line)["total_pressure_drop"]
    return {"J": j, "seconds": time.perf_counter() - start,
            "triangles": int(solution.info["triangles"])}


class Solver:
    """Solves on a process pool, each remembered in solves.jsonl."""

    def __init__(self, out, workers):
        self.path = out / "solves.jsonl"
        self.lock = threading.Lock()
        self.cache = {}
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.cache[self.key(r["case"], r["mesh"], r["split"])] = r
        self.pool = concurrent.futures.ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn"))

    @staticmethod
    def key(case, mesh, s):
        return f"{case}|{mesh[0]!r}/{mesh[1]}|{float(s)!r}"

    def submit(self, case, run, mesh, s):
        """A future for J; already-known values resolve at once."""
        k = self.key(case, mesh, s)
        with self.lock:
            if k in self.cache:
                done = concurrent.futures.Future()
                done.set_result(self.cache[k]["J"])
                return done
        inner = self.pool.submit(fe_drop, (str(run), float(s), mesh[0], mesh[1]))
        outer = concurrent.futures.Future()

        def finished(f):
            try:
                r = f.result()
            except BaseException as error:          # surfaced where .result() is called
                outer.set_exception(error)
                return
            row = {"case": case, "mesh": list(mesh), "split": float(s), **r}
            with self.lock:
                self.cache[k] = row
                with self.path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            outer.set_result(r["J"])

        inner.add_done_callback(finished)
        return outer

    def values(self, case, mesh):
        with self.lock:
            return sorted((r["split"], r["J"]) for r in self.cache.values()
                          if r["case"] == case and r["mesh"] == list(mesh))


# --------------------------------------------------------------------------- #
# Measurements on one reference case
# --------------------------------------------------------------------------- #


def brent(fun, bounds, xatol):
    """parametric.evaluate's search with its tolerance replaced."""
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(fun, bounds=bounds, method="bounded", options={"xatol": xatol})
    return float(result.x), float(result.fun), int(result.nfev)


def parabola(points, centre):
    """Least-squares parabola through (s, J): vertex, J''/J, rms jitter
    relative to J, the noise half-width sqrt(4 sigma / J''), and the vertex's
    leave-one-out spread."""
    s = np.array([p[0] for p in points]) - centre
    j = np.array([p[1] for p in points])
    c2, c1, c0 = np.polyfit(s, j, 2)
    sigma = float(np.sqrt(np.mean((j - np.polyval([c2, c1, c0], s)) ** 2)))
    vertices = []
    for k in range(len(s)):
        keep = np.arange(len(s)) != k
        a2, a1, _ = np.polyfit(s[keep], j[keep], 2)
        vertices.append(centre - a1 / (2.0 * a2))
    return {"vertex": float(centre - c1 / (2.0 * c2)), "curvature": float(2.0 * c2 / c0),
            "jitter": sigma / float(c0), "noise_half_width": float(math.sqrt(2.0 * sigma / c2))
            if c2 > 0 else math.inf,
            "vertex_spread": float(max(vertices) - min(vertices)), "points": len(s)}


def mesh_shift(j_ref, j_check, xs, curvature):
    """sensitivity.mesh_bound's estimate, check mesh against reference mesh."""
    e = np.array([r / c - 1.0 for r, c in zip(j_ref, j_check)])
    slope = float(np.polyfit(np.array(xs), e, 1)[0])
    return {"error_slope": slope, "shift": -slope / curvature,
            "relative_difference": e.tolist()}


def reference_case(solver, key, spec, brackets, args):
    """Every measurement on one reference case; brackets maps a bracket
    (lo, hi) to the runs that share it, the first being the primary."""
    fine = (spec["mesh"][0], spec["mesh"][1])
    check = (args.check_mesh[0], int(args.check_mesh[1]))
    out = {"case": key, "spec": spec, "brent": []}
    for (lo, hi), runs in brackets.items():
        run = runs[0]
        s_b, j_b, nfev = brent(lambda s: solver.submit(key, run, fine, s).result(),
                               (lo, hi), args.xatol)
        out["brent"].append({"bracket": [lo, hi], "runs": [r.name for r in runs],
                             "s": s_b, "J": j_b, "evaluations": nfev})
    centre, run = out["brent"][0]["s"], next(iter(brackets.values()))[0]
    scan = [centre + k * args.scan_step for k in range(-args.scan_points, args.scan_points + 1)
            if k]
    mesh_points = [centre - args.mesh_delta, centre, centre + args.mesh_delta]
    futures = [solver.submit(key, run, fine, s) for s in scan + mesh_points]
    futures += [solver.submit(key, run, check, s) for s in mesh_points]
    for f in futures:
        f.result()
    window = args.scan_points * args.scan_step + 1e-12
    near = [p for p in solver.values(key, fine) if abs(p[0] - centre) <= window]
    fit = parabola(near, centre)
    shift = mesh_shift([solver.submit(key, run, fine, s).result() for s in mesh_points],
                       [solver.submit(key, run, check, s).result() for s in mesh_points],
                       mesh_points, fit["curvature"])
    out.update({"scan": fit, "mesh": {"check_mesh": list(check), "delta": args.mesh_delta,
                                      **shift},
                "eps_s": {"xatol": args.xatol, "noise": fit["noise_half_width"],
                          "mesh": abs(shift["shift"]),
                          "brent": args.xatol + fit["noise_half_width"] + abs(shift["shift"]),
                          "vertex": fit["vertex_spread"] + abs(shift["shift"])}})
    return out


def run_reference(args):
    from bifurcation import parametric as P
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases, skipped = {}, []
    for d in finished_runs():
        config = P.load_config(d / "config.yaml")
        metrics = json.loads((d / "metrics.json").read_text())
        key, spec = case_of(config)
        b = bracket_of(metrics)
        if b is None:
            skipped.append({"run": d.name, "why": "reference optimum on an evaluation-grid end"})
            continue
        entry = cases.setdefault(key, {"spec": spec, "brackets": {}})
        if entry["spec"] != spec:
            raise ValueError(f"{d.name}: same case key {key} with a different reference problem")
        entry["brackets"].setdefault(b, []).append(d)
    solver = Solver(out, args.workers)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(len(cases)) as threads:
        jobs = {threads.submit(reference_case, solver, k, v["spec"], v["brackets"], args): k
                for k, v in cases.items()}
        for job in concurrent.futures.as_completed(jobs):
            r = job.result()
            results[r["case"]] = r
            print(f"{r['case']}: Brent {r['brent'][0]['s']:.6f} ({r['brent'][0]['evaluations']} "
                  f"solves), vertex {r['scan']['vertex']:.6f}, jitter {r['scan']['jitter']:.1e}, "
                  f"eps_s {r['eps_s']['brent']:.5f} (Brent) {r['eps_s']['vertex']:.5f} (vertex)",
                  flush=True)
    solver.pool.shutdown()
    settings = {k: getattr(args, k) for k in ("xatol", "scan_step", "scan_points", "mesh_delta",
                                              "check_mesh")}
    (out / "reference.json").write_text(json.dumps(
        {"settings": settings, "skipped": skipped,
         "cases": dict(sorted(results.items()))}, indent=1))


# --------------------------------------------------------------------------- #
# The PINN's search
# --------------------------------------------------------------------------- #


def run_pinn(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    from bifurcation import parametric as P
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pinn.json"
    done = json.loads(path.read_text())["runs"] if path.is_file() else {}
    for d in finished_runs():
        if d.name in done:
            continue
        model, config, _ = P.load_run(d)
        o = json.loads((d / "metrics.json").read_text())["optimum"]
        grid = np.linspace(config.split_min, config.split_max, config.curve_points)
        values = np.array([P.pinn_drop(config, model, s) for s in grid])
        k = int(np.argmin(values))
        row = {"family": family(d.name), "case": case_of(config)[0], "seed": config.seed,
               "stored": {"s": o["pinn_split"], "J": o["pinn_min_drop"],
                          "reference_s": o["reference_split"]},
               "curve_reproduces": bool(np.array_equal(values, np.array(o["curve"]["pinn_drop"])))}
        if k in (0, len(grid) - 1):
            row.update({"s": float(grid[k]), "J": float(values[k]), "on_bound": True,
                        "evaluations": 0})
        else:
            s, j, nfev = brent(lambda x: P.pinn_drop(config, model, x),
                               (grid[k - 1], grid[k + 1]), args.xatol)
            row.update({"s": s, "J": j, "on_bound": False, "evaluations": nfev})
        done[d.name] = row
        print(f"{d.name}: {row['stored']['s']:.6f} -> {row['s']:.6f}"
              f"{' (bound)' if row['on_bound'] else ''}", flush=True)
        path.write_text(json.dumps({"xatol": args.xatol, "runs": done}, indent=1))


# --------------------------------------------------------------------------- #
# The recount
# --------------------------------------------------------------------------- #


def verdict(error, eps):
    """Clear pass, clear miss, or within eps_s of P1's boundary."""
    if abs(abs(error) - P1_TOLERANCE) <= eps:
        return "borderline"
    return "pass" if abs(error) < P1_TOLERANCE else "miss"


def run_report(args):
    out = pathlib.Path(args.out)
    ref = json.loads((out / "reference.json").read_text())
    pinn_runs = json.loads((out / "pinn.json").read_text())["runs"]
    rows = []
    for name, p in pinn_runs.items():
        case = ref["cases"].get(p["case"])
        row = {"run": name, "family": p["family"], "case": p["case"], "seed": p["seed"],
               "s_pinn": p["s"], "pinn_on_bound": p["on_bound"],
               "stored_error": p["stored"]["s"] - p["stored"]["reference_s"]}
        if case is None:
            row["note"] = "no interior reference optimum"
            rows.append(row)
            continue
        brent_s = next((b["s"] for b in case["brent"] if name in b["runs"]), case["brent"][0]["s"])
        row.update({
            "s_ref_brent": brent_s, "s_ref_vertex": case["scan"]["vertex"],
            "error_brent": p["s"] - brent_s, "error_vertex": p["s"] - case["scan"]["vertex"],
            "eps_brent": case["eps_s"]["brent"], "eps_vertex": case["eps_s"]["vertex"]})
        row["verdict_stored"] = verdict(row["stored_error"], 0.0)
        row["verdict_brent"] = verdict(row["error_brent"], row["eps_brent"])
        row["verdict_vertex"] = verdict(row["error_vertex"], row["eps_vertex"])
        rows.append(row)
    (out / "recount.json").write_text(json.dumps(rows, indent=1))

    lines = ["## P1 re-searched to a tight tolerance", "",
             f"Settings: {json.dumps(ref['settings'])}; PINN xatol "
             f"{json.loads((out / 'pinn.json').read_text())['xatol']:g}. P1: |s*_PINN - s*_ref| "
             f"<= {P1_TOLERANCE}; 'borderline' = within eps_s of that boundary.", "",
             "### Reference cases", "",
             "| case | Brent s* (solves) | vertex | J''/J | jitter | noise | mesh shift | "
             "eps_s Brent | eps_s vertex |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in ref["cases"].items():
        b = c["brent"]
        lines.append(
            f"| {k} | " + "; ".join(f"{x['s']:.5f} ({x['evaluations']})" for x in b)
            + f" | {c['scan']['vertex']:.5f} | {c['scan']['curvature']:.1f} | "
            f"{c['scan']['jitter']:.1e} | {c['scan']['noise_half_width']:.5f} | "
            f"{c['mesh']['shift']:+.5f} | {c['eps_s']['brent']:.5f} | {c['eps_s']['vertex']:.5f} |")
    lines += ["", "### Runs", "",
              "| run | family | stored δs* | δs* (Brent) | verdict | δs* (vertex) | verdict |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["family"], r["run"])):
        if "note" in r:
            lines.append(f"| {r['run']} | {r['family']} | {r['stored_error']:+.5f} | "
                         f"{r['note']} | | | |")
            continue
        lines.append(f"| {r['run']} | {r['family']} | {r['stored_error']:+.5f} | "
                     f"{r['error_brent']:+.5f} | {r['verdict_brent']} | "
                     f"{r['error_vertex']:+.5f} | {r['verdict_vertex']} |")
    (out / "recount.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="part", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--out", required=True)
    r.add_argument("--xatol", type=float, required=True)
    r.add_argument("--scan-step", type=float, required=True)
    r.add_argument("--scan-points", type=int, required=True)
    r.add_argument("--mesh-delta", type=float, required=True)
    r.add_argument("--check-mesh", type=float, nargs=2, required=True,
                   metavar=("MAX_AREA", "CORNER_LEVELS"))
    r.add_argument("--workers", type=int, required=True)
    p = sub.add_parser("pinn")
    p.add_argument("--out", required=True)
    p.add_argument("--xatol", type=float, required=True)
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"reference": run_reference, "pinn": run_pinn, "report": run_report}[args.part](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
