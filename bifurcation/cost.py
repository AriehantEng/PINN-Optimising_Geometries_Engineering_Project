"""What a design evaluation costs: finite elements against the parametric PINN.

    python -m bifurcation.cost --config bifurcation/cost_re10.yaml --out docs/bifurcation_re10-cost

Measured on this machine, one process at a time, nothing else running:

    finite elements   for each mesh in the config, on the parametric runs' layout
                      at the config's split: triangles, unknowns, solve time,
                      peak memory, and the total-pressure drop. Its error is
                      taken against the finest mesh listed, so the mesh whose
                      error matches the PINN's can be read off (the fair
                      baseline: equal accuracy, CLAUDE.md and the paper's 2.5).
                      The exponent of time against unknowns is fitted.
    the PINN          for each run: the time of one drop evaluation J(s) (the
                      design query: three 256-point line integrals), of the
                      field at every Gauss point of the reference mesh, and of
                      one Levenberg-Marquardt step (the normal equations on one
                      draw and their solve), each with its peak memory; and the
                      training time the run recorded.

Each measurement runs in its own subprocess, so peak memory (ru_maxrss) is
that measurement's alone. Times are medians over `repeats` after one warm-up
call (which pays for JIT compilation; reported separately).
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Meshes below this many unknowns are left out of the time-exponent fit.
FIT_FROM_UNKNOWNS = 20000

SCHEMA = {
    "name": str,
    "split": float,               # the geometry every measurement uses
    "meshes": list,               # [{max_area, corner_levels}], finest last
    "runs": list,                 # [{label, run}] finished parametric runs
    "repeats": int,
    "widths": list,               # LM-step cost at these widths (random networks)
}


def parse_config(document):
    from developing_flow import pinn
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    if document["repeats"] < 1 or len(document["meshes"]) < 2:
        raise pinn.ConfigError("repeats must be at least 1 and meshes list at least two")
    for m in document["meshes"]:
        if not isinstance(m, dict) or set(m) != {"max_area", "corner_levels"}:
            raise pinn.ConfigError("each mesh needs exactly max_area and corner_levels")
    return document


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _median_time(fn, repeats):
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return first, float(np.median(times))


# --------------------------------------------------------------------------- #
# The measurements, each run in a child process
# --------------------------------------------------------------------------- #

def measure_fe(run_dir, split, max_area, corner_levels):
    from bifurcation import parametric as P, reference
    from developing_flow import pinn
    c = P.load_config(ROOT / run_dir / "config.yaml")
    _, inlet, _, _ = P.build(c)
    start = time.perf_counter()
    sol = reference.solve_bifurcation(c.geometry(split), c.reynolds, c.pressure_scaling, inlet,
                                      c.outlet1_share, max_area, corner_levels,
                                      c.fe_corner_radius, continuation=())
    seconds = time.perf_counter() - start
    from bifurcation import flow
    a, _ = pinn.scaling_coefficients(c.reynolds, c.pressure_scaling)
    drop = flow.opening_measures(sol, c.geometry(split), a, c.n_line)["total_pressure_drop"]
    return {"max_area": max_area, "corner_levels": corner_levels, "seconds": seconds,
            "triangles": sol.info["triangles"], "unknowns": sol.info["unknowns"],
            "newton_iterations": len(sol.info["newton"]) if "newton" in sol.info else None,
            "drop": drop, "peak_mb": peak_mb()}


def measure_query(run_dir, split, repeats):
    import jax
    from bifurcation import parametric as P, reference
    model, c, params = P.load_run(ROOT / run_dir)
    first, query = _median_time(lambda: P.pinn_drop(c, model, split), repeats)
    # the field at every Gauss point of the reference's mesh
    _, inlet, _, _ = P.build(c)
    sol = reference.solve_bifurcation(c.geometry(split), c.reynolds, c.pressure_scaling, inlet,
                                      c.outlet1_share, c.fe_max_area, c.fe_corner_levels,
                                      c.fe_corner_radius, continuation=())
    x, y = sol.basis_u.global_coordinates().value
    x, y = x.ravel(), y.ravel()
    field_first, field = _median_time(lambda: model(x, y, np.full_like(x, split)), max(1, repeats // 4))
    return {"drop_query_seconds": query, "drop_query_first_seconds": first,
            "field_points": int(x.size), "field_seconds": field,
            "field_first_seconds": field_first, "parameters": int(sum(
                np.size(leaf) for leaf in jax.tree_util.tree_leaves(params))),
            "peak_mb": peak_mb()}


def measure_lm_step(run_dir, width, repeats):
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from bifurcation import flow, parametric as P
    from developing_flow import pinn
    import dataclasses
    c = P.load_config(ROOT / run_dir / "config.yaml")
    if width is not None:
        c = dataclasses.replace(c, width=width, init_from="random")
    st, inlet, q_in, field = P.build(c)
    groups, _ = P.make_residuals(c, st, field)
    params = P.init_params(c, jax.random.PRNGKey(0))
    flat, unflatten = ravel_pytree(params)
    normal_equations, _ = pinn.gauss_newton(groups, unflatten, flow.LM_CHUNK)
    batch = P.sample_batch(np.random.default_rng(0), c, inlet, c.designs)

    def step():
        A, g, _ = normal_equations(flat, batch)
        A = np.asarray(A)
        np.linalg.solve(A + 1e-4 * np.mean(np.diag(A)) * np.eye(len(A)), np.asarray(g))

    first, seconds = _median_time(step, repeats)
    rows = sum(len(r) for _, r in groups(batch))
    return {"width": c.width, "parameters": int(flat.size), "rows": int(rows),
            "seconds": seconds, "first_seconds": first, "peak_mb": peak_mb()}


def child(kind, payload):
    """Run one measurement in a fresh interpreter; return its JSON result."""
    code = (f"import json, sys; sys.path.insert(0, {str(ROOT)!r}); "
            f"from bifurcation import cost; "
            f"print('RESULT', json.dumps(cost.{kind}(**json.loads(sys.argv[1]))))")
    out = subprocess.run([sys.executable, "-c", code, json.dumps(payload)], check=True,
                         capture_output=True, text=True, cwd=ROOT)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def measure(document, verbose=True):
    first_run = document["runs"][0]["run"]
    fe = []
    for m in document["meshes"]:
        fe.append(child("measure_fe", {"run_dir": first_run, "split": document["split"], **m}))
        if verbose:
            print("  FE", fe[-1], flush=True)
    finest = fe[-1]["drop"]
    for r in fe:
        r["drop_error_vs_finest"] = r["drop"] / finest - 1.0
    # the exponent over the meshes large enough for the solve, not the setup, to dominate
    big = [r for r in fe if r["unknowns"] >= FIT_FROM_UNKNOWNS]
    slope = float(np.polyfit(np.log([r["unknowns"] for r in big]),
                             np.log([r["seconds"] for r in big]), 1)[0]) if len(big) > 1 else None
    pinn_rows = []
    for entry in document["runs"]:
        q = child("measure_query", {"run_dir": entry["run"], "split": document["split"],
                                    "repeats": document["repeats"]})
        lm = child("measure_lm_step", {"run_dir": entry["run"], "width": None,
                                       "repeats": max(1, document["repeats"] // 10)})
        metrics = json.loads((ROOT / entry["run"] / "metrics.json").read_text())
        pinn_rows.append({"label": entry["label"], "run": entry["run"], "query": q, "lm_step": lm,
                          "training_seconds": metrics["training"]["total_seconds"],
                          "lm_iterations": metrics["training"]["levenberg_marquardt"]["iterations"]})
        if verbose:
            print("  PINN", pinn_rows[-1], flush=True)
    widths = [child("measure_lm_step", {"run_dir": first_run, "width": w,
                                        "repeats": max(1, document["repeats"] // 10)})
              for w in document["widths"]]
    return {"config": document, "fe": fe, "fe_time_exponent": slope, "pinn": pinn_rows,
            "lm_step_by_width": widths}


def markdown(result):
    lines = ["## What a design evaluation costs: finite elements against the parametric PINN", "",
             f"Layout of the parametric runs at split {result['config']['split']:g}; one process, "
             "nothing else running; times are medians after a warm-up call.", "",
             "### Finite elements", "",
             "| max area | corner levels | triangles | unknowns | solve (s) | peak memory (MB) "
             "| drop | drop error vs finest |", "|---|---|---|---|---|---|---|---|"]
    for r in result["fe"]:
        lines.append(f"| {r['max_area']:g} | {r['corner_levels']} | {r['triangles']} | "
                     f"{r['unknowns']} | {r['seconds']:.2f} | {r['peak_mb']:.0f} | "
                     f"{r['drop']:.6f} | {100 * r['drop_error_vs_finest']:+.4f}% |")
    if result["fe_time_exponent"] is not None:
        lines += ["", f"Solve time grows like (unknowns)^{result['fe_time_exponent']:.2f} over "
                  f"the meshes with at least {FIT_FROM_UNKNOWNS} unknowns."]
    lines += ["", "### Parametric PINN", "",
              "| run | parameters | one drop query (s) | first query, with JIT (s) "
              "| field at reference Gauss points | one LM step (s) | LM step peak memory (MB) "
              "| training (min) |", "|---|---|---|---|---|---|---|---|"]
    for r in result["pinn"]:
        q, lm = r["query"], r["lm_step"]
        lines.append(f"| {r['label']} | {q['parameters']} | {q['drop_query_seconds']:.4f} | "
                     f"{q['drop_query_first_seconds']:.1f} | {q['field_points']} points in "
                     f"{q['field_seconds']:.2f} s | {lm['seconds']:.1f} | {lm['peak_mb']:.0f} | "
                     f"{r['training_seconds'] / 60:.0f} |")
    lines += ["", "### One Levenberg-Marquardt step against the network's width", "",
              "| width | parameters | rows | step (s) | peak memory (MB) |", "|---|---|---|---|---|"]
    for r in result["lm_step_by_width"]:
        lines.append(f"| {r['width']} | {r['parameters']} | {r['rows']} | {r['seconds']:.1f} | "
                     f"{r['peak_mb']:.0f} |")
    return "\n".join(lines) + "\n"


def run(config_path, out):
    document = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    result = measure(document)
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    out.with_suffix(".md").write_text(markdown(result))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
