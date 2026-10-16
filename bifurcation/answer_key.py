"""Finite-element answer keys for the parametric PINNs at several Reynolds numbers.

    python -m bifurcation.answer_key --config bifurcation/answer_keys_re.yaml \\
        --out docs/re_study/answer_keys

For each layout (daughter angles) and each Reynolds number in the config, on
the parametric runs' geometry (outer walls daughter_length long, one share):

    optimum       the design study's procedure (design_study.optimise): a scan
                  over every split the width bounds allow, Brent to search.xatol
                  on the search mesh, then the optimum and its two grid
                  neighbours re-solved on the check mesh. The argmin bound is
                  sensitivity.mesh_bound's: |delta_s| + xatol, with delta_s the
                  shift the check mesh's error slope implies.
    reference     J on the PINNs' reference mesh at reference.splits and at the
                  optimum, measured by flow.opening_measures, the function that
                  measures every PINN (CLAUDE.md: one measuring function).
    lengths       each daughter's centreline (and closed) length at the optimum,
                  the Murray split s_M and the length-corrected s_L.

J is in the units of the case's pressure scaling: rho U_c^2 (inertial) or
mu U_c / h (viscous); s* and every relative quantity are independent of it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import pathlib
import time

import numpy as np
import yaml

from bifurcation import design_study, flow, reference, sensitivity
from developing_flow import pinn

ConfigError = pinn.ConfigError
MESH = {"max_area": float, "corner_levels": int}
SCHEMA = {
    "name": str,
    "geometry": {"parent_length": float, "daughter_length": float, "min_width": float},
    "share": float,
    "layouts": list,              # [[angle1, angle2], ...]
    "flows": list,                # [{reynolds, pressure_scaling, continuation}]
    "search": {"grid": int, "xatol": float},
    "mesh": {"max_area": float, "corner_levels": int, "corner_radius": float},
    "check": MESH,
    "reference": {"max_area": float, "corner_levels": int, "n_line": int, "splits": list},
    "workers": int,
}
FLOW = {"reynolds": float, "pressure_scaling": str, "continuation": list}


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    if not 0.0 < document["share"] < 1.0:
        raise ConfigError("share must lie in (0, 1)")
    for pair in document["layouts"]:
        if not (isinstance(pair, list) and len(pair) == 2):
            raise ConfigError(f"layouts must be pairs [angle1, angle2], got {pair!r}")
    if not document["flows"]:
        raise ConfigError("flows must list at least one Reynolds number")
    for k, case in enumerate(document["flows"]):
        pinn._check(case, FLOW, f"flows[{k}]")
        if case["pressure_scaling"] not in ("inertial", "viscous"):
            raise ConfigError(f"flows[{k}].pressure_scaling must be inertial or viscous")
        steps = list(case["continuation"]) + [case["reynolds"]]
        if any(not 0.0 < lo < hi for lo, hi in zip(steps, steps[1:])) or steps[-1] <= 0.0:
            raise ConfigError(f"flows[{k}].continuation must increase to below its reynolds")
    if document["search"]["grid"] < 3 or document["workers"] < 1:
        raise ConfigError("search.grid must be at least 3 and workers at least 1")
    return document


def study_config(config, case):
    """The design-study config one (layout, Reynolds number) case is optimised with."""
    return {"physics": {"reynolds": float(case["reynolds"]),
                        "pressure_scaling": case["pressure_scaling"],
                        "continuation": [float(r) for r in case["continuation"]]},
            "geometry": dict(config["geometry"]), "search": dict(config["search"]),
            "mesh": dict(config["mesh"]), "check": dict(config["check"])}


def key(angles, reynolds):
    return f"{angles[0]:g}-{angles[1]:g}_Re{reynolds:g}"


def answer(task):
    config, angles, case = task
    start = time.perf_counter()
    study = study_config(config, case)
    row = design_study.optimise((study, angles, config["share"]))
    out = {"layout": angles, "reynolds": case["reynolds"],
           "pressure_scaling": case["pressure_scaling"], "continuation": case["continuation"],
           "share": config["share"], "key": key(angles, case["reynolds"])}
    if not row["feasible"]:
        return {**out, "failed": "no feasible split", "optimum": row}
    fine = {round(b["split"], 12): b["total_pressure_drop"] for b in row["fine_bracket"]}
    bracket = sensitivity.bracket(row)
    bound = None
    if bracket is not None:
        bound = sensitivity.mesh_bound(row, fine.get(round(bracket[0][0], 12)),
                                       fine.get(round(bracket[2][0], 12)),
                                       config["search"]["xatol"])
    s_star = row["best_split"]

    ref = config["reference"]
    a, _ = pinn.scaling_coefficients(case["reynolds"], case["pressure_scaling"])

    def reference_drop(s):
        g = design_study.make_geometry(study, s, angles)
        sol = reference.solve_bifurcation(
            g, case["reynolds"], case["pressure_scaling"], reference.parabola_inlet,
            config["share"], ref["max_area"], ref["corner_levels"], config["mesh"]["corner_radius"],
            continuation=case["continuation"])
        return flow.opening_measures(sol, g, a, ref["n_line"])["total_pressure_drop"]

    reference_rows, unsolved = [], []
    for s in list(ref["splits"]) + [s_star]:
        try:
            reference_rows.append({"split": float(s), "J": reference_drop(float(s))})
        except RuntimeError as error:
            if not str(error).startswith(design_study.NO_STEADY):
                raise
            reference_rows.append({"split": float(s), "J": None})
            unsolved.append({"split": float(s), "reason": str(error)})

    g_star = design_study.make_geometry(study, s_star, angles)
    l1, l2 = sensitivity.centreline_lengths(g_star)
    c1, c2 = sensitivity.closed_lengths(g_star)
    s_l, _, _ = sensitivity.length_corrected_split(study, angles, config["share"], 1e-10)
    return {**out,
            "s_fe": s_star, "J_fe_search": row["total_pressure_drop"],
            "J_fe_check": row["total_pressure_drop_fine"], "on_bound": row["on_bound"],
            "minimum_survives_refinement": row["minimum_survives_refinement"],
            "steady_bracket": row["steady_bracket"],
            "argmin_bound": None if bound is None else bound["bound"], "mesh_bound": bound,
            "J_fe_reference_at_s_fe": reference_rows[-1]["J"],
            "reference": reference_rows[:-1], "reference_unsolved": unsolved,
            "centreline_lengths": [l1, l2], "closed_lengths": [c1, c2],
            "s_murray": sensitivity.murray_split(config["share"]), "s_length_corrected": s_l,
            "inside_pinn_range": bool(min(ref["splits"]) <= s_star <= max(ref["splits"])),
            "optimum": row, "seconds": time.perf_counter() - start}


def markdown(result):
    c = result["config"]
    lines = ["## Finite-element answer keys for the parametric PINNs", "",
             f"Geometry: parent {c['geometry']['parent_length']:g} long, outer walls "
             f"{c['geometry']['daughter_length']:g}, share q = {c['share']:g}. Search mesh "
             f"{c['mesh']['max_area']:g}/{c['mesh']['corner_levels']}, check mesh "
             f"{c['check']['max_area']:g}/{c['check']['corner_levels']}, reference mesh "
             f"{c['reference']['max_area']:g}/{c['reference']['corner_levels']}. J in the case's "
             "pressure units (inertial: rho U_c^2; viscous: mu U_c / h).", "",
             "| case | s_FE | argmin bound ε_s | J_FE (search) | J_FE (check) | J_FE (reference, at s_FE) "
             "| L1, L2 (centreline at s_FE) | s_M | s_L | survives refinement |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["cases"]:
        if "failed" in r:
            lines.append(f"| {r['key']} | failed: {r['failed']} |" + " |" * 8)
            continue
        bound = "—" if r["argmin_bound"] is None else f"{r['argmin_bound']:.4f}"
        ref = r["J_fe_reference_at_s_fe"]
        lines.append(f"| {r['key']} | {r['s_fe']:.4f} | {bound} | {r['J_fe_search']:.6g} | "
                     f"{r['J_fe_check']:.6g} | {'—' if ref is None else f'{ref:.6g}'} | "
                     f"{r['centreline_lengths'][0]:.4f}, {r['centreline_lengths'][1]:.4f} | "
                     f"{r['s_murray']:.4f} | {r['s_length_corrected']:.4f} | "
                     f"{'yes' if r['minimum_survives_refinement'] else 'NO'} |")
    splits = c["reference"]["splits"]
    lines += ["", "### J on the reference mesh", "",
              "| case | " + " | ".join(f"s = {s:.2f}" for s in splits) + " |",
              "|---|" + "---|" * len(splits)]
    for r in result["cases"]:
        if "failed" in r:
            continue
        lines.append(f"| {r['key']} | " + " | ".join(
            "—" if x["J"] is None else f"{x['J']:.6g}" for x in r["reference"]) + " |")
    return "\n".join(lines) + "\n"


def write_csvs(result, directory):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for r in result["cases"]:
        if "failed" in r:
            continue
        rows = ["s,J_fe_reference"] + [f"{x['split']:.6f},{'' if x['J'] is None else repr(x['J'])}"
                                       for x in r["reference"]]
        (directory / f"fe_{r['key']}_reference.csv").write_text("\n".join(rows) + "\n")


def run(config_path, out):
    config = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    tasks = [(config, [float(a) for a in pair], case)
             for case in config["flows"] for pair in config["layouts"]]
    start = time.perf_counter()
    cases = []
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        for r in pool.imap(answer, tasks):
            cases.append(r)
            if "failed" in r:
                print(f"  {r['key']}: FAILED ({r['failed']})", flush=True)
            else:
                print(f"  {r['key']}: s_FE {r['s_fe']:.4f} (bound {r['argmin_bound']}), "
                      f"J {r['J_fe_search']:.6g}, {r['seconds']:.0f} s", flush=True)
    result = {"config": config, "cases": cases, "seconds": time.perf_counter() - start}
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=str))
    out.with_suffix(".md").write_text(markdown(result))
    write_csvs(result, out.parent)
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
