"""How much of the departure from the Murray split is real? Four checks on the
design study (bifurcation/design_study.py), asked for in review.

1. Murray regret. J at the Murray split against J at the resolved optimum,
   J(s_M)/J* - 1, both on the study's coarse mesh: what ignoring the junction
   costs in pressure drop, over every one of the 165 optima.

2. Daughter lengths. The study fixes each daughter's OUTER wall at L_d, but
   the septum tip T moves with the angles and the split, so the daughters'
   centrelines differ in length whenever the layout is asymmetric. With no
   junction loss at all, unequal lengths move the optimum (paper, A.2(b)):
       w1/w2 = (L1/L2)^(1/4) (Q1/Q2)^(1/2).
   Each daughter's centreline runs from the midpoint of its entrance (its
   outer corner to the septum tip T; for the face septum, to the start of
   its inner wall) to the midpoint of its outlet. The length-corrected Murray
   split s_L solves the relation above with the lengths of the geometry at
   s_L itself. Departures from s_L are what the lengths do not explain.

3. Where the optimum is, within the mesh error. The argmin was located on the
   coarse mesh. A mesh error e(s) = J_fine/J_coarse - 1 that varies with s
   moves the argmin by about -e'/(J''/J) (A.3). e is measured at s* and at
   both bracketing grid points (fine solves of the study's check mesh); J''/J
   is the curvature of the coarse J through the same three points. The bound
   on s* is |delta_s| + the search tolerance.

4. H2, quantitatively. (a) Each daughter's loss coefficient measured on the
   finite-element solution at s*: its path's drop in flux-weighted total
   pressure, less fully developed friction in the parent and along its own
   centreline, per unit of 1/2 U_i^2. (b) The junction-loss model adds
   sum_i K_i (1/2) rho U_i^2 to the friction law with the daughters' own
   lengths. K is fitted, one value per layout, on the symmetric layouts
   (K1 = K2 = K(alpha); a negative K means the lengths over-explain the
   departure there), then K(alpha) = a + b (1 - cos alpha) predicts the
   asymmetric layouts' optima, which were not used, and the shares at which
   s - s_M changes sign.

    python -m bifurcation.sensitivity --config bifurcation/sensitivity_re10.yaml \\
        --out docs/bifurcation_re10-sensitivity
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
from scipy.optimize import minimize_scalar

from bifurcation import design_study
from bifurcation.geometry import Geometry, GeometryError, corner_points, split_bounds
from developing_flow import pinn

ROOT = pathlib.Path(__file__).resolve().parent.parent
ConfigError = pinn.ConfigError

SCHEMA = {
    "name": str,
    "data": str,                       # the design study's merged JSON
    "workers": int,
    "lengths": {"tolerance": float},   # fixed-point tolerance for s_L
    "junction": {"fit_layouts": list, "k_max": float, "xatol": float},
}


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    for pair in document["junction"]["fit_layouts"]:
        if not (isinstance(pair, list) and len(pair) == 2 and pair[0] == pair[1]):
            raise ConfigError("junction.fit_layouts must be symmetric pairs [a, a]")
    return document


# --------------------------------------------------------------------------- #
# Lengths and the length-corrected Murray split
# --------------------------------------------------------------------------- #


def murray_split(share):
    return math.sqrt(share) / (math.sqrt(share) + math.sqrt(1.0 - share))


def centreline_lengths(geometry: Geometry):
    """(L1, L2): each daughter's centreline, entrance midpoint to outlet midpoint."""
    q = corner_points(geometry.w1, geometry.w2, math.radians(geometry.angle1),
                      math.radians(geometry.angle2), geometry.parent_length,
                      geometry.daughter_length)
    if geometry.septum == "face":
        inner1, inner2 = q["start1"], q["start2"]      # the face closes the junction
    else:
        inner1 = inner2 = q["T"]
    l1 = np.linalg.norm(0.5 * (q["E1o"] + q["E1i"]) - 0.5 * (q["C"] + inner1))
    l2 = np.linalg.norm(0.5 * (q["E2o"] + q["E2i"]) - 0.5 * (q["B"] + inner2))
    return float(l1), float(l2)


def make_geometry(study_config, angles, split):
    g = study_config["geometry"]
    return Geometry.from_split(split, angles[0], angles[1], g["parent_length"],
                               g["daughter_length"], g["min_width"])


def length_corrected_split(study_config, angles, share, tolerance):
    """s_L: s/(1-s) = (L1/L2)^(1/4) sqrt(q/(1-q)), lengths of the geometry at s_L."""
    lo, hi = split_bounds(study_config["geometry"]["min_width"])
    s = murray_split(share)
    for _ in range(200):
        l1, l2 = centreline_lengths(make_geometry(study_config, angles, min(max(s, lo), hi)))
        ratio = (l1 / l2) ** 0.25 * math.sqrt(share / (1.0 - share))
        new = ratio / (1.0 + ratio)
        if abs(new - s) < tolerance:
            return new, l1, l2
        s = new
    raise RuntimeError(f"s_L did not converge for {angles} share {share}")


# --------------------------------------------------------------------------- #
# The junction-loss model
# --------------------------------------------------------------------------- #


def model_drop(study_config, angles, share, split, k1, k2):
    """Friction with the daughters' own lengths plus sum K_i (1/2) U_i^2, per unit flow
    (the parent's fixed friction left out)."""
    inertia, nu = pinn.scaling_coefficients(study_config["physics"]["reynolds"],
                                            study_config["physics"]["pressure_scaling"])
    q0, w0 = 4.0 / 3.0, 2.0
    l1, l2 = centreline_lengths(make_geometry(study_config, angles, split))
    friction = 12.0 * nu * q0 / w0**3 * (l1 * share**2 / split**3
                                        + l2 * (1 - share) ** 2 / (1 - split) ** 3)
    junction = inertia * q0**2 / (2.0 * w0**2) * (k1 * share**3 / split**2
                                        + k2 * (1 - share) ** 3 / (1 - split) ** 2)
    return friction + junction


def model_split(study_config, angles, share, k1, k2, xatol):
    lo, hi = split_bounds(study_config["geometry"]["min_width"])
    s_m = murray_split(share)
    a, b = max(lo, s_m - 0.2), min(hi, s_m + 0.2)

    def objective(s):
        try:
            return model_drop(study_config, angles, share, s, k1, k2)
        except GeometryError:
            return math.inf

    return float(minimize_scalar(objective, bounds=(a, b), method="bounded",
                                 options={"xatol": xatol}).x)


def loss_coefficients(study_config, angles, share, split, measures):
    """(K1, K2) measured on one finite-element solution: each path's drop in
    flux-weighted total pressure, less fully developed friction in the parent and
    along the daughter's own centreline, per unit of the daughter's 1/2 U_i^2."""
    inertia, nu = pinn.scaling_coefficients(study_config["physics"]["reynolds"],
                                            study_config["physics"]["pressure_scaling"])
    q0, w0 = 4.0 / 3.0, 2.0
    parent = 12.0 * nu * q0 * study_config["geometry"]["parent_length"] / w0**3
    lengths = centreline_lengths(make_geometry(study_config, angles, split))
    out = []
    for i, (q, w, length) in enumerate(((share, w0 * split, lengths[0]),
                                        (1.0 - share, w0 * (1.0 - split), lengths[1])), 1):
        drop = measures["total_pressure_in"] - measures[f"total_pressure_out{i}"]
        friction = 12.0 * nu * q * q0 * length / w**3
        out.append((drop - parent - friction) / (0.5 * inertia * (q * q0 / w) ** 2))
    return tuple(out)


def _junction_task(task):
    """K1, K2 at one optimum, on the mesh the search used."""
    study_config, angles, share, split = task
    mesh = study_config["mesh"]
    m = design_study.solve(study_config, split, angles, share, mesh["max_area"],
                           mesh["corner_levels"])
    return loss_coefficients(study_config, angles, share, split, m), m["total_pressure_drop"]


def k_of_angle(alpha, a, b):
    return a + b * (1.0 - math.cos(math.radians(alpha)))


# --------------------------------------------------------------------------- #
# Mesh-error bound on the argmin
# --------------------------------------------------------------------------- #


def _fine_neighbours(task):
    """Fine-mesh J at the two grid points bracketing one optimum."""
    study_config, angles, share, left, right = task
    check = study_config["check"]
    out = {}
    for s in (left, right):
        try:
            out[s] = design_study.solve(study_config, s, angles, share, check["max_area"],
                                        check["corner_levels"])["total_pressure_drop"]
        except GeometryError:
            out[s] = None
    return {"angles": angles, "share": share, "fine": out}


def bracket(row):
    """(left, s*, right) and their coarse J, from the stored grid scan."""
    grid = [(g["split"], g["total_pressure_drop"]) for g in row["grid"]]
    k = min(range(len(grid)), key=lambda i: grid[i][1])
    if k in (0, len(grid) - 1):
        return None
    return grid[k - 1], (row["best_split"], row["total_pressure_drop"]), grid[k + 1]


def mesh_bound(row, fine_left, fine_right, tolerance):
    """delta_s = -e'/(J''/J), and the bound |delta_s| + tolerance."""
    (sl, jl), (s0, j0), (sr, jr) = bracket(row)
    if fine_left is None or fine_right is None:
        return None
    xs = np.array([sl, s0, sr])
    e = np.array([fine_left / jl - 1.0, row["total_pressure_drop_fine"] / j0 - 1.0,
                  fine_right / jr - 1.0])
    slope = float(np.polyfit(xs, e, 1)[0])
    curvature = float(2.0 * np.polyfit(xs, np.array([jl, j0, jr]), 2)[0]) / j0
    shift = -slope / curvature
    return {"mesh_error_slope": slope, "curvature": curvature, "argmin_shift": shift,
            "bound": abs(shift) + tolerance}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def reversal_share(shares, departures):
    """The first share at which a departure turns from negative to positive, as
    the bracketing pair of sampled shares, or None if it never does."""
    for (qa, da), (qb, db) in zip(zip(shares, departures), zip(shares[1:], departures[1:])):
        if da < 0.0 <= db:
            return [qa, qb]
    return None


def reversal_table(records):
    """Per layout below 180 degrees, the shares between which s - s_M turns from
    negative to positive: observed, for s_L, and for the junction model. At 180
    degrees the lengths are equal by construction (s_L = s_M to round-off) and
    the model is not applied, so those layouts are left out."""
    table = {}
    for angles in sorted({tuple(r["angles"]) for r in records}):
        if sum(angles) >= 180.0:
            continue
        rs = sorted((r for r in records if tuple(r["angles"]) == angles),
                    key=lambda r: r["share"])
        shares = [r["share"] for r in rs]
        table[f"{angles[0]:g}/{angles[1]:g}"] = {
            "observed": reversal_share(shares, [r["departure_murray"] for r in rs]),
            "lengths": reversal_share(shares, [r["s_length"] - r["s_murray"] for r in rs]),
            "model": reversal_share(shares, [r["s_model"] - r["s_murray"] for r in rs])}
    return table


def report(out):
    """Re-derive the reversal table and the markdown from a stored run."""
    out = pathlib.Path(out)
    summary = json.loads(out.with_suffix(".json").read_text())
    summary["reversals"] = reversal_table(summary["records"])
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    out.with_suffix(".md").write_text(markdown(summary))
    return summary


def run(config_path, out):
    config = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    study = json.loads((ROOT / config["data"]).read_text())
    sc = study["config"]
    tolerance = sc["search"]["xatol"]
    rows = [r for r in study["rows"] if r["feasible"] and r["share"] is not None]
    start = time.perf_counter()

    # 1-2. regret and lengths
    records = []
    for r in rows:
        s_l, l1, l2 = length_corrected_split(sc, r["angles"], r["share"],
                                             config["lengths"]["tolerance"])
        records.append({
            "angles": r["angles"], "share": r["share"], "s_star": r["best_split"],
            "s_murray": r["friction_law_split"], "s_length": s_l, "L1": l1, "L2": l2,
            "departure_murray": r["best_split"] - r["friction_law_split"],
            "departure_length": r["best_split"] - s_l,
            "murray_regret": r["friction_law_drop"] / r["total_pressure_drop"] - 1.0,
            "equal_widths_regret": r["equal_widths_drop"] / r["total_pressure_drop"] - 1.0,
        })

    # 3 and 4a. fine solves at the bracketing grid points; coarse re-solves at s*
    fine_tasks, junction_tasks = [], []
    for r in rows:
        b = bracket(r)
        fine_tasks.append((sc, r["angles"], r["share"], b[0][0], b[2][0]))
        junction_tasks.append((sc, r["angles"], r["share"], r["best_split"]))
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        fine = list(pool.imap(_fine_neighbours, fine_tasks))
        measured = list(pool.imap(_junction_task, junction_tasks))
    for rec, r, f, t, (ks, drop) in zip(records, rows, fine, fine_tasks, measured):
        bound = mesh_bound(r, f["fine"][t[3]], f["fine"][t[4]], tolerance)
        rec["mesh"] = bound
        if bound is not None:
            rec["resolved_murray"] = abs(rec["departure_murray"]) > bound["bound"]
            rec["resolved_length"] = abs(rec["departure_length"]) > bound["bound"]
        if abs(drop / r["total_pressure_drop"] - 1.0) > 1e-9:
            raise RuntimeError(f"re-solve at {r['angles']} {r['share']} does not reproduce J*")
        rec["K_measured"] = list(ks)

    # 4b. junction model: K per symmetric layout, then K(alpha), then predictions
    xatol = config["junction"]["xatol"]
    fits = {}
    for a1, a2 in config["junction"]["fit_layouts"]:
        mine = [r for r in rows if r["angles"] == [a1, a2]]

        def misfit(k, mine=mine, a1=a1, a2=a2):
            return sum((model_split(sc, [a1, a2], r["share"], k, k, xatol)
                        - r["best_split"]) ** 2 for r in mine)

        # K may come out negative: the lengths then over-explain the departure there
        k = float(minimize_scalar(misfit, bounds=(-config["junction"]["k_max"],
                                                  config["junction"]["k_max"]),
                                  method="bounded", options={"xatol": 1e-3}).x)
        fits[a1] = {"K": k, "rms": math.sqrt(misfit(k) / len(mine)),
                    "rms_lengths_only": math.sqrt(misfit(0.0) / len(mine))}
    alphas = np.array(sorted(fits))
    ks = np.array([fits[a]["K"] for a in alphas])
    design = np.stack([np.ones_like(alphas), 1.0 - np.cos(np.radians(alphas))], axis=1)
    (ka, kb), *_ = np.linalg.lstsq(design, ks, rcond=None)
    for rec in records:
        a1, a2 = rec["angles"]
        if a1 + a2 >= 180.0:
            continue
        rec["s_model"] = model_split(sc, rec["angles"], rec["share"], k_of_angle(a1, ka, kb),
                                     k_of_angle(a2, ka, kb), xatol)
        rec["model_error"] = rec["s_star"] - rec["s_model"]

    reversals = reversal_table(records)
    summary = {"config": config, "study": config["data"], "records": records,
               "junction": {"per_layout": fits, "a": float(ka), "b": float(kb),
                            "form": "K(alpha) = a + b (1 - cos alpha)"},
               "reversals": reversals, "search_tolerance": tolerance,
               "seconds": time.perf_counter() - start}
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    out.with_suffix(".md").write_text(markdown(summary))
    return summary


def _median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def markdown(summary):
    recs = summary["records"]
    lay = []
    for r in recs:
        if r["angles"] not in lay:
            lay.append(r["angles"])
    lay.sort(key=lambda a: (a[0] + a[1], a[1] - a[0]))
    lines = ["## How much of the departure from the Murray split is real? Re_h = 10", "",
             "s_M: Murray split. s_L: Murray split corrected for the daughters' centreline "
             "lengths (no fitted parameter). s_model: lengths plus the junction-loss model, "
             "K(alpha) fitted on the symmetric layouts only. Bound: |mesh-induced argmin "
             f"shift| + search tolerance ({summary['search_tolerance']:g}). Resolved: "
             "departures larger than their own bound. Murray regret: J(s_M)/J* - 1, both on "
             "the search mesh.", "",
             "| layout | max abs(s*-s_M) | max abs(s*-s_L) | max abs(s*-s_model) | max bound | "
             "resolved vs s_M | resolved vs s_L | Murray regret: median, max |",
             "|---|---|---|---|---|---|---|---|"]
    for a in lay:
        rs = [r for r in recs if r["angles"] == a]
        mm = max(abs(r["departure_murray"]) for r in rs)
        ml = max(abs(r["departure_length"]) for r in rs)
        mk = (max(abs(r["model_error"]) for r in rs) if "model_error" in rs[0] else None)
        bounds = [r["mesh"]["bound"] for r in rs if r["mesh"]]
        rm = sum(1 for r in rs if r.get("resolved_murray"))
        rl = sum(1 for r in rs if r.get("resolved_length"))
        reg = [r["murray_regret"] for r in rs]
        lines.append(f"| {a[0]:g}/{a[1]:g} | {mm:.4f} | {ml:.4f} | "
                     f"{'-' if mk is None else f'{mk:.4f}'} | {max(bounds):.4f} | {rm}/{len(rs)} | "
                     f"{rl}/{len(rs)} | {100 * _median(reg):.3f}%, {100 * max(reg):.3f}% |")

    def regret_line(name, chosen):
        reg = [r["murray_regret"] for r in chosen]
        return (f"- {name} ({len(chosen)} optima): median {100 * _median(reg):.4f}%, "
                f"max {100 * max(reg):.3f}%")

    below = [r for r in recs if sum(r["angles"]) < 180.0]
    lines += ["", "Murray regret:", regret_line("all", recs),
              regret_line("bifurcation angle up to 90 degrees",
                          [r for r in recs if sum(r["angles"]) <= 90.0]),
              regret_line("120 and 150 degrees", [r for r in below if sum(r["angles"]) > 90.0]),
              regret_line("180 degrees", [r for r in recs if sum(r["angles"]) >= 180.0])]
    j = summary["junction"]
    lines += ["", f"Junction-loss fit on the symmetric layouts: {j['form']}, a = {j['a']:.3f}, "
              f"b = {j['b']:.3f}. Per layout: " + "; ".join(
                  f"{float(k):g}/{float(k):g}: K = {v['K']:.3f} (rms {v['rms']:.4f}, lengths only "
                  f"{v['rms_lengths_only']:.4f})" for k, v in j["per_layout"].items()), "",
              "Measured loss coefficients at q = 0.5 (path drop less centreline friction, per "
              "1/2 U_i^2): " + "; ".join(
                  f"{r['angles'][0]:g}/{r['angles'][1]:g}: {r['K_measured'][0]:.2f}, "
                  f"{r['K_measured'][1]:.2f}" for a in lay for r in recs
                  if r["angles"] == a and r["share"] == 0.5), "",
              "Reversal shares below 180 degrees (s - s_M turns from negative to positive "
              "between):", ""]
    for name, entry in sorted(summary["reversals"].items()):
        if entry["observed"] is None and entry["lengths"] is None and entry["model"] is None:
            continue
        lines.append(f"- {name}: observed {entry['observed']}, lengths {entry['lengths']}, "
                     f"lengths + junction model {entry['model']}")
    lines += ["", f"{summary['seconds'] / 60:.1f} min."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", action="store_true",
                        help="re-derive the tables from the stored run at --out, no solves")
    args = parser.parse_args(argv)
    if args.report:
        report(args.out)
    else:
        run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
