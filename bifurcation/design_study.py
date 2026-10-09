"""How low can the pressure drop go? For each set flow share, the best widths.

The design question, by finite elements (bifurcation/reference.py), which
Network 3 is to learn. For each pair of daughter angles and each prescribed
share of the flow through daughter 1 (Q1 / Q_in, held by the flux-defective
outlet), find the width split w1 / w0 with the lowest flux-weighted
total-pressure drop, w1 + w2 = w0, both widths at least min_width:

    1. scan the split on a grid (every geometry the bounds allow; any the
       geometry refuses is reported, not skipped silently)
    2. refine with a bounded Brent search inside the bracket around the grid's
       best point
    3. re-solve each optimum, and the grid neighbours either side, on a finer
       mesh: the minimum must survive refinement, and the change in the
       pressure drop is the mesh's error estimate

Two comparisons come with every optimum:
    equal widths       the pressure drop at split 0.5, the design without
                       optimisation
    friction law       w_i ~ Q_i^(1/2): minimising sum Q_i^2 L / w_i^3 (plane
                       Poiseuille daughters) under w1 + w2 = w0 gives
                       w1 / w0 = sqrt(q) / (sqrt(q) + sqrt(1 - q)). The
                       distance from it is what the junction and the angles do.

Equal outlet pressures (no share set) are included for reference: there the
optimum sits on a width bound, the narrow daughter carrying almost nothing.

    python -m bifurcation.design_study --config bifurcation/study_re10_shares.yaml
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

from bifurcation import reference
from bifurcation.geometry import Geometry, GeometryError, split_bounds
from developing_flow import pinn

ConfigError = pinn.ConfigError

#: How the solver reports a flow with no steady solution Newton can reach.
NO_STEADY = "Newton did not converge"

SCHEMA = {
    "name": str,
    "physics": {"reynolds": float, "pressure_scaling": str,
                "continuation": list},   # lower Re solved first, each starting Newton at the next
    "geometry": {"parent_length": float, "min_width": float},   # plus one of LENGTH_RULES
    "angles": list,               # [[angle1, angle2], ...] in degrees
    "shares": list,               # Q1 / Q_in values in (0, 1)
    "search": {"grid": int, "xatol": float},
    "mesh": {"max_area": float, "corner_levels": int, "corner_radius": float},
    "check": {"max_area": float, "corner_levels": int},
    "workers": int,
}


#: How the daughters' length is set, one key of the two in `geometry`:
#: daughter_length  both outer walls this long (the paper's Re_h = 10 map);
#: closed_length    both daughters have both walls over this length
#:                  (Geometry.from_split_closed), so the friction law's optimum is
#:                  exactly the Murray split.
LENGTH_RULES = ("daughter_length", "closed_length")


def length_rule(geometry_config):
    rules = [k for k in LENGTH_RULES if k in geometry_config]
    if len(rules) != 1:
        raise ConfigError(f"geometry needs exactly one of {list(LENGTH_RULES)}, got {rules}")
    return rules[0]


def parse_config(document):
    document = copy.deepcopy(document)
    schema = copy.deepcopy(SCHEMA)
    if isinstance(document.get("geometry"), dict):
        schema["geometry"][length_rule(document["geometry"])] = float
    pinn._check(document, schema)
    steps = document["physics"]["continuation"] + [document["physics"]["reynolds"]]
    if any(isinstance(r, bool) or not isinstance(r, (int, float)) for r in steps) or any(
            not 0.0 < lo < hi for lo, hi in zip(steps, steps[1:])):
        raise ConfigError("physics.continuation must be increasing Reynolds numbers below "
                          f"physics.reynolds, got {document['physics']['continuation']}")
    for pair in document["angles"]:
        if not (isinstance(pair, list) and len(pair) == 2):
            raise ConfigError(f"angles must be pairs [angle1, angle2], got {pair!r}")
    for share in document["shares"]:
        if not 0.0 < share < 1.0:
            raise ConfigError(f"every share must lie in (0, 1), got {share}")
    if document["search"]["grid"] < 3:
        raise ConfigError("search.grid must be at least 3")
    return document


def make_geometry(config, split, angles):
    """The design geometry at one split, by the config's length rule."""
    g = config["geometry"]
    if length_rule(g) == "closed_length":
        return Geometry.from_split_closed(split, angles[0], angles[1], g["parent_length"],
                                          g["closed_length"], g["min_width"])
    return Geometry.from_split(split, angles[0], angles[1], g["parent_length"],
                               g["daughter_length"], g["min_width"])


def solve(config, split, angles, share, max_area, corner_levels):
    g = make_geometry(config, split, angles)
    sol = reference.solve_bifurcation(
        g, config["physics"]["reynolds"], config["physics"]["pressure_scaling"],
        reference.parabola_inlet, share, max_area, corner_levels,
        config["mesh"]["corner_radius"], continuation=config["physics"]["continuation"])
    return sol.measures()


def friction_split(share, min_width):
    """The friction-law optimum w1 / w0 = sqrt(q) / (sqrt(q) + sqrt(1 - q)), in the bounds."""
    lo, hi = split_bounds(min_width)
    s = math.sqrt(share) / (math.sqrt(share) + math.sqrt(1.0 - share))
    return min(max(s, lo), hi)


def optimise(task):
    """The best split for one (angles, share); share None = equal outlet pressures."""
    config, angles, share = task
    mesh = config["mesh"]
    lo, hi = split_bounds(config["geometry"]["min_width"])
    start = time.perf_counter()
    solves = 0
    cache = {}

    refused, unsolved = [], []

    def drop(split):
        """The coarse-mesh drop. A geometry the rules refuse, or one with no
        steady solution Newton can reach, counts as infinite and is recorded;
        an optimum next to an unsolved split is flagged, not trusted."""
        nonlocal solves
        key = round(float(split), 12)
        if key not in cache:
            try:
                solves += 1
                cache[key] = solve(config, float(split), angles, share, mesh["max_area"],
                                   mesh["corner_levels"])
            except GeometryError as error:
                cache[key] = {"total_pressure_drop": math.inf}
                refused.append({"split": float(split), "reason": str(error)})
            except RuntimeError as error:
                if not str(error).startswith(NO_STEADY):
                    raise
                cache[key] = {"total_pressure_drop": math.inf}
                unsolved.append({"split": float(split), "mesh": "search", "reason": str(error)})
        return cache[key]["total_pressure_drop"]

    grid = np.linspace(lo, hi, config["search"]["grid"])
    values = np.array([drop(s) for s in grid])
    if not np.isfinite(values).any():
        return {"angles": angles, "share": share, "refused": refused, "unsolved": unsolved,
                "feasible": False}
    k = int(np.argmin(values))
    left, right = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    if k in (0, len(grid) - 1):
        best_split, best = float(grid[k]), float(values[k])   # on a bound: nothing to refine
        on_bound = True
    else:
        result = minimize_scalar(drop, bounds=(left, right), method="bounded",
                                 options={"xatol": config["search"]["xatol"]})
        best_split, best = float(result.x), float(result.fun)
        if values[k] < best:                                   # Brent never does worse
            best_split, best = float(grid[k]), float(values[k])
        on_bound = False
    measured = cache[round(best_split, 12)]
    # Trusted only if every split in the optimum's bracket had a steady solution.
    steady_bracket = not any(left <= u["split"] <= right for u in unsolved)

    # the optimum and its neighbours on the finer mesh
    check = config["check"]
    fine = {}
    for s in sorted({float(left), best_split, float(right)}):
        try:
            fine[s] = solve(config, s, angles, share, check["max_area"],
                            check["corner_levels"])["total_pressure_drop"]
        except GeometryError:
            fine[s] = math.inf
        except RuntimeError as error:
            if not str(error).startswith(NO_STEADY):
                raise
            fine[s] = math.inf
            unsolved.append({"split": s, "mesh": "check", "reason": str(error)})
            steady_bracket = False
    fine_best = fine[best_split] if math.isfinite(fine[best_split]) else None
    survives = fine_best is not None and all(fine_best <= v for v in fine.values())

    def finite(value):
        return value if value is not None and math.isfinite(value) else None

    equal = finite(drop(0.5)) if lo <= 0.5 <= hi else None
    friction = friction_split(share, config["geometry"]["min_width"]) if share else None
    friction_drop = None if friction is None else finite(drop(friction))
    return {
        "angles": angles, "share": share, "feasible": True,
        "best_split": best_split, "w1": 2.0 * best_split, "w2": 2.0 * (1.0 - best_split),
        "on_bound": on_bound,
        "total_pressure_drop": best,
        "total_pressure_drop_fine": fine_best,
        "mesh_change": None if fine_best is None else abs(fine_best - best) / abs(fine_best),
        "steady_bracket": steady_bracket,
        "fine_bracket": [{"split": s, "total_pressure_drop": finite(v)} for s, v in fine.items()],
        "minimum_survives_refinement": survives,
        "equal_widths_drop": equal,
        "saving_vs_equal_widths": None if equal is None else 1.0 - best / equal,
        "friction_law_split": friction, "friction_law_drop": friction_drop,
        "outlet1_pressure_offset": measured["outlet1_pressure_offset"],
        "flow_split": measured["split_flux"],
        "grid": [{"split": float(s), "total_pressure_drop": float(v)}
                 for s, v in zip(grid, values)],
        "refused": refused, "unsolved": unsolved, "solves": solves,
        "seconds": time.perf_counter() - start,
    }


def run(config_path, out):
    config = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    tasks = [(config, list(map(float, pair)), share)
             for pair in config["angles"] for share in [None] + list(config["shares"])]
    start = time.perf_counter()
    # spawn, not fork: this module imports JAX (through developing_flow.pinn),
    # and forking a process with JAX's threads running can deadlock
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        rows = []
        for row in pool.imap(optimise, tasks):
            rows.append(row)
            if row["feasible"]:
                label = "equal pressures" if row["share"] is None else f"share {row['share']:.2f}"
                print(f"  {row['angles'][0]:g}/{row['angles'][1]:g} {label}: best split "
                      f"{row['best_split']:.4f}{' (bound)' if row['on_bound'] else ''}, "
                      f"drop {row['total_pressure_drop']:.5f} (fine {row['total_pressure_drop_fine']}), "
                      f"{len(row['unsolved'])} unsolved, "
                      f"{row['solves']} solves, {row['seconds']:.0f} s", flush=True)
    summary = {"config": config, "rows": rows, "seconds": time.perf_counter() - start}
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
    out.with_suffix(".md").write_text(markdown(summary))
    plot(summary, out.with_suffix(".png"))
    return summary


def markdown(summary):
    config = summary["config"]
    g = config["geometry"]
    lines = [
        f"## How low can the pressure drop go? Re_h = {config['physics']['reynolds']:g}",
        "",
        f"Finite elements (P2/P1, mesh max area {config['mesh']['max_area']}, "
        f"{config['mesh']['corner_levels']} corner refinements), parabolic inflow, parent "
        f"{g['parent_length']:g} long, " + (
            f"daughters {g['daughter_length']:g} long along the outer wall, "
            if length_rule(g) == "daughter_length" else
            f"each daughter with both walls over {g['closed_length']:g}, ") +
        f"w1 + w2 = 2, every width >= {g['min_width']:g}. 'share' is the fraction of the "
        "flow daughter 1 must carry. 'drop' is the flux-weighted total-pressure drop "
        "(E_in - E_out1 - E_out2) / Q_in, in units of rho U^2. 'fine' re-solves the "
        f"optimum on max area {config['check']['max_area']} with "
        f"{config['check']['corner_levels']} refinements. 'friction law' is "
        "w1 / w0 = sqrt(q) / (sqrt(q) + sqrt(1 - q)).",
        "",
    ]
    by_angles = {}
    for row in summary["rows"]:
        by_angles.setdefault(tuple(row["angles"]), []).append(row)
    lines += ["### At a glance", "",
              "| angles | lowest drop at a set share (share, split) | at share 0.5 | "
              "equal outlet pressures (floor) | largest gap to the friction-law split |",
              "|---|---|---|---|---|"]
    for angles, rows in by_angles.items():
        shared = [r for r in rows if r["feasible"] and r["share"] is not None]
        if not shared:
            continue
        low = min(shared, key=lambda r: r["total_pressure_drop"])
        half = [r for r in shared if abs(r["share"] - 0.5) < 1e-12]
        free = [r for r in rows if r["feasible"] and r["share"] is None]
        gap = max(shared, key=lambda r: abs(r["best_split"] - r["friction_law_split"]))
        lines.append(
            f"| {angles[0]:g} / {angles[1]:g} | {low['total_pressure_drop']:.3f} "
            f"({low['share']:.2f}, {low['best_split']:.3f}) | "
            + (f"{half[0]['total_pressure_drop']:.3f} (split {half[0]['best_split']:.3f})"
               if half else "-") + " | "
            + (f"{free[0]['total_pressure_drop']:.3f} (split {free[0]['best_split']:.3f}, "
               f"{100 * min(free[0]['flow_split'], 1 - free[0]['flow_split']):.1f}% of the flow "
               "in the narrow daughter)" if free else "-")
            + f" | {abs(gap['best_split'] - gap['friction_law_split']):.3f} at share "
              f"{gap['share']:.2f} |")
    lines.append("")
    for angles, rows in by_angles.items():
        lines += [f"### Angles {angles[0]:g} / {angles[1]:g} (bifurcation angle "
                  f"{angles[0] + angles[1]:g})", "",
                  "| share | best split w1/w0 | w1, w2 | lowest drop | fine mesh | "
                  "equal widths | saving | friction-law split | outlet 1 offset |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if not r["feasible"]:
                lines.append(f"| {r['share']} | no solution (refused, or no steady flow) | | | | | | | |")
                continue
            share = "equal pressures" if r["share"] is None else f"{r['share']:.2f}"
            fmt = lambda v, f: "-" if v is None else format(v, f)   # noqa: E731
            lines.append(
                f"| {share} | {r['best_split']:.4f}{' (bound)' if r['on_bound'] else ''}"
                f"{'' if r['minimum_survives_refinement'] else ' (moves on refinement)'}"
                f"{'' if r.get('steady_bracket', True) else ' (no steady solution nearby)'} | "
                f"{r['w1']:.3f}, {r['w2']:.3f} | {r['total_pressure_drop']:.5f} | "
                f"{fmt(r['total_pressure_drop_fine'], '.5f')} | {fmt(r['equal_widths_drop'], '.5f')} | "
                f"{fmt(r['saving_vs_equal_widths'] and 100 * r['saving_vs_equal_widths'], '.1f')}% | "
                f"{fmt(r['friction_law_split'], '.4f')} | {r['outlet1_pressure_offset']:+.4f} |")
        lines.append("")
    lines.append(f"{sum(r.get('solves', 0) for r in summary['rows'])} finite-element solves "
                 f"(plus the fine-mesh checks), {summary['seconds'] / 60:.1f} min on "
                 f"{config['workers']} processes.")
    return "\n".join(lines) + "\n"


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6B6B6B"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    by_angles = {}
    for row in summary["rows"]:
        if row["feasible"] and row["share"] is not None:
            by_angles.setdefault(tuple(row["angles"]), []).append(row)
    for colour, (angles, rows) in zip(series, by_angles.items()):
        rows = sorted(rows, key=lambda r: r["share"])
        q = [r["share"] for r in rows]
        label = f"{angles[0]:g} / {angles[1]:g}"
        ax1.plot(q, [r["total_pressure_drop"] for r in rows], "o-", color=colour, label=label)
        ax2.plot(q, [r["best_split"] for r in rows], "o-", color=colour, label=label)
    q = np.linspace(0.01, 0.99, 200)
    ax2.plot(q, [friction_split(v, summary["config"]["geometry"]["min_width"]) for v in q],
             "--", color="#333333", label="friction law sqrt(q)")
    ax1.set_xlabel("share of the flow through daughter 1, Q1 / Q_in")
    ax1.set_ylabel("lowest total-pressure drop (best split)")
    ax2.set_xlabel("share of the flow through daughter 1, Q1 / Q_in")
    ax2.set_ylabel("best split w1 / w0")
    for ax in (ax1, ax2):
        ax.grid(color="#E6E6E6")
        ax.legend(fontsize=8)
    fig.suptitle("For each set flow share, the width split with the lowest pressure drop",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def merge(paths, out):
    """One study from several with the same physics, geometry, search and
    meshes (only angles and shares may differ): their optima side by side."""
    studies = [json.loads(pathlib.Path(p).read_text()) for p in paths]
    fixed = ("physics", "geometry", "search", "mesh", "check")
    first = studies[0]["config"]
    for path, study in zip(paths[1:], studies[1:]):
        for key in fixed:
            if study["config"][key] != first[key]:
                raise ConfigError(f"{path}: {key} differs from {paths[0]}'s; its optima are "
                                  "not comparable")
    seen = set()
    rows = []
    for study in studies:
        for row in study["rows"]:
            key = (tuple(row["angles"]), row["share"])
            if key in seen:
                raise ConfigError(f"angles {row['angles']} share {row['share']} appear twice")
            seen.add(key)
            rows.append(row)
    config = copy.deepcopy(first)
    config["name"] = "+".join(s["config"]["name"] for s in studies)
    config["angles"] = [list(a) for a in dict.fromkeys(tuple(r["angles"]) for r in rows)]
    config["shares"] = sorted({r["share"] for r in rows if r["share"] is not None})
    summary = {"config": config, "rows": rows,
               "seconds": sum(s["seconds"] for s in studies), "merged_from": list(paths)}
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
    out.with_suffix(".md").write_text(markdown(summary))
    plot(summary, out.with_suffix(".png"))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config")
    parser.add_argument("--merge", nargs="+", help="study JSONs to combine instead of running")
    parser.add_argument("--out", required=True, help="path stem: writes .md, .json, .png")
    args = parser.parse_args(argv)
    if (args.config is None) == (args.merge is None):
        parser.error("give exactly one of --config and --merge")
    if args.merge:
        merge(args.merge, args.out)
    else:
        run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
