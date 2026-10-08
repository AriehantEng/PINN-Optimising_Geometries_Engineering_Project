"""Baselines for the learned design rule, and how much the length-corrected
Murray split depends on how a daughter's length is counted.

    python -m bifurcation.baselines --config bifurcation/baselines_re10.yaml \\
        --out docs/bifurcation_re10-baselines [--parts heldout lengths]

1. Held-out regret (`heldout`, finite elements). On the angle pairs Network 3
   never saw, the optimum is found again by the design study's own procedure
   (deterministic: the same optimum Network 3 was scored against), and J is
   solved on the same mesh at three proposed splits: the Murray split s_M, the
   length-corrected split s_L (free to compute), and Network 3's split as
   reported in its results file. Regret = J(proposed)/J* - 1.

2. Length definition (`lengths`, geometry only). Strip area over width equals
   the centreline length and so does the mean of the outer- and inner-wall
   lengths (the midline of a trapezoid), so neither is an independent check. The
   alternatives are the inner wall alone and the closed length, over which a
   daughter has both walls. For each, per layout below 180 degrees: the share of
   the squared departure from s_M removed, the largest residual, the residuals
   beyond their location bound and their sign, and how far the optimum's drop
   lies above the friction law with those lengths.

3. Loss coefficients against closed lengths (`junction`, no new solves). The
   sensitivity run measured K_i against centreline friction; moving the friction
   baseline to the closed lengths changes K_i by an exact, known amount. With
   these K, the friction law (closed lengths) plus K_i (1/2) U_i^2 is minimised
   over s and compared with s*.

4. Regret of s_L over the map (`map`, finite elements). For every optimum below
   180 degrees, J is solved on the search mesh at s_L with centreline and with
   closed lengths: what a designer pays for using the free rule.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import pathlib
import re
import statistics

import yaml
from scipy.optimize import minimize_scalar

from bifurcation import design_study
from bifurcation import sensitivity as S
from bifurcation.geometry import GeometryError
from developing_flow import pinn

ROOT = pathlib.Path(__file__).resolve().parent.parent
ConfigError = pinn.ConfigError
PARTS = ("heldout", "lengths", "junction", "map")
DEFINITIONS = {"centreline": S.centreline_lengths, "inner wall": S.inner_wall_lengths,
               "closed": S.closed_lengths}

SCHEMA = {
    "name": str,
    "study": str,                 # the design study's merged JSON (its config sets the mesh)
    "sensitivity": str,           # the sensitivity results, for each optimum's location bound
    "network3": str,              # Network 3's results file, with its proposed splits
    "pairs": list,                # the held-out angle pairs
    "shares": list,
    "workers": int,
    "tolerance": float,           # fixed-point tolerance for s_L
}


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    return document


def network3_splits(path):
    """{(a1, a2, share): split} from the 'New angle pairs' table of Network 3's
    results file."""
    text = pathlib.Path(path).read_text()
    table = text[text.index("### New angle pairs"):]
    out = {}
    for line in table.splitlines():
        m = re.match(r"\| (\d+) / (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            a1, a2, share, _best, network = m.groups()
            out[(float(a1), float(a2), float(share))] = float(network)
    return out


def _heldout_task(task):
    study_config, angles, share, network_split, tolerance = task
    optimum = design_study.optimise((study_config, angles, share))
    mesh = study_config["mesh"]

    def drop(s):
        return design_study.solve(study_config, s, angles, share, mesh["max_area"],
                                  mesh["corner_levels"])["total_pressure_drop"]

    s_l, l1, l2 = S.length_corrected_split(study_config, angles, share, tolerance)
    best = optimum["total_pressure_drop"]
    return {
        "angles": angles, "share": share, "best_split": optimum["best_split"],
        "best_drop": best,
        "murray_split": optimum["friction_law_split"],
        "murray_regret": optimum["friction_law_drop"] / best - 1.0,
        "length_split": s_l, "L1": l1, "L2": l2,
        "length_regret": drop(s_l) / best - 1.0,
        "network_split": network_split,
        "network_regret": drop(network_split) / best - 1.0,
    }


def heldout(config, study_config):
    splits = network3_splits(ROOT / config["network3"])
    tasks = []
    for a1, a2 in config["pairs"]:
        for share in config["shares"]:
            key = (float(a1), float(a2), float(share))
            if key not in splits:
                raise ConfigError(f"no Network 3 split for {key} in {config['network3']}")
            tasks.append((study_config, [float(a1), float(a2)], float(share), splits[key],
                          config["tolerance"]))
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        return list(pool.imap(_heldout_task, tasks))


def friction_drop(study_config, lengths, share, split):
    """The fully developed drop per unit flow, parent included, with the given
    daughter lengths."""
    inertia, nu = pinn.scaling_coefficients(study_config["physics"]["reynolds"],
                                            study_config["physics"]["pressure_scaling"])
    q0, w0 = 4.0 / 3.0, 2.0
    l1, l2 = lengths
    return 12.0 * nu * q0 / w0**3 * (study_config["geometry"]["parent_length"]
                                     + l1 * share**2 / split**3
                                     + l2 * (1 - share) ** 2 / (1 - split) ** 3)


def lengths(config, study, records):
    """Per layout below 180 degrees and per length definition: share of the squared
    departure removed, largest residual, residuals beyond their bound and their
    signs, and min over shares of J*/friction - 1 at the optimum."""
    sc = study["config"]
    bound = {(tuple(r["angles"]), r["share"]): r["mesh"]["bound"] for r in records}
    rows = [r for r in study["rows"] if r["feasible"] and r["share"] is not None
            and sum(r["angles"]) < 180.0]
    outer = sc["geometry"]["daughter_length"]
    out = {}
    for angles in sorted({tuple(r["angles"]) for r in rows}, key=lambda a: (sum(a), a[1] - a[0])):
        mine = [r for r in rows if tuple(r["angles"]) == angles]
        layout = {"departure_squared": sum((r["best_split"] - r["friction_law_split"]) ** 2
                                           for r in mine),
                  "above_friction": {"outer_wall": min(
                      r["total_pressure_drop"] / friction_drop(sc, (outer, outer), r["share"],
                                                               r["best_split"]) - 1.0
                      for r in mine)}}
        for name, rule in DEFINITIONS.items():
            residuals, above = [], []
            for r in mine:
                s_l = S.length_corrected_split(sc, list(angles), r["share"], config["tolerance"],
                                               lengths=rule)[0]
                residuals.append((r["share"], r["best_split"] - s_l,
                                  bound[(angles, r["share"])]))
                g = S.make_geometry(sc, list(angles), r["best_split"])
                above.append(r["total_pressure_drop"]
                             / friction_drop(sc, rule(g), r["share"], r["best_split"]) - 1.0)
            beyond = [x for x in residuals if abs(x[1]) > x[2]]
            layout[name] = {
                "residual_squared": sum(x[1] ** 2 for x in residuals),
                "max_residual": max(abs(x[1]) for x in residuals),
                "beyond_bound": len(beyond),
                "beyond_positive": sum(1 for x in beyond if x[1] > 0),
                "residuals": residuals}
            layout["above_friction"][name] = min(above)
        out[f"{angles[0]:g}/{angles[1]:g}"] = layout
    return out


def _coefficients(study_config, record, lengths):
    """K1, K2 at one optimum against friction along `lengths`: the sensitivity
    run's K (centreline friction) shifted by the friction the new lengths add back."""
    inertia, nu = pinn.scaling_coefficients(study_config["physics"]["reynolds"],
                                            study_config["physics"]["pressure_scaling"])
    q0, w0 = 4.0 / 3.0, 2.0
    share, split = record["share"], record["s_star"]
    g = S.make_geometry(study_config, record["angles"], split)
    centre, new = S.centreline_lengths(g), lengths(g)
    out = []
    for i, (q, w) in enumerate(((share, w0 * split), (1.0 - share, w0 * (1.0 - split)))):
        dynamic = 0.5 * inertia * (q * q0 / w) ** 2
        out.append(record["K_measured"][i]
                   + 12.0 * nu * q * q0 * (centre[i] - new[i]) / w**3 / dynamic)
    return out


def _model_split(study_config, angles, share, k, lengths, xatol):
    """argmin over s of the friction law along `lengths` (taken at each s) plus
    K_i (1/2) U_i^2, within 0.2 of s_M."""
    inertia, _ = pinn.scaling_coefficients(study_config["physics"]["reynolds"],
                                           study_config["physics"]["pressure_scaling"])
    q0, w0 = 4.0 / 3.0, 2.0

    def objective(s):
        try:
            g = S.make_geometry(study_config, angles, s)
        except GeometryError:
            return math.inf
        return (friction_drop(study_config, lengths(g), share, s)
                + inertia * q0**2 / (2.0 * w0**2) * (k[0] * share**3 / s**2
                                                     + k[1] * (1 - share) ** 3 / (1 - s) ** 2))

    s_m = S.murray_split(share)
    lo, hi = S.split_bounds(study_config["geometry"]["min_width"])
    return float(minimize_scalar(objective, bounds=(max(lo, s_m - 0.2), min(hi, s_m + 0.2)),
                                 method="bounded", options={"xatol": xatol}).x)


def junction(config, study, records):
    """Per layout: negative coefficients against centreline and closed friction,
    the closed coefficients at q = 0.5, and how well friction (closed lengths)
    with and without those coefficients locates s*."""
    sc = study["config"]
    out = {}
    for r in records:
        name = f"{r['angles'][0]:g}/{r['angles'][1]:g}"
        k = _coefficients(sc, r, S.closed_lengths)
        v = out.setdefault(name, {"negative_centreline": 0, "negative_closed": 0,
                                  "max_error_lengths": 0.0, "max_error_model": 0.0,
                                  "model_beyond_bound": 0})
        v["negative_centreline"] += sum(1 for x in r["K_measured"] if x < 0)
        v["negative_closed"] += sum(1 for x in k if x < 0)
        if r["share"] == 0.5:
            v["K_closed_q05"] = k
        if sum(r["angles"]) >= 180.0:
            continue
        plain = _model_split(sc, r["angles"], r["share"], (0.0, 0.0), S.closed_lengths,
                             config["tolerance"])
        model = _model_split(sc, r["angles"], r["share"], k, S.closed_lengths,
                             config["tolerance"])
        v["max_error_lengths"] = max(v["max_error_lengths"], abs(r["s_star"] - plain))
        v["max_error_model"] = max(v["max_error_model"], abs(r["s_star"] - model))
        v["model_beyond_bound"] += abs(r["s_star"] - model) > r["mesh"]["bound"]
    return out


def _map_task(task):
    study_config, angles, share, splits = task
    mesh = study_config["mesh"]
    return [design_study.solve(study_config, s, angles, share, mesh["max_area"],
                               mesh["corner_levels"])["total_pressure_drop"] for s in splits]


def regret_map(config, study):
    """J(s_L)/J* - 1 for every optimum below 180 degrees, s_L with centreline and
    with closed lengths."""
    sc = study["config"]
    rows = [r for r in study["rows"] if r["feasible"] and r["share"] is not None
            and sum(r["angles"]) < 180.0]
    names = ("centreline", "closed")
    splits = [[S.length_corrected_split(sc, r["angles"], r["share"], config["tolerance"],
                                        lengths=DEFINITIONS[n])[0] for n in names] for r in rows]
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        drops = list(pool.imap(_map_task, [(sc, r["angles"], r["share"], s)
                                           for r, s in zip(rows, splits)]))
    return [{"angles": r["angles"], "share": r["share"], "best_drop": r["total_pressure_drop"],
             "murray_regret": r["friction_law_drop"] / r["total_pressure_drop"] - 1.0,
             **{f"{n}_split": s for n, s in zip(names, sp)},
             **{f"{n}_regret": d / r["total_pressure_drop"] - 1.0 for n, d in zip(names, dr)}}
            for r, sp, dr in zip(rows, splits, drops)]


def run(config_path, out, parts=PARTS):
    config = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    study = json.loads((ROOT / config["study"]).read_text())
    records = json.loads((ROOT / config["sensitivity"]).read_text())["records"]
    out = pathlib.Path(out)
    path = out.with_suffix(".json")
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["config"] = config
    if "heldout" in parts:
        summary["heldout"] = heldout(config, study["config"])
    if "lengths" in parts:
        summary["lengths"] = lengths(config, study, records)
    if "junction" in parts:
        summary["junction"] = junction(config, study, records)
    if "map" in parts:
        summary["map"] = regret_map(config, study)
    path.write_text(json.dumps(summary, indent=1))
    out.with_suffix(".md").write_text(markdown(summary))
    return summary


def _stats(values):
    return statistics.mean(values), max(values)


def _heldout_markdown(h):
    def pct(x):
        return f"{100 * x:+.3f}%"

    lines = ["### Held-out angle pairs (never seen by Network 3)", "",
             "Regret: J at the proposed split over the finite-element optimum, minus 1, on the "
             "design study's search mesh (its noise is about 1e-4).", "",
             "| angles | share | optimum s* | Murray s_M | regret | length-corrected s_L | "
             "regret | Network 3 | regret |", "|---|---|---|---|---|---|---|---|---|"]
    for r in h:
        lines.append(f"| {r['angles'][0]:g}/{r['angles'][1]:g} | {r['share']:.2f} | "
                     f"{r['best_split']:.4f} | {r['murray_split']:.4f} | {pct(r['murray_regret'])} | "
                     f"{r['length_split']:.4f} | {pct(r['length_regret'])} | "
                     f"{r['network_split']:.4f} | {pct(r['network_regret'])} |")
    lines.append("")
    groups = (("all", lambda r: True),
              ("a daughter beyond 90 degrees", lambda r: max(r["angles"]) > 90.0),
              ("no daughter beyond 90 degrees", lambda r: max(r["angles"]) <= 90.0))
    lines += ["| proposed split | " + " | ".join(f"{g} (mean, max)" for g, _ in groups) + " |",
              "|---|---|---|---|"]
    for name, key in (("Murray s_M", "murray_regret"), ("length-corrected s_L", "length_regret"),
                      ("Network 3", "network_regret")):
        cells = []
        for _, keep in groups:
            mean, worst = _stats([max(r[key], 0.0) for r in h if keep(r)])
            cells.append(f"{100 * mean:.3f}%, {100 * worst:.3f}%")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Network 3's splits are those of its results file, to four decimals (at full "
              "precision its regret was 0.016% mean, 0.077% max). Negative regrets (search noise) "
              "count as 0. Case counts: "
              + ", ".join(f"{g} {sum(1 for r in h if keep(r))}" for g, keep in groups) + "."]
    return lines


def _lengths_markdown(lengths):
    names = list(DEFINITIONS)
    lines = ["### How the length is counted", "",
             "Share of the squared departure from s_M that s_L removes, the largest residual "
             "|s* - s_L|, and the residuals beyond their location bound (how many widen daughter 1). "
             "Centreline = strip area / width = mean of outer and inner wall; closed = the length "
             "over which the daughter has both walls.", "",
             "| layout | " + " | ".join(f"{n}: removed, largest, beyond (+)" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    total = {n: 0.0 for n in names}
    departure = 0.0
    for name, v in lengths.items():
        departure += v["departure_squared"]
        cells = []
        for n in names:
            d = v[n]
            total[n] += d["residual_squared"]
            cells.append(f"{100 * (1 - d['residual_squared'] / v['departure_squared']):.0f}%, "
                         f"{d['max_residual']:.4f}, {d['beyond_bound']}/11 ({d['beyond_positive']})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Pooled over the twelve layouts: "
              + ", ".join(f"{n} {100 * (1 - total[n] / departure):.0f}%" for n in names) + ".", "",
              "### Is the optimum's drop above the friction law?", "",
              "Smallest J*/J_friction - 1 over the shares, with the friction law evaluated at s* "
              "using each length (parent included). Junction and entrance losses are positive, so "
              "a negative value means those lengths over-count the friction.", "",
              "| layout | outer wall (L_d) | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for name, v in lengths.items():
        a = v["above_friction"]
        lines.append(f"| {name} | {100 * a['outer_wall']:+.1f}% | "
                     + " | ".join(f"{100 * a[n]:+.1f}%" for n in names) + " |")
    return lines


def _junction_markdown(junction):
    lines = ["### Loss coefficients against closed lengths", "",
             "K_i = (path drop - parent friction - daughter friction) / (1/2 U_i^2), measured at each "
             "optimum. Model: the friction law along the closed lengths plus K_i (1/2) U_i^2 with "
             "that optimum's own K, minimised over s; 'lengths only' is the same with K = 0.", "",
             "| layout | negative K, centreline friction | negative K, closed friction | "
             "K1, K2 at q = 0.5 (closed) | largest |s* - s|: lengths only | with K | "
             "beyond bound, with K |", "|---|---|---|---|---|---|---|"]
    order = sorted(junction, key=lambda n: (sum(map(float, n.split("/"))),
                                            float(n.split("/")[1]) - float(n.split("/")[0])))
    for name in order:
        v = junction[name]
        k = v.get("K_closed_q05")
        below = sum(map(float, name.split("/"))) < 180.0
        lines.append(f"| {name} | {v['negative_centreline']}/22 | {v['negative_closed']}/22 | "
                     + (f"{k[0]:.2f}, {k[1]:.2f}" if k else "-") + " | "
                     + (f"{v['max_error_lengths']:.4f} | {v['max_error_model']:.4f} | "
                        f"{v['model_beyond_bound']}/11 |" if below else "- | - | - |"))
    return lines


def _map_markdown(rows):
    groups = (("no daughter beyond 90 degrees", lambda r: max(r["angles"]) <= 90.0),
              ("a daughter turned back", lambda r: max(r["angles"]) > 90.0),
              ("all below 180 degrees", lambda r: True))
    lines = ["### Regret over the map, below 180 degrees", "",
             "J at the proposed split over the optimum, minus 1, on the search mesh (noise about "
             "1e-4); median and maximum.", "",
             "| proposed split | " + " | ".join(f"{g} ({sum(1 for r in rows if k(r))})"
                                               for g, k in groups) + " |", "|---|---|---|---|"]
    for name, key in (("Murray s_M", "murray_regret"), ("s_L, centreline", "centreline_regret"),
                      ("s_L, closed", "closed_regret")):
        cells = []
        for _, keep in groups:
            v = [r[key] for r in rows if keep(r)]
            cells.append(f"{100 * statistics.median(v):.4f}%, {100 * max(v):.3f}%")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Largest regret by layout (Murray, centreline s_L, closed s_L): " + "; ".join(
        f"{a}: " + ", ".join(f"{100 * max(r[k] for r in rows if _name(r) == a):.3f}%"
                             for k in ("murray_regret", "centreline_regret", "closed_regret"))
        for a in dict.fromkeys(_name(r) for r in rows))]
    return lines


def _name(r):
    return f"{r['angles'][0]:g}/{r['angles'][1]:g}"


def markdown(summary):
    lines = ["## Baselines for Network 3, and the length behind s_L", ""]
    if "heldout" in summary:
        lines += _heldout_markdown(summary["heldout"]) + [""]
    if "lengths" in summary:
        lines += _lengths_markdown(summary["lengths"]) + [""]
    if "junction" in summary:
        lines += _junction_markdown(summary["junction"]) + [""]
    if "map" in summary:
        lines += _map_markdown(summary["map"])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    args = parser.parse_args(argv)
    run(args.config, args.out, args.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
