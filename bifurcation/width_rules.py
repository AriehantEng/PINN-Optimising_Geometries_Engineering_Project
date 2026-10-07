"""Must the daughters share the parent's width? Width rules against finite elements.

The design study fixes w1 + w2 = w0 and asks for the best split. This asks the
question behind that constraint: with the angles and the flow share fixed, how
do the classical width rules compare, and how wide should the daughters be in
total?

    equal            w1 = w2 = w0 / 2                        (sum w0)
    shared_murray    w_i = w0 sqrt(q_i) / (sqrt(q) + sqrt(1 - q))   (sum w0)
    shared_optimum   the finite-element optimum of the design study (sum w0)
    square           w_i = w0 sqrt(q_i): w1^2 + w2^2 = w0^2, the planar
                     Murray law (minimum pumping power plus volume cost)
    cube             w_i = w0 q_i^(1/3): w1^3 + w2^3 = w0^3, Murray's
                     round-pipe law applied to widths

and a scan of the total width at the friction-law proportions:
w_i = k * (shared_murray widths), so w1 + w2 = k w0.

Pressure drop alone has no optimum in k: a wider channel always loses less.
What sets the width is what it costs. Murray's cost is pumping power plus a
price b per unit of channel area (volume, per unit depth),

    C = Q0 * J + b * A,   b = 36 nu Q0^2 / w0^4,

the price at which the PARENT's width w0 is itself optimal for its flow
(d/dw of 12 nu Q0^2 / w^3 + b w = 0 at w0). Under it the friction law puts the
optimum at k = sqrt(q) + sqrt(1 - q) (the square law); the finite-element
scan shows where the junction moves it. Every geometry the rules refuse is
recorded with its reason.

    python -m bifurcation.width_rules --config bifurcation/width_rules_re10.yaml \\
        --out docs/bifurcation_re10-width-rules
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import pathlib
import time

import yaml

from bifurcation import reference
from bifurcation.geometry import PARENT_WIDTH, Geometry, GeometryError
from developing_flow import pinn

ROOT = pathlib.Path(__file__).resolve().parent.parent
ConfigError = pinn.ConfigError

RULES = ("equal", "shared_murray", "shared_optimum", "square", "cube")

SCHEMA = {
    "name": str,
    "physics": {"reynolds": float, "pressure_scaling": str},
    "geometry": {"parent_length": float, "daughter_length": float, "min_width": float},
    "data": str,                  # the design study whose optima are the shared_optimum rule
    "angles": list,
    "shares": list,
    "rules": list,
    "scan": {"angles": list, "share": float, "scales": list},
    "mesh": {"max_area": float, "corner_levels": int, "corner_radius": float},
    "workers": int,
}


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    for rule in document["rules"]:
        if rule not in RULES:
            raise ConfigError(f"unknown rule {rule!r}; rules are {list(RULES)}")
    for share in document["shares"] + [document["scan"]["share"]]:
        if not 0.0 < share < 1.0:
            raise ConfigError(f"every share must lie in (0, 1), got {share}")
    if any(k <= 0.0 for k in document["scan"]["scales"]):
        raise ConfigError("scan.scales must be positive")
    return document


def rule_widths(rule, share, optimum_split=None):
    """(w1, w2) of a width rule for daughter 1's share q, parent width w0 = 2."""
    w0, q = PARENT_WIDTH, share
    if rule == "equal":
        return 0.5 * w0, 0.5 * w0
    if rule == "shared_murray":
        s = math.sqrt(q) / (math.sqrt(q) + math.sqrt(1.0 - q))
        return w0 * s, w0 * (1.0 - s)
    if rule == "shared_optimum":
        if optimum_split is None:
            raise ConfigError("shared_optimum needs the design study's optimum for this case")
        return w0 * optimum_split, w0 * (1.0 - optimum_split)
    if rule == "square":
        return w0 * math.sqrt(q), w0 * math.sqrt(1.0 - q)
    if rule == "cube":
        return w0 * q ** (1.0 / 3.0), w0 * (1.0 - q) ** (1.0 / 3.0)
    raise ConfigError(f"unknown rule {rule!r}")


def murray_price(reynolds, pressure_scaling):
    """b = 36 nu Q0^2 / w0^4: the area price at which the parent's width is optimal."""
    _, nu = pinn.scaling_coefficients(reynolds, pressure_scaling)
    q0 = 4.0 / 3.0                                       # the parabolic inflow
    return 36.0 * nu * q0**2 / PARENT_WIDTH**4


def load_optima(config):
    """{(a1, a2, share): (split, fine-mesh drop)} from the design study, which
    must have been run on this geometry and this mesh."""
    study = json.loads((ROOT / config["data"]).read_text())
    if study["config"]["geometry"] != config["geometry"]:
        raise ConfigError("the design study's geometry differs from this config's")
    if study["config"]["physics"] != config["physics"]:
        raise ConfigError("the design study's physics differs from this config's")
    check = study["config"]["check"]
    mesh = config["mesh"]
    if (check["max_area"], check["corner_levels"]) != (mesh["max_area"], mesh["corner_levels"]) \
            or study["config"]["mesh"]["corner_radius"] != mesh["corner_radius"]:
        raise ConfigError("this config's mesh must be the design study's check mesh, so "
                          "shared_optimum reproduces its fine-mesh drop")
    return {(r["angles"][0], r["angles"][1], r["share"]):
            (r["best_split"], r["total_pressure_drop_fine"])
            for r in study["rows"] if r["feasible"] and r["share"] is not None}


def evaluate(task):
    """One geometry: (config, label dict, w1, w2, share) -> its row."""
    config, label, w1, w2, share = task
    g, m = config["geometry"], config["mesh"]
    row = {**label, "w1": w1, "w2": w2, "width_sum": (w1 + w2) / PARENT_WIDTH, "share": share}
    try:
        geometry = Geometry(w1, w2, label["angles"][0], label["angles"][1], g["parent_length"],
                            g["daughter_length"], g["min_width"])
    except GeometryError as error:
        return {**row, "refused": str(error)}
    start = time.perf_counter()
    solution = reference.solve_bifurcation(
        geometry, config["physics"]["reynolds"], config["physics"]["pressure_scaling"],
        reference.parabola_inlet, share, m["max_area"], m["corner_levels"], m["corner_radius"])
    measures = solution.measures()
    area = geometry.area()
    b = murray_price(config["physics"]["reynolds"], config["physics"]["pressure_scaling"])
    q0 = measures["flux_in"]
    return {**row, "refused": None, "drop": measures["total_pressure_drop"],
            "delivered_share": measures["split_flux"],
            "energy_consistency": measures["energy_consistency"],
            "area": area, "murray_cost": q0 * measures["total_pressure_drop"] + b * area,
            "seconds": time.perf_counter() - start}


def tasks(config):
    optima = load_optima(config) if "shared_optimum" in config["rules"] else {}
    out = []
    for a1, a2 in config["angles"]:
        for share in config["shares"]:
            for rule in config["rules"]:
                optimum = optima.get((float(a1), float(a2), share), (None, None))[0]
                if rule == "shared_optimum" and optimum is None:
                    raise ConfigError(f"no design-study optimum for {a1}/{a2} share {share}")
                w1, w2 = rule_widths(rule, share, optimum)
                out.append((config, {"kind": "rule", "rule": rule, "angles": [a1, a2]},
                            w1, w2, share))
    share = config["scan"]["share"]
    base = rule_widths("shared_murray", share)
    for a1, a2 in config["scan"]["angles"]:
        for k in config["scan"]["scales"]:
            out.append((config, {"kind": "scan", "scale": k, "angles": [a1, a2]},
                        k * base[0], k * base[1], share))
    return out, optima


def run(config_path, out):
    config = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    work, optima = tasks(config)
    start = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        rows = list(pool.imap(evaluate, work))
    # shared_optimum re-solves a design-study geometry on its own check mesh: the
    # drop must be the study's to round-off, or the two are not measuring alike
    for row in rows:
        if row.get("rule") == "shared_optimum" and row["refused"] is None:
            a1, a2 = row["angles"]
            expected = optima[(float(a1), float(a2), row["share"])][1]
            row["study_drop"] = expected
            if abs(row["drop"] / expected - 1.0) > 1e-9:
                raise RuntimeError(f"{a1}/{a2} share {row['share']}: drop {row['drop']} does "
                                   f"not reproduce the design study's {expected}")
    summary = {"config": config, "murray_price": murray_price(
        config["physics"]["reynolds"], config["physics"]["pressure_scaling"]),
        "rows": rows, "seconds": time.perf_counter() - start}
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    out.with_suffix(".md").write_text(markdown(summary))
    return summary


def markdown(summary):
    config, rows = summary["config"], summary["rows"]
    lines = [f"## Width rules against finite elements, Re_h = {config['physics']['reynolds']:g}", "",
             f"Mesh max area {config['mesh']['max_area']}, {config['mesh']['corner_levels']} corner "
             f"refinements; parent {config['geometry']['parent_length']:g} long, daughters "
             f"{config['geometry']['daughter_length']:g} long; w0 = 2. 'sum' is (w1 + w2) / w0. "
             f"'drop' is J, the total-pressure drop per unit flow. Murray cost C = Q0 J + b A with "
             f"b = {summary['murray_price']:.4g}, the area price at which the parent's own width is "
             "optimal; A is the channel area.", "",
             "### Rules", "",
             "| angles | share | rule | w1, w2 | sum | drop J | vs shared_optimum | area A | "
             "Murray cost C | vs shared_optimum |", "|---|---|---|---|---|---|---|---|---|---|"]
    rule_rows = [r for r in rows if r["kind"] == "rule"]
    for r in rule_rows:
        a1, a2 = r["angles"]
        ref = next((x for x in rule_rows if x["angles"] == r["angles"] and x["share"] == r["share"]
                    and x["rule"] == "shared_optimum" and x["refused"] is None), None)
        if r["refused"]:
            lines.append(f"| {a1:g}/{a2:g} | {r['share']:g} | {r['rule']} | {r['w1']:.3f}, "
                         f"{r['w2']:.3f} | {r['width_sum']:.3f} | refused: {r['refused']} | | | | |")
            continue
        def rel(key, r=r, ref=ref):
            return "" if ref is None else f"{100 * (r[key] / ref[key] - 1):+.1f}%"

        lines.append(f"| {a1:g}/{a2:g} | {r['share']:g} | {r['rule']} | {r['w1']:.3f}, {r['w2']:.3f} "
                     f"| {r['width_sum']:.3f} | {r['drop']:.4f} | {rel('drop')} | {r['area']:.3f} | "
                     f"{r['murray_cost']:.4f} | {rel('murray_cost')} |")
    lines += ["", f"### Total width scan at share {config['scan']['share']:g} (friction-law "
              "proportions, w1 + w2 = k w0)", "",
              "| angles | k | drop J | area A | Murray cost C |", "|---|---|---|---|---|"]
    for r in [r for r in rows if r["kind"] == "scan"]:
        a1, a2 = r["angles"]
        if r["refused"]:
            lines.append(f"| {a1:g}/{a2:g} | {r['scale']:g} | refused: {r['refused']} | | |")
        else:
            lines.append(f"| {a1:g}/{a2:g} | {r['scale']:g} | {r['drop']:.4f} | {r['area']:.3f} | "
                         f"{r['murray_cost']:.4f} |")
    lines.append("")
    lines.append(f"{len(rows)} geometries, {summary['seconds'] / 60:.1f} min on "
                 f"{config['workers']} processes.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="path stem: writes .md and .json")
    args = parser.parse_args(argv)
    run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
