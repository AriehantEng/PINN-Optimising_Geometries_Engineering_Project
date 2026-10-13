"""Three checks on the cost benchmark (bifurcation/cost.py), for the paper.

    python -m bifurcation.cost_checks --config bifurcation/cost_checks_re10.yaml \\
        --out docs/bifurcation_re10-cost-checks

    accuracy    the coarse, reference and finest meshes at each listed split, so
                the equal-accuracy baseline (the coarse mesh) is checked at the
                edges of the PINN's range and not only at s = 0.5. The coarse
                mesh's error is taken against the finest mesh, as in the cost
                table; the PINNs' errors are their stored ones, against the
                reference mesh, and the reference drop recomputed here must
                equal the one they were scored against.
    curvature   J''/J at the run's finite-element optimum, by central second
                differences on the reference mesh at the listed steps and their
                Richardson extrapolation, and the regret it predicts for the
                run's argmin error, (J''/J) delta^2 / 2, beside the measured one.
    memory      the peak memory of one Levenberg-Marquardt step, each run
                measured in several separate processes, since one process's
                ru_maxrss varies from one to the next.

Every measurement runs in its own subprocess (cost.child), one at a time.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import numpy as np
import yaml

from bifurcation import cost
from developing_flow import pinn

ROOT = cost.ROOT
ConfigError = pinn.ConfigError
MESH = {"max_area": float, "corner_levels": int}
SCHEMA = {
    "name": str,
    "run": str,                   # the layout, and the optimum whose curvature is measured
    "meshes": {"coarse": MESH, "reference": MESH, "finest": MESH},
    "accuracy_splits": list,
    "curvature_steps": list,      # h for (J(s*+h) - 2J(s*) + J(s*-h)) / h^2, smallest first
    "memory_runs": list,          # [{label, run}]
    "memory_processes": int,
    "memory_repeats": int,        # LM steps timed in each process after its warm-up
}


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    steps = document["curvature_steps"]
    if len(steps) != 2 or not 0 < steps[0] < steps[1]:
        raise ConfigError("curvature_steps must be two increasing positive steps")
    if not document["accuracy_splits"]:
        raise ConfigError("accuracy_splits must list at least one split")
    for k, entry in enumerate(document["memory_runs"]):
        if not isinstance(entry, dict) or set(entry) != {"label", "run"}:
            raise ConfigError(f"memory_runs[{k}] must be a mapping with exactly 'label' and 'run'")
    if document["memory_processes"] < 1 or document["memory_repeats"] < 1:
        raise ConfigError("memory_processes and memory_repeats must be at least 1")
    return document


def _drop(document, mesh, split):
    m = document["meshes"][mesh]
    return cost.child("measure_fe", {"run_dir": document["run"], "split": split, **m})["drop"]


def richardson(second_differences, steps):
    """Second-order central differences D(h) have error c h^2 + O(h^4)."""
    (d1, d2), (h1, h2) = second_differences, steps
    ratio = (h2 / h1) ** 2
    return (ratio * d1 - d2) / (ratio - 1.0)


def accuracy(document, verbose=True):
    metrics = json.loads((ROOT / document["run"] / "metrics.json").read_text())
    stored = {round(r["split"], 6): r for r in metrics["splits"]}
    runs = {e["label"]: json.loads((ROOT / e["run"] / "metrics.json").read_text())
            for e in document["memory_runs"]}
    rows = []
    for split in document["accuracy_splits"]:
        drops = {mesh: _drop(document, mesh, split) for mesh in ("coarse", "reference", "finest")}
        row = {"split": split, "drops": drops,
               "coarse_vs_finest": drops["coarse"] / drops["finest"] - 1.0,
               "coarse_vs_reference": drops["coarse"] / drops["reference"] - 1.0,
               "reference_vs_finest": drops["reference"] / drops["finest"] - 1.0,
               "pinn_vs_reference": {}}
        for label, m in runs.items():
            match = [r for r in m["splits"] if abs(r["split"] - split) < 1e-9]
            if match:
                row["pinn_vs_reference"][label] = match[0]["drop_error"]
        if round(split, 6) in stored:
            row["stored_reference_drop"] = stored[round(split, 6)]["reference_drop"]
        rows.append(row)
        if verbose:
            print("  accuracy", row, flush=True)
    return rows


def curvature(document, verbose=True):
    o = json.loads((ROOT / document["run"] / "metrics.json").read_text())["optimum"]
    centre, steps = o["reference_split"], document["curvature_steps"]
    j0 = _drop(document, "reference", centre)
    seconds = []
    for h in steps:
        left, right = (_drop(document, "reference", centre + sign * h) for sign in (-1.0, 1.0))
        seconds.append({"h": h, "left": left, "right": right,
                        "second_difference": (left - 2.0 * j0 + right) / h**2})
    extrapolated = richardson([d["second_difference"] for d in seconds], steps)
    ratio = extrapolated / j0
    delta = o["argmin_error"]
    result = {"centre": centre, "drop": j0, "stored_min_drop": o["reference_min_drop"],
              "differences": seconds, "second_derivative": extrapolated, "ratio": ratio,
              "ratio_by_step": [d["second_difference"] / j0 for d in seconds],
              "argmin_error": delta, "predicted_regret": 0.5 * ratio * delta**2,
              "measured_regret": o["regret"]}
    if verbose:
        print("  curvature", result, flush=True)
    return result


def memory(document, verbose=True):
    rows = []
    for entry in document["memory_runs"]:
        peaks, seconds = [], []
        for _ in range(document["memory_processes"]):
            r = cost.child("measure_lm_step", {"run_dir": entry["run"], "width": None,
                                               "repeats": document["memory_repeats"]})
            peaks.append(r["peak_mb"])
            seconds.append(r["seconds"])
        rows.append({"label": entry["label"], "run": entry["run"], "parameters": r["parameters"],
                     "rows": r["rows"], "peak_mb": peaks, "seconds": seconds})
        if verbose:
            print("  memory", rows[-1], flush=True)
    return rows


def markdown(result):
    c = result["config"]
    m = c["meshes"]
    name = {k: f"{v['max_area']:g}/{v['corner_levels']}" for k, v in m.items()}
    lines = ["## Checks on the cost benchmark", "",
             f"Layout of `{c['run']}`; finite elements as in `bifurcation/cost.py`.", "",
             "### The equal-accuracy baseline across the range", "",
             f"Coarse mesh {name['coarse']}, reference {name['reference']}, finest "
             f"{name['finest']}. Finite-element errors are against the finest mesh; the PINNs' "
             "errors are their stored ones, against the reference mesh.", "",
             "| split | J coarse | J reference | J finest | coarse vs finest | coarse vs reference "
             "| reference vs finest | " + " | ".join(f"PINN {k} vs reference"
                                                    for k in result["accuracy"][0]["pinn_vs_reference"])
             + " |", "|" + "---|" * (7 + len(result["accuracy"][0]["pinn_vs_reference"]))]
    for r in result["accuracy"]:
        d = r["drops"]
        lines.append(f"| {r['split']:g} | {d['coarse']:.6f} | {d['reference']:.6f} | "
                     f"{d['finest']:.6f} | {100 * r['coarse_vs_finest']:+.4f}% | "
                     f"{100 * r['coarse_vs_reference']:+.4f}% | {100 * r['reference_vs_finest']:+.4f}% | "
                     + " | ".join(f"{100 * e:+.3f}%" for e in r["pinn_vs_reference"].values()) + " |")
    stored = [r for r in result["accuracy"] if "stored_reference_drop" in r]
    if stored:
        worst = max(abs(r["drops"]["reference"] / r["stored_reference_drop"] - 1.0) for r in stored)
        lines += ["", f"The reference drop recomputed here matches the one the PINNs were scored "
                      f"against to {worst:.1e} relative."]
    k = result["curvature"]
    lines += ["", "### Curvature of the objective at the finite-element optimum", "",
              f"s* = {k['centre']:.5f}, J* = {k['drop']:.6f} (reference mesh).", "",
              "| h | J(s* - h) | J(s* + h) | (J(s*+h) - 2J* + J(s*-h)) / h² / J* |", "|---|---|---|---|"]
    for d, ratio in zip(k["differences"], k["ratio_by_step"]):
        lines.append(f"| {d['h']:g} | {d['left']:.6f} | {d['right']:.6f} | {ratio:.2f} |")
    lines += ["", f"Richardson extrapolation: **J''/J = {k['ratio']:.2f}**. For the run's argmin "
                  f"error δ = {k['argmin_error']:+.5f}, (J''/J) δ²/2 predicts a regret of "
                  f"{100 * k['predicted_regret']:.4f}%; measured {100 * k['measured_regret']:.4f}%."]
    lines += ["", "### Peak memory of one Levenberg-Marquardt step", "",
              f"{c['memory_processes']} separate processes per run, {c['memory_repeats']} timed "
              "steps each after a warm-up.", "",
              "| run | parameters | rows | peak memory per process (MB) | median (MB) | step (s) |",
              "|---|---|---|---|---|---|"]
    for r in result["memory"]:
        lines.append(f"| {r['label']} | {r['parameters']} | {r['rows']} | "
                     f"{', '.join(f'{p:.0f}' for p in r['peak_mb'])} | {np.median(r['peak_mb']):.0f} | "
                     f"{np.median(r['seconds']):.1f} |")
    return "\n".join(lines) + "\n"


def run(config_path, out):
    document = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    result = {"config": document, "accuracy": accuracy(document),
              "curvature": curvature(document), "memory": memory(document)}
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
