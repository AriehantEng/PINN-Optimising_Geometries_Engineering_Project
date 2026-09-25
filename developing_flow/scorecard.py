"""One fixed scorecard for every developing-flow run, so runs can be ranked.

    python -m developing_flow.scorecard --nx 2001 --ny 1001 --reference baseline \\
        --out developing_flow/runs/scorecard \\
        baseline=developing_flow/runs/developing_re10 \\
        baseline=developing_flow/runs/developing_re10_seed1 \\
        baseline=developing_flow/runs/developing_re10_seed2 \\
        rar=developing_flow/runs/developing_re10_rar

Runs sharing a label are seeds of one setup and are summarised as a range.
Every other setup is then compared with the reference setup's SEED RANGE: a
value only counts as better or worse when it lies outside that range, because
a difference smaller than seed-to-seed scatter is not an effect of the change.

The metrics, all lower-is-better, never loss-based except the held-out loss:

    developed_u_error    u vs Poiseuille parabola, relative L2, dense sweep over x* >= 2 L_e
    inlet_u_error        u(0, y) vs the prescribed inlet, relative L2, 401 points
    wiggle_x4, wiggle_x10  largest |u - parabola| across the channel at x* = 4 and 10
    flux_error           largest |flux / (4/3) - 1| over 201 stations
    rms_x, rms_y, rms_c  RMS Navier-Stokes residuals, dense sweep
    max_residual         largest residual of any equation, dense sweep
    heldout_loss         loss on the held-out points (a diagnostic, not accuracy)
    minutes              training time
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from developing_flow import pinn, verify

METRICS = (
    ("developed_u_error", "u error, developed region (rel. L2)"),
    ("inlet_u_error", "u error at the inlet (rel. L2)"),
    ("wiggle_x4", "wiggle at x* = 4 (max abs error)"),
    ("wiggle_x10", "wiggle at x* = 10 (max abs error)"),
    ("flux_error", "worst flux error (relative)"),
    ("rms_x", "RMS x-momentum residual"),
    ("rms_y", "RMS y-momentum residual"),
    ("rms_c", "RMS continuity residual"),
    ("max_residual", "largest residual, any equation"),
    ("heldout_loss", "held-out loss"),
    ("minutes", "training time (min)"),
)


def score(directory: pathlib.Path, nx: int, ny: int) -> dict:
    sweep_path = directory / "verify.json"
    sweep = json.loads(sweep_path.read_text()) if sweep_path.is_file() else None
    if sweep is None or sweep.get("grid") != [nx, ny]:
        sweep = verify.sweep(directory, nx, ny, verbose=False)
    metrics = json.loads((directory / "metrics.json").read_text())
    model = pinn.load_run(directory)
    config = model.config
    y = np.linspace(-1.0, 1.0, 401)
    u_in = model(np.zeros_like(y), y)[0]
    inlet = pinn.inlet_profile(y, config.inlet_sharpness)
    parabola = pinn.developed_profile(y, pinn.INLET_FLUX)

    def wiggle(x):
        return float(np.max(np.abs(model(np.full_like(y, x), y)[0] - parabola)))

    return {
        "developed_u_error": sweep["developed_u_relative_l2"],
        "inlet_u_error": float(np.linalg.norm(u_in - inlet) / np.linalg.norm(inlet)),
        "wiggle_x4": wiggle(4.0),
        "wiggle_x10": wiggle(config.length),
        "flux_error": sweep["flux_max_relative_error"],
        "rms_x": sweep["residual_momentum_x"]["rms"],
        "rms_y": sweep["residual_momentum_y"]["rms"],
        "rms_c": sweep["residual_continuity"]["rms"],
        "max_residual": max(sweep[k]["max"] for k in (
            "residual_momentum_x", "residual_momentum_y", "residual_continuity")),
        "heldout_loss": metrics["loss_heldout"],
        "minutes": metrics["info_total_seconds"] / 60.0,
    }


def summarise(groups: dict, reference: str):
    """Mean and range per group, and each group's position against the reference range."""
    summary = {}
    for label, rows in groups.items():
        summary[label] = {}
        for key, _ in METRICS:
            values = np.array([r[key] for r in rows])
            summary[label][key] = {"mean": float(values.mean()), "min": float(values.min()),
                                   "max": float(values.max()), "n": len(values)}
    verdicts = {}
    ref = summary.get(reference)
    for label, stats in summary.items():
        if label == reference or ref is None:
            continue
        verdicts[label] = {}
        for key, _ in METRICS:
            if key == "minutes":
                continue
            lo, hi = ref[key]["min"], ref[key]["max"]
            value = stats[key]["mean"]
            verdicts[label][key] = ("better" if value < lo else "worse" if value > hi
                                    else "within seed range")
    return summary, verdicts


def format_markdown(runs, summary, verdicts, reference):
    lines = ["## Every run", "",
             "| Setup | Run | " + " | ".join(name for _, name in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for label, directory, row in runs:
        cells = [f"{row[k]:.1f}" if k == "minutes" else f"{row[k]:.2e}" for k, _ in METRICS]
        lines.append(f"| {label} | {directory.name} | " + " | ".join(cells) + " |")
    lines += ["", "## Each setup: mean, with the range over its seeds", "",
              "| Setup | Seeds | " + " | ".join(name for _, name in METRICS) + " |",
              "|---|---|" + "---|" * len(METRICS)]
    for label, stats in summary.items():
        n = next(iter(stats.values()))["n"]
        cells = []
        for key, _ in METRICS:
            s = stats[key]
            fmt = (lambda v: f"{v:.1f}") if key == "minutes" else (lambda v: f"{v:.2e}")
            cells.append(fmt(s["mean"]) if n == 1 else
                         f"{fmt(s['mean'])} ({fmt(s['min'])} to {fmt(s['max'])})")
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    if verdicts:
        keys = [(k, n) for k, n in METRICS if k != "minutes"]
        lines += ["", f"## Against the {reference} seed range", "",
                  "A setup counts as better or worse on a metric only if its mean lies outside "
                  f"the range spanned by the {reference} seeds.", "",
                  "| Setup | " + " | ".join(n for _, n in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for label, v in verdicts.items():
            lines.append(f"| {label} | " + " | ".join(v[k] for k, _ in keys) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="label=path, one per run")
    parser.add_argument("--reference", required=True, help="label of the reference setup")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    runs, groups = [], {}
    for item in args.runs:
        label, _, path = item.partition("=")
        if not path:
            parser.error(f"expected label=path, got {item!r}")
        directory = pathlib.Path(path)
        row = score(directory, args.nx, args.ny)
        runs.append((label, directory, row))
        groups.setdefault(label, []).append(row)
    if args.reference not in groups:
        parser.error(f"no run is labelled {args.reference!r}")

    summary, verdicts = summarise(groups, args.reference)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    markdown = format_markdown(runs, summary, verdicts, args.reference)
    (out / "scorecard.md").write_text(markdown, encoding="utf-8")
    with (out / "scorecard.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": [{"label": l, "run": str(d), **r} for l, d, r in runs],
                   "summary": summary, "verdicts": verdicts, "reference": args.reference,
                   "grid": [args.nx, args.ny]}, handle, indent=2)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
