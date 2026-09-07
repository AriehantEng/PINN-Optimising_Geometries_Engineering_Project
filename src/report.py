"""Run directory -> accuracy table.

    python -m src.report --run runs/<id>
    python -m src.report --all              # every run under runs/

Reports relative L2 error against the analytic solution as the accuracy
measure, and the loss separately and explicitly labelled as a diagnostic
(CLAUDE.md: "Loss value is a diagnostic, never an accuracy claim").

Each accuracy number is printed next to its SPEC target and a PASS/FAIL, so a
table cannot be read as a success without the reader seeing the target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS_ROOT = Path("runs")

#: SPEC section 5 targets, duplicated here for reporting only. The tests are
#: the gate; this module never decides whether something passed on its own
#: terms, it just displays the same numbers.
TARGETS = {
    "u_relative_l2": ("test 1  u relative L2", 1e-5, "<=", "A"),
    # SPEC test 2 is "Transverse velocity, Case A". Developing flow has a
    # genuine non-zero v*, so applying this target to Case B would report a
    # false failure for correct physics.
    "v_max_abs": ("test 2  max |v*|", 1e-5, "<=", "A"),
    "flux_conservation_relative_error": ("test 3  flux conservation", 1e-6, "<=", None),
    "outlet_profile_relative_l2": ("test 9  outlet profile L2", 1e-4, "<=", "B"),
    "entrance_length_relative_error": ("test 9  entrance length", 0.15, "<=", "B"),
}


def load_metrics(directory: Path) -> dict:
    path = directory / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; has this run finished?")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def format_run(metrics: dict) -> str:
    lines = []
    name = metrics.get("name", "?")
    lines.append(f"=== {name} ===")
    lines.append(
        f"  case {metrics['case']}   Re = {metrics['reynolds']:g}   "
        f"{metrics['formulation']}   {metrics['pressure_scaling']} pressure   "
        f"seed {metrics.get('seed')}"
    )

    lines.append("")
    lines.append("  ACCURACY (relative L2 against the analytic solution)")
    any_target = False
    for key, (label, target, _, only_case) in TARGETS.items():
        if key not in metrics:
            continue
        if only_case is not None and metrics["case"] != only_case:
            continue
        any_target = True
        value = metrics[key]
        verdict = "PASS" if value <= target else "FAIL"
        lines.append(f"    {label:<32} {value:10.3e}   target <= {target:<8.1e} {verdict}")
    if not any_target:
        lines.append("    (no targeted metrics present)")

    if metrics["case"] != "A":
        lines.append(
            f"    {'max |v*| (no target: case B has':<32} {metrics['v_max_abs']:10.3e}"
        )
        lines.append(f"    {'  genuine transverse velocity)':<32}")

    for key, label in (
        ("p_relative_l2", "p relative L2 (gauged at outlet)"),
        ("flux_mean", "mean flux"),
        ("flux_relative_error_vs_inlet", "flux error vs inlet flux"),
        ("outlet_profile_relative_l2_vs_nominal_flux", "outlet profile vs NOMINAL flux"),
        ("inlet_flux_deficit_relative", "inlet flux deficit (corner artefact)"),
        ("entrance_length_measured", "entrance length measured"),
        ("entrance_length_correlation", "entrance length correlation"),
    ):
        if key in metrics:
            lines.append(f"    {label:<32} {metrics[key]:10.3e}")

    lines.append("")
    lines.append("  DIAGNOSTICS -- these are NOT accuracy claims")
    for key, label in (
        ("diag_loss_after_adam", "loss after Adam"),
        ("diag_loss_after_lbfgs", "loss after L-BFGS"),
        ("diag_handoff_relative_gap", "Adam->L-BFGS loss discontinuity"),
        ("residual_momentum_x_max", "max |momentum-x residual|"),
        ("residual_momentum_y_max", "max |momentum-y residual|"),
        ("residual_continuity_max", "max |continuity residual|"),
        ("diag_parameter_count", "parameters"),
        ("diag_lbfgs_iterations", "L-BFGS iterations"),
        ("wall_seconds", "wall time (s)"),
    ):
        if key in metrics:
            value = metrics[key]
            formatted = f"{value:10.3e}" if isinstance(value, float) else f"{value:>10}"
            lines.append(f"    {label:<32} {formatted}")
    if "diag_lbfgs_message" in metrics:
        lines.append(f"    {'L-BFGS termination':<32} {metrics['diag_lbfgs_message']}")
    return "\n".join(lines)


def format_table(all_metrics: list) -> str:
    header = (
        f"{'run':<28}{'case':>5}{'Re':>7}{'form':>18}"
        f"{'u relL2':>11}{'max|v|':>11}{'flux cons':>11}{'loss':>11}"
    )
    lines = [header, "-" * len(header)]
    for metrics in all_metrics:
        loss = metrics.get("diag_loss_after_lbfgs", metrics.get("diag_loss_after_adam"))
        accuracy = metrics.get("u_relative_l2", metrics.get("outlet_profile_relative_l2"))
        lines.append(
            f"{metrics.get('name', '?'):<28}{metrics['case']:>5}"
            f"{metrics['reynolds']:>7g}{metrics['formulation']:>18}"
            f"{accuracy:>11.2e}{metrics['v_max_abs']:>11.2e}"
            f"{metrics['flux_conservation_relative_error']:>11.2e}"
            f"{(loss if loss is not None else float('nan')):>11.2e}"
        )
    lines.append("")
    lines.append(
        "u relL2 is against the analytic solution; for case B it is the outlet "
        "profile against the flux-consistent parabola."
    )
    lines.append("loss is a DIAGNOSTIC and is never an accuracy claim.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="a single run directory, e.g. runs/caseA_re10")
    group.add_argument("--all", action="store_true", help="tabulate every run")
    parser.add_argument("--table", action="store_true", help="one-line-per-run table")
    args = parser.parse_args(argv)

    if args.run:
        metrics = load_metrics(Path(args.run))
        print(format_table([metrics]) if args.table else format_run(metrics))
        return 0

    directories = sorted(d for d in RUNS_ROOT.iterdir() if (d / "metrics.json").is_file())
    if not directories:
        print(f"no finished runs found under {RUNS_ROOT}/")
        return 1
    all_metrics = [load_metrics(d) for d in directories]
    if args.table:
        print(format_table(all_metrics))
    else:
        for metrics in all_metrics:
            print(format_run(metrics))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
