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

#: Below this, D sits at the rounding floor: the formulation conserves mass by
#: construction, so test 3 is VACUOUS rather than passed. It is not a failure
#: either -- the test simply provides no evidence about that run.
VACUOUS_DRIFT_FLOOR = 1e-12


def gate_results(metrics):
    """Return (failures, vacuous) among the SPEC targets applicable to a run.

    Each entry is (label, value, target). A run whose `failures` is non-empty
    has missed a SPEC section 5 target and must not exit zero: the targets are
    the whole point of the benchmark, and a green exit code on a printed FAIL
    means no CI job, queue script or reviewer can ever catch a regression.
    """
    failures, vacuous = [], []
    for key, (label, target, _, only_case) in TARGETS.items():
        if key not in metrics:
            continue
        if only_case is not None and metrics.get("case") != only_case:
            continue
        value = metrics[key]
        if (key == "flux_conservation_relative_error"
                and value < VACUOUS_DRIFT_FLOOR):
            vacuous.append((label, value, target))
        elif value > target:
            failures.append((label, value, target))
    return failures, vacuous


def format_gate(failures, vacuous) -> str:
    """One block naming every vacuous and failed target, or a clean line."""
    lines = []
    for label, value, _ in vacuous:
        lines.append(
            f"  {label}: D = {value:.3e} is at the rounding floor -- this "
            "formulation conserves mass by construction, so the test is "
            "VACUOUS here, not passed"
        )
    if failures:
        lines.append("GATE FAILED -- SPEC section 5 targets not met:")
        for label, value, target in failures:
            lines.append(f"  {label}: {value:.4e} exceeds the {target:.0e} target")
    return "\n".join(lines)


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


def _write_falsifiability_matrix() -> int:
    """Regenerate the S5 matrix. Must be re-run on any ansatz change."""
    import yaml

    from .config import parse_config
    from .evaluation.falsifiability import (
        all_vacuous_rows, build_matrix, degenerate_targets, format_matrix,
    )

    columns = {
        "vp-soft": "experiments/caseA_re10_baseline.yaml",
        "vp-hard": "experiments/caseA_re10.yaml",
        "psi-cubic": "experiments/caseA_re10_psiomega_cubic.yaml",
        "psi-sin": "experiments/caseA_re10_psiomega_sinusoidal.yaml",
    }
    configs = {
        label: parse_config(yaml.safe_load(open(path, encoding="utf-8")))
        for label, path in columns.items()
    }
    matrix = build_matrix(configs)
    table = format_matrix(matrix, list(columns))

    document = [
        "# Falsifiability matrix (SPEC v2, S5)",
        "",
        "Generated by `python -m src.report --falsifiability`.",
        "**Regenerate on every ansatz change** -- a stale matrix is the silent",
        "disarming this exists to prevent, and `tests/test_sentinels.py` fails",
        "if this file does not match the code.",
        "",
        "A check is classified by running it on an UNTRAINED, randomly",
        "initialised network at five seeds:",
        "",
        "* `LIVE` - fails at every seed, so the check bites.",
        "* `WEAKENED` - passes at some seeds, so it partly measures the ansatz.",
        "* `VACUOUS` - passes at every seed, so it measures the ansatz and not",
        "  the trained solution.",
        "",
        "Columns: `vp-*` velocity-pressure with the soft/hard pressure gauge;",
        "`psi-*` the psi-omega system under the cubic/sinusoidal lift.",
        "",
        table,
        "",
    ]
    degenerate = {label: degenerate_targets(cfg) for label, cfg in configs.items()}
    document += [
        "## Degenerate targets",
        "",
        "A check can be `LIVE` -- an untrained network fails it -- and still be",
        "nearly free, because the ansatz places the correct answer at a single",
        "point the optimiser can reach exactly. Sampling at the initialisation",
        "distribution cannot see that, so this probes it directly by zeroing the",
        "output layer, i.e. removing the network's contribution entirely.",
        "",
        "Checks that PASS with no network contribution at all:",
        "",
    ]
    for label in columns:
        names = degenerate[label]
        document.append(f"* `{label}`: " + (", ".join(names) if names else "none"))
    document += [
        "",
        "Read `psi-cubic` carefully: essentially the whole Case A suite passes",
        "with the network removed, because the cubic lift's derivative IS the",
        "Case A solution. Its accuracy must never be quoted as a result.",
        "",
        "`test 2  max |v*|` is degenerate in EVERY column: the zero field passes",
        "it. It cannot distinguish a correct solution on its own and is only",
        "meaningful conjoined with test 1.",
        "",
    ]

    vacuous = all_vacuous_rows(matrix)
    if vacuous:
        document += [
            "## Untested properties",
            "",
            "These are `VACUOUS` in every column, so the suite does not test",
            "them at all:",
            "",
        ] + [f"* {name}" for name in vacuous] + [""]
    Path("docs/falsifiability-matrix.md").write_text(
        "\n".join(document), encoding="utf-8"
    )
    print("\n".join(document))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="a single run directory, e.g. runs/caseA_re10")
    group.add_argument("--all", action="store_true", help="tabulate every run")
    group.add_argument(
        "--falsifiability", action="store_true",
        help="regenerate docs/falsifiability-matrix.md (SPEC v2 S5)",
    )
    parser.add_argument("--table", action="store_true", help="one-line-per-run table")
    args = parser.parse_args(argv)

    if args.falsifiability:
        return _write_falsifiability_matrix()

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
