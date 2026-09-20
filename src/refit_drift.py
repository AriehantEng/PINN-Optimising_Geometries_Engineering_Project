"""Recompute the stored drift fit of a finished run from its traced D(n).

`fit_drift` returned an unweighted least-squares fit until the residual
weighting fix. On a problem whose residuals are O(1e-7) the cost at the
initial guess already sits below `curve_fit`'s default `ftol`, so on some
traces the optimiser returned `p0` untouched while reporting convergence, and
the run recorded a verdict drawn from a curve that never fitted anything.

The traced D(n) itself is unaffected -- it is measured, not fitted -- so the
correction needs no retraining: refit the stored `drift_trace.csv` and rewrite
only the `diag_drift_*` keys. Everything else in `metrics.json` stands.

    python -m src.refit_drift --run runs/caseA_re10
    python -m src.refit_drift --all --write

Without `--write` this reports what would change and touches nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from .evaluation.drift import fit_drift, mass_drift

#: Column holding D. Early runs wrote the generic `value` header.
DRIFT_COLUMNS = ("drift", "value")


def read_trace(path):
    """Return (iterations, drifts) from a stored drift trace."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} holds no rows")
    columns = rows[0].keys()
    for candidate in DRIFT_COLUMNS:
        if candidate in columns:
            key = candidate
            break
    else:
        raise ValueError(
            f"{path} has no drift column; looked for {DRIFT_COLUMNS}, got "
            f"{sorted(columns)}"
        )
    return (
        [float(row["iteration"]) for row in rows],
        [float(row[key]) for row in rows],
    )


def refit_flux_metrics(metrics):
    """Recompute the flux metrics from the stored per-station fluxes.

    SPEC v2 S7 redefined D as `max|M(x) - M(0)| / M(0)`, referenced to the
    FIRST station. Runs from before that recorded the spread about the MEAN
    under the same key, which is a different and systematically smaller
    number -- 6.657e-03 against 1.197e-02 on caseB_re10, a factor of 1.8.
    Reading such a value as S7's D understates the drift.

    The per-station fluxes are raw measurements and are unaffected, so the
    current definitions are simply reapplied to them. Returns the updated
    keys, or None when there is nothing stored to recompute from.
    """
    values = metrics.get("flux_values")
    if not values:
        return None
    fluxes = np.asarray(values, dtype=np.float64)
    mean = float(fluxes.mean())
    return {
        "flux_conservation_relative_error": mass_drift(fluxes),
        "flux_spread_about_mean": float(
            np.max(np.abs(fluxes - mean)) / abs(mean)
        ),
    }


def refit(directory, write=False):
    """Refit one run. Returns (before, after) verdict strings."""
    directory = pathlib.Path(directory)
    trace_path = directory / "drift_trace.csv"
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    before = metrics.get("diag_drift_verdict", "(none recorded)")

    # The flux metrics come from the stored per-station fluxes and do NOT
    # depend on the L-BFGS trace, so they are recomputed first. Gating them
    # behind the trace skipped caseB_re10 -- the one run whose stored D was
    # actually wrong -- because that run has no drift_trace.csv at all.
    updated = {}
    flux = refit_flux_metrics(metrics)
    if flux is not None:
        previous = metrics.get("flux_conservation_relative_error")
        if previous is None or abs(
            flux["flux_conservation_relative_error"] - previous
        ) > 0:
            updated.update(flux)
            updated["diag_flux_metrics_recomputed"] = (
                "flux_conservation_relative_error and flux_spread_about_mean "
                "recomputed from the stored flux_values under the SPEC v2 S7 "
                "definition (D referenced to the first station). The value "
                "previously stored predated that definition."
            )

    if not trace_path.exists():
        if updated and write:
            metrics.update(updated)
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
        return (before, "(no drift trace; flux metrics only)", None) if updated else None

    iterations, drifts = read_trace(trace_path)
    if len(iterations) < 5:
        return None

    fit = fit_drift(iterations, drifts)
    updated |= {
        "diag_drift_d_infinity": fit.d_infinity,
        "diag_drift_d_infinity_stderr": fit.d_infinity_stderr,
        "diag_drift_exponent": fit.exponent,
        "diag_drift_fit_points": fit.points,
        "diag_drift_fit_converged": fit.converged,
        "diag_drift_max_relative_residual": fit.max_relative_residual,
        "diag_drift_moved_off_guess": fit.moved_off_guess,
        "diag_drift_verdict": fit.verdict(),
        "diag_drift_refitted": (
            "diag_drift_* recomputed by src.refit_drift after the residual "
            "weighting fix in evaluation.drift; the traced D(n) and every "
            "other metric are from the original run"
        ),
    }
    if write:
        metrics.update(updated)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    return before, updated["diag_drift_verdict"], fit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="a single run directory")
    group.add_argument("--all", action="store_true", help="every run under runs/")
    parser.add_argument(
        "--write", action="store_true",
        help="rewrite metrics.json; without it nothing is touched",
    )
    args = parser.parse_args(argv)

    directories = (
        sorted(pathlib.Path("runs").iterdir()) if args.all
        else [pathlib.Path(args.run)]
    )
    for directory in directories:
        if not directory.is_dir():
            continue
        outcome = refit(directory, write=args.write)
        if outcome is None:
            print(f"{directory.name}: no usable drift trace, skipped")
            continue
        before, after, fit = outcome
        print(f"\n{directory.name}")
        if fit is None:
            print("  flux metrics recomputed from stored flux_values "
                  "(no drift trace in this run)")
        else:
            print(f"  D_inf   {fit.d_infinity:.4e} +- {fit.d_infinity_stderr:.2e}  "
                  f"b={fit.exponent:.4f}  max rel residual "
                  f"{fit.max_relative_residual:.3f}")
        print(f"  before: {before}")
        print(f"  after : {after}")
    if not args.write:
        print("\n(dry run: pass --write to rewrite metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
