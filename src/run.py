"""Config -> experiment.

    python -m src.run --config experiments/<name>.yaml

Writes everything a run produced into runs/<name>/. That directory is written
by the code and never edited by hand (CLAUDE.md, "Module layout").
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax.flatten_util import ravel_pytree

from .config import Config, load_config
from .evaluation.accuracy import evaluate
from .evaluation import plots
from .models.build import build_model
from .training.loop import train

RUNS_ROOT = Path("runs")


def run_directory(config: Config, override: str | None) -> Path:
    return Path(override) if override else RUNS_ROOT / config.name


def save_parameters(path: Path, params) -> None:
    """Store the flattened parameter vector.

    The pytree structure is fully determined by the config, so `load_parameters`
    rebuilds it rather than pickling it. That keeps run directories readable and
    version-independent.
    """
    flat, _ = ravel_pytree(params)
    np.savez_compressed(path, flat=np.asarray(flat, dtype=np.float64))


def load_parameters(path: Path, config: Config):
    template, model = build_model(config, jax.random.PRNGKey(config.seed))
    _, unflatten = ravel_pytree(template)
    flat = np.load(path)["flat"]
    return unflatten(flat), model


def write_history(path: Path, rows: list) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def execute(config_path: str, out: str | None = None, verbose: bool = True) -> Path:
    config = load_config(config_path)
    directory = run_directory(config, out)
    directory.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[{config.name}] case {config.case}, Re = {config.physics.reynolds}, "
              f"{config.physics.formulation}, {config.physics.pressure_scaling} pressure")
        print(f"[{config.name}] active loss terms: {', '.join(config.loss_terms)}")

    started = time.time()
    result = train(config, verbose=verbose)
    metrics = evaluate(config, result.model, result.params)
    metrics["wall_seconds"] = time.time() - started
    metrics.update({f"diag_{k}": v for k, v in result.diagnostics.items()})
    metrics["name"] = config.name
    metrics["seed"] = config.seed

    shutil.copyfile(config_path, directory / "config.yaml")
    save_parameters(directory / "params.npz", result.params)
    write_history(directory / "history.csv", result.history)
    # T-S7.2: the D(n) trace is written for every run.
    write_history(directory / "drift_trace.csv", result.drift_trace)
    with (directory / "lbfgs.json").open("w", encoding="utf-8") as handle:
        json.dump(result.lbfgs_history, handle, indent=2)
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plots.write_all(config, result.model, result.params, directory)
    with (directory / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "jax": jax.__version__,
                "jax_devices": [str(d) for d in jax.devices()],
                "x64_enabled": bool(jax.config.jax_enable_x64),
            },
            handle,
            indent=2,
        )

    if verbose:
        print(f"[{config.name}] wrote {directory}")
        _print_summary(metrics)
    return directory


def _show(metrics: dict, key: str, label: str, suffix: str = "") -> None:
    """Print one metric if present. Absence is normal, not an error.

    Which metrics exist depends on the formulation and the case: the psi-omega
    system has no pressure in it, and Case A has no entrance length. Printing
    unconditionally is how this crashed every psi-omega run after its results
    were already safely written.
    """
    if key in metrics and metrics[key] is not None:
        print(f"  {label:<21}: {metrics[key]:.3e}{suffix}")


def _print_summary(metrics: dict) -> None:
    print("  --- accuracy (relative L2 against the analytic solution) ---")
    if metrics["case"] == "A":
        _show(metrics, "u_relative_l2", "u relative L2", "   (target <= 1e-5)")
        _show(metrics, "p_relative_l2", "p relative L2")
        _show(metrics, "p_relative_l2_recovered", "p relative L2 (recovered)")
    else:
        _show(metrics, "outlet_profile_relative_l2", "outlet profile L2",
              "  (target <= 1e-4)")
        if "entrance_length_measured" in metrics:
            print(f"  {'entrance length':<21}: {metrics['entrance_length_measured']:.4f} vs "
                  f"{metrics['entrance_length_correlation']:.4f} correlation "
                  f"({metrics['entrance_length_relative_error'] * 100:.1f}%, target <= 15%)")
    _show(metrics, "v_max_abs", "max |v*|",
          "   (target <= 1e-5)" if metrics["case"] == "A"
          else "   (no target: case B has genuine transverse velocity)")
    _show(metrics, "flux_conservation_relative_error", "mass drift D",
          "   (target <= 1e-6)")
    _show(metrics, "flux_relative_error_vs_inlet", "flux vs inlet flux")

    print("  --- diagnostics (NOT accuracy claims) ---")
    final = metrics.get("diag_loss_after_lbfgs", metrics.get("diag_loss_after_adam"))
    if final is not None:
        print(f"  {'final loss':<21}: {final:.3e}")
    _show(metrics, "residual_momentum_x_max", "max momentum-x resid")
    _show(metrics, "residual_vorticity_max", "max r1 resid")
    _show(metrics, "residual_vorticity_transport_max", "max r2 resid")
    _show(metrics, "pressure_path_max_disagreement", "pressure path disagree")
    # Guard at display time as well as at fit time: metrics.json is a snapshot
    # written by whatever code produced the run, so a stored verdict may predate
    # the machine-precision guard in drift.py.
    drift = metrics.get("flux_conservation_relative_error")
    if drift is not None and drift < 1e-12:
        print(f"  {'drift':<21}: D is at machine precision -- this formulation "
              "conserves mass by")
        print(f"  {'':<21}  construction, so test 3 is VACUOUS here, not passed")
    elif metrics.get("diag_drift_verdict"):
        print(f"  {'drift D_inf':<21}: {metrics['diag_drift_d_infinity']:.3e} "
              f"+- {metrics['diag_drift_d_infinity_stderr']:.1e}")
        print(f"  {'drift verdict':<21}: {metrics['diag_drift_verdict']}")
    print(f"  {'wall time':<21}: {metrics['wall_seconds']:.1f} s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to an experiments/*.yaml")
    parser.add_argument("--out", default=None, help="override the run directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    execute(args.config, args.out, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
