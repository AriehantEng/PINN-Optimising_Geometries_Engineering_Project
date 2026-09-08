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


def _print_summary(metrics: dict) -> None:
    print("  --- accuracy (relative L2 against the analytic solution) ---")
    if metrics["case"] == "A":
        print(f"  u relative L2        : {metrics['u_relative_l2']:.3e}   (target <= 1e-5)")
        print(f"  p relative L2        : {metrics['p_relative_l2']:.3e}")
    else:
        print(f"  outlet profile L2    : {metrics['outlet_profile_relative_l2']:.3e}  (target <= 1e-4)")
        print(f"  entrance length      : {metrics['entrance_length_measured']:.4f} vs "
              f"{metrics['entrance_length_correlation']:.4f} correlation "
              f"({metrics['entrance_length_relative_error'] * 100:.1f}%, target <= 15%)")
    print(f"  max |v*|             : {metrics['v_max_abs']:.3e}   (target <= 1e-5)")
    print(f"  flux conservation    : {metrics['flux_conservation_relative_error']:.3e}   (target <= 1e-6)")
    print(f"  flux vs inlet flux   : {metrics['flux_relative_error_vs_inlet']:.3e}")
    print("  --- diagnostics (NOT accuracy claims) ---")
    print(f"  final loss           : {metrics.get('diag_loss_after_lbfgs', metrics.get('diag_loss_after_adam')):.3e}")
    print(f"  max momentum-x resid : {metrics['residual_momentum_x_max']:.3e}")
    print(f"  wall time            : {metrics['wall_seconds']:.1f} s")


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
