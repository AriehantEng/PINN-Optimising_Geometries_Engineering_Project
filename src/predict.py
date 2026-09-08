"""Use a trained model.

A finished run in `runs/<id>/` is a reusable solver, not just a folder of
numbers. This module loads one back and evaluates it anywhere in the channel.

    from src.predict import load_run
    solution = load_run("runs/caseA_re10")
    u, v = solution.velocity(3.0, 0.5)
    y, u = solution.profile(x=3.0)

or from a shell:

    python -m src.predict --run runs/caseA_re10 --at 3.0,0.5
    python -m src.predict --run runs/caseA_re10 --profile-at 3.0 --out profile.csv
    python -m src.predict --run runs/caseA_re10 --grid 201 101 --out field.csv

Two things this deliberately refuses to do quietly:

* It will not evaluate outside the trained domain. A PINN is fitted, not
  solved, so outside `x* in [0, L*]` and `y* in [-1, 1]` it is extrapolating
  and the residual it was trained on says nothing there. Pass
  `allow_extrapolation=True` if you mean it.
* It will not report a pressure the model does not have. The psi-omega system
  has no pressure in it, so `pressure()` recovers one post hoc by path
  integration and says so, rather than returning the placeholder zero that
  lives in the field's third slot.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .config import Config, load_config
from .geometry.domain import WALL
from .models.build import build_model
from .physics import streamfunction
from .evaluation.metrics import cross_channel_flux


@dataclass
class TrainedModel:
    """A trained solution, ready to evaluate."""

    config: Config
    params: object
    model: object
    run_directory: Path

    # -- introspection --------------------------------------------------------

    @property
    def domain_length(self) -> float:
        return self.config.domain.length

    def summary(self) -> str:
        c = self.config
        return (
            f"{c.name}: case {c.case}, Re_h = {c.physics.reynolds:g}, "
            f"{c.physics.formulation}, L* = {c.domain.length:.3f}, "
            f"pressure in system: {self.model.has_pressure}"
        )

    # -- point evaluation -----------------------------------------------------

    def _check_domain(self, x, y, allow_extrapolation):
        if allow_extrapolation:
            return
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        outside = (
            (x < 0.0) | (x > self.domain_length) | (y < -WALL) | (y > WALL)
        )
        if np.any(outside):
            first = int(np.argmax(outside))
            raise ValueError(
                f"point (x*={x[first]:g}, y*={y[first]:g}) lies outside the "
                f"trained domain x* in [0, {self.domain_length:g}], "
                f"y* in [-1, 1]. A PINN is fitted, not solved: outside the "
                "region where its residual was enforced it is extrapolating "
                "and its output is not a solution of anything. Pass "
                "allow_extrapolation=True if that is what you want."
            )

    def velocity(self, x, y, *, allow_extrapolation=False):
        """(u*, v*) at the given non-dimensional coordinates. Broadcasts."""
        self._check_domain(x, y, allow_extrapolation)
        points = self._as_points(x, y)
        out = jax.vmap(lambda p: self.model.velocity(self.params, p))(points)
        u = np.asarray(out[:, 0]).reshape(np.shape(x) if np.ndim(x) else ())
        v = np.asarray(out[:, 1]).reshape(np.shape(x) if np.ndim(x) else ())
        return (float(u), float(v)) if u.ndim == 0 else (u, v)

    def pressure(self, x, y, *, allow_extrapolation=False):
        """p* at the given coordinates, gauged to zero at the outlet centreline.

        For the velocity-pressure formulation this is a network output. For the
        psi-omega system pressure is absent from the solved system and is
        recovered by integrating grad p along a path from the gauge point,
        which is slower and carries its own consistency error -- the run's
        `pressure_path_max_disagreement` metric measures it.
        """
        self._check_domain(x, y, allow_extrapolation)
        if self.model.has_pressure:
            points = self._as_points(x, y)
            out = jax.vmap(lambda p: self.model.field(self.params, p))(points)
            p = np.asarray(out[:, 2]).reshape(np.shape(x) if np.ndim(x) else ())
            return float(p) if p.ndim == 0 else p

        velocity_fn = lambda xy: self.model.velocity(self.params, xy)  # noqa: E731
        gauge = (self.domain_length, 0.0)
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
        xs, ys = np.broadcast_arrays(xs, ys)
        values = np.array([
            streamfunction.recover_pressure(
                velocity_fn, (float(a), float(b)), gauge,
                self.config.physics.reynolds, self.config.physics.pressure_scaling,
            )
            for a, b in zip(xs.ravel(), ys.ravel())
        ]).reshape(xs.shape)
        return float(values) if np.ndim(x) == 0 and np.ndim(y) == 0 else values

    def fields(self, x, y, *, allow_extrapolation=False) -> dict:
        u, v = self.velocity(x, y, allow_extrapolation=allow_extrapolation)
        return {
            "u": u, "v": v,
            "p": self.pressure(x, y, allow_extrapolation=allow_extrapolation),
        }

    # -- derived quantities ---------------------------------------------------

    def profile(self, x, n=201):
        """Cross-channel velocity profile at one streamwise station."""
        y = np.linspace(-WALL, WALL, n)
        u, v = self.velocity(np.full_like(y, float(x)), y)
        return y, u, v

    def flux(self, x):
        """Volumetric flux per unit depth through the station(s) x*."""
        stations = np.atleast_1d(np.asarray(x, dtype=np.float64))
        values = cross_channel_flux(self.model.field, self.params, stations)
        return float(values[0]) if np.ndim(x) == 0 else np.asarray(values)

    def grid(self, nx=201, ny=101):
        """(x, y, u, v) on a regular grid covering the whole channel."""
        x = np.linspace(0.0, self.domain_length, nx)
        y = np.linspace(-WALL, WALL, ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        u, v = self.velocity(xx.ravel(), yy.ravel())
        return x, y, u.reshape(nx, ny), v.reshape(nx, ny)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _as_points(x, y):
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
        xs, ys = np.broadcast_arrays(xs, ys)
        return jnp.asarray(np.stack([xs.ravel(), ys.ravel()], axis=-1))


def load_run(run_directory) -> TrainedModel:
    """Load a finished run back into an evaluable model.

    The parameter pytree's STRUCTURE is rebuilt from the run's own config
    rather than stored, so a run directory stays readable and version
    independent. That makes it essential that the config travels with the
    parameters -- which is why `run.py` copies it into the run directory.
    """
    directory = Path(run_directory)
    config_path = directory / "config.yaml"
    params_path = directory / "params.npz"
    for path in (config_path, params_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. A usable run directory needs both "
                "config.yaml and params.npz; has this run finished?"
            )

    config = load_config(config_path)
    template, model = build_model(config, jax.random.PRNGKey(config.seed))
    flat_template, unflatten = ravel_pytree(template)

    flat = np.load(params_path)["flat"]
    if flat.shape != flat_template.shape:
        raise ValueError(
            f"{params_path} holds {flat.shape[0]} parameters but the config in "
            f"{config_path} builds a model with {flat_template.shape[0]}. The "
            "config and the saved weights are from different models."
        )
    return TrainedModel(config, unflatten(jnp.asarray(flat)), model, directory)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--run", required=True, help="a finished runs/<id> directory")
    parser.add_argument("--at", help="a single point as 'x,y'")
    parser.add_argument("--profile-at", type=float, help="cross-channel profile at x*")
    parser.add_argument("--flux-at", type=float, help="flux through the station x*")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("NX", "NY"),
                        help="evaluate on an NX by NY grid")
    parser.add_argument("--out", help="write CSV here instead of printing")
    parser.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args(argv)

    solution = load_run(args.run)
    print(solution.summary())

    if args.at:
        x, y = (float(v) for v in args.at.split(","))
        f = solution.fields(x, y, allow_extrapolation=args.allow_extrapolation)
        print(f"  at (x*={x:g}, y*={y:g}):  u* = {f['u']:+.9f}  "
              f"v* = {f['v']:+.9f}  p* = {f['p']:+.9f}")

    if args.profile_at is not None:
        y, u, v = solution.profile(args.profile_at)
        rows = list(zip(y, u, v))
        if args.out:
            _write_csv(args.out, ["y", "u", "v"], rows)
            print(f"  wrote {len(rows)} profile points to {args.out}")
        else:
            print(f"  profile at x* = {args.profile_at:g} (every 20th point):")
            for yy, uu, vv in rows[::20]:
                print(f"    y*={yy:+.4f}  u*={uu:+.9f}  v*={vv:+.9f}")

    if args.flux_at is not None:
        print(f"  flux at x* = {args.flux_at:g}: {solution.flux(args.flux_at):.12f}")

    if args.grid:
        nx, ny = args.grid
        x, y, u, v = solution.grid(nx, ny)
        rows = [
            (x[i], y[j], u[i, j], v[i, j]) for i in range(nx) for j in range(ny)
        ]
        if args.out:
            _write_csv(args.out, ["x", "y", "u", "v"], rows)
            print(f"  wrote {len(rows)} grid points to {args.out}")
        else:
            print(f"  grid {nx}x{ny}: u* in [{u.min():.6f}, {u.max():.6f}], "
                  f"max |v*| = {np.abs(v).max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
