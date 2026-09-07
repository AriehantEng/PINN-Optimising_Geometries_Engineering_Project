"""Figures for a finished run. Written into the run directory.

Matplotlib only, no seaborn, one figure per file, default colours. These are
diagnostic figures for the Phase 1 write-up, not publication graphics.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import Config  # noqa: E402
from ..geometry.domain import Channel  # noqa: E402
from ..geometry.sampling import evaluation_grid  # noqa: E402
from ..physics import analytic  # noqa: E402
from ..physics import residuals  # noqa: E402
from ..physics.derivatives import batched_field_derivatives  # noqa: E402
from .metrics import evaluate_fields  # noqa: E402


def _grid_fields(config: Config, field_fn, params):
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)
    u, v, p = evaluate_fields(field_fn, params, points)
    shape = (config.evaluation.nx, config.evaluation.ny)
    return np.asarray(x), np.asarray(y), u.reshape(shape), v.reshape(shape), p.reshape(shape), points


def plot_velocity_profiles(config, field_fn, params, path: Path):
    """Predicted vs analytic profile at several stations."""
    x, y, u, _, _, _ = _grid_fields(config, field_fn, params)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    stations = np.linspace(0, len(x) - 1, 5).astype(int)
    for index in stations:
        axes[0].plot(u[index], y, label=f"x* = {x[index]:.1f}")
    if config.case == "A":
        reference = analytic.u_nd(y)
        label = "analytic  1 - y*²"
    else:
        reference = analytic.developed_profile_nd(y, config.inlet_flux)
        label = f"developed (flux {config.inlet_flux:.3f})"
    axes[0].plot(reference, y, "k--", linewidth=2, label=label)
    axes[0].set_xlabel("u*")
    axes[0].set_ylabel("y*")
    axes[0].set_title(f"{config.name}: streamwise profiles")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if config.case == "A":
        error = np.abs(u - analytic.u_nd(y)[None, :])
        image = axes[1].pcolormesh(x, y, error.T, shading="auto")
        fig.colorbar(image, ax=axes[1], label="|u* - u*_exact|")
        axes[1].set_title("pointwise error in u*")
    else:
        axes[1].plot(x, u[:, u.shape[1] // 2])
        axes[1].axhline(0.75 * config.inlet_flux, color="k", linestyle="--",
                        label="developed centreline")
        axes[1].set_ylabel("u*(x*, 0)")
        axes[1].set_title("centreline development")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("x*")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residual_map(config, field_fn, params, path: Path):
    """Where the momentum residual lives. A diagnostic, never an accuracy claim."""
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)
    d = batched_field_derivatives(lambda xy: field_fn(params, xy), points)
    shape = (config.evaluation.nx, config.evaluation.ny)
    r_x, r_y, r_c = residuals.navier_stokes(
        d, config.physics.reynolds, config.physics.pressure_scaling
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for axis, values, name in zip(
        axes, (r_x, r_y, r_c), ("momentum x", "momentum y", "continuity")
    ):
        magnitude = np.abs(np.asarray(values)).reshape(shape).T
        floor = max(magnitude.max(), 1e-300) * 1e-8
        image = axis.pcolormesh(
            np.asarray(x), np.asarray(y), np.log10(np.maximum(magnitude, floor)),
            shading="auto",
        )
        fig.colorbar(image, ax=axis, label="log10 |residual|")
        axis.set_ylabel("y*")
        axis.set_title(f"{name} residual (unweighted, diagnostic only)")
    axes[-1].set_xlabel("x*")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_loss_history(history_path: Path, path: Path, title: str):
    """Per-term loss history from the Adam phase."""
    if not history_path.is_file():
        return
    with history_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    steps = np.array([float(row["step"]) for row in rows])
    term_columns = sorted(c for c in rows[0] if c.startswith("term_"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(steps, [float(row["loss"]) for row in rows], "k", label="total")
    for column in term_columns:
        values = np.array([float(row[column]) for row in rows])
        axes[0].semilogy(steps, np.maximum(values, 1e-300), label=column[5:], alpha=0.8)
    axes[0].set_xlabel("Adam step")
    axes[0].set_ylabel("loss (diagnostic, not accuracy)")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    weight_columns = sorted(c for c in rows[0] if c.startswith("weight_"))
    for column in weight_columns:
        axes[1].semilogy(steps, [float(row[column]) for row in rows], label=column[7:])
    axes[1].set_xlabel("Adam step")
    axes[1].set_ylabel("loss weight")
    axes[1].set_title("loss weights")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_all(config: Config, field_fn, params, directory: Path):
    plot_velocity_profiles(config, field_fn, params, directory / "profiles.png")
    plot_residual_map(config, field_fn, params, directory / "residuals.png")
    plot_loss_history(
        directory / "history.csv", directory / "loss_history.png", config.name
    )
