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


def _grid_fields(config: Config, model, params):
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)
    u, v, p = evaluate_fields(model.field, params, points)
    shape = (config.evaluation.nx, config.evaluation.ny)
    return np.asarray(x), np.asarray(y), u.reshape(shape), v.reshape(shape), p.reshape(shape), points


def plot_velocity_profiles(config, model, params, path: Path):
    """Predicted vs analytic profile at several stations.

    For Case A the agreement is around 1e-7, which means an overlay plot shows
    six pixel-identical curves: it proves the match and displays nothing. So
    Case A gets a middle panel carrying the SIGNED DIFFERENCE against the
    analytic profile on its own axis, which is the panel that actually shows
    how close the two are. The overlay keeps the predicted profile as sparse
    markers over a solid analytic line for the same reason -- coincidence has
    to be visible rather than hidden by overplotting.
    """
    x, y, u, _, _, _ = _grid_fields(config, model, params)
    case_a = config.case == "A"
    columns = 3 if case_a else 2
    fig, axes = plt.subplots(1, columns, figsize=(5.5 * columns, 4.2))

    stations = np.linspace(0, len(x) - 1, 5).astype(int)
    if case_a:
        reference = analytic.u_nd(y)
        label = "analytic  1 - y*²"
    else:
        reference = analytic.developed_profile_nd(y, config.inlet_flux)
        label = f"developed (flux {config.inlet_flux:.3f})"

    # Analytic first and solid, prediction over it as markers.
    axes[0].plot(reference, y, "k-", linewidth=2.0, label=label, zorder=1)
    every = max(1, len(y) // 24)
    for order, index in enumerate(stations):
        axes[0].plot(u[index][::every], y[::every], marker="o", linestyle="none",
                     markersize=4.0, markerfacecolor="none",
                     label=f"PINN  x* = {x[index]:.1f}", zorder=2 + order)
    axes[0].set_xlabel("u*")
    axes[0].set_ylabel("y*")
    axes[0].set_title(f"{config.name}: streamwise profiles")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if case_a:
        exact = analytic.u_nd(y)
        for index in stations:
            axes[1].plot(u[index] - exact, y, linewidth=1.2,
                         label=f"x* = {x[index]:.1f}")
        axes[1].axvline(0.0, color="k", linewidth=0.8)
        axes[1].set_xlabel("u* - u*_exact")
        axes[1].set_ylabel("y*")
        axes[1].set_title("difference from the analytic profile")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        error = np.abs(u - exact[None, :])
        relative_l2 = float(
            np.sqrt(np.sum(error**2) / np.sum(np.broadcast_to(exact, u.shape) ** 2))
        )
        axes[1].annotate(
            f"relative L2 = {relative_l2:.3e}\nmax |error| = {error.max():.3e}",
            xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        image = axes[2].pcolormesh(x, y, error.T, shading="auto")
        fig.colorbar(image, ax=axes[2], label="|u* - u*_exact|")
        axes[2].set_title("pointwise error in u*")
        axes[2].set_xlabel("x*")
        axes[2].grid(alpha=0.3)
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


def plot_residual_map(config, model, params, path: Path):
    """Where the momentum residual lives. A diagnostic, never an accuracy claim."""
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)
    shape = (config.evaluation.nx, config.evaluation.ny)
    if config.physics.formulation == "psi_omega":
        from ..physics import streamfunction
        from ..physics.derivatives import batched_stream_derivatives

        sd = batched_stream_derivatives(lambda xy: model.stream(params, xy), points)
        values_and_names = list(
            zip(
                streamfunction.residuals(sd, config.physics.reynolds),
                ("r1  omega + lap psi", "r2  vorticity transport"),
            )
        )
    else:
        d = batched_field_derivatives(lambda xy: model.field(params, xy), points)
        values_and_names = list(
            zip(
                residuals.navier_stokes(
                    d,
                    config.physics.reynolds,
                    config.physics.pressure_scaling,
                    config.physics.convective_coefficient,
                ),
                ("momentum x", "momentum y", "continuity"),
            )
        )

    fig, axes = plt.subplots(len(values_and_names), 1, figsize=(9, 8), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (values, name) in zip(axes, values_and_names):
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


def plot_loss_history(history_path: Path, path: Path, title: str,
                      drift_path: Path | None = None, weights_vary: bool = True):
    """The whole run: per-term Adam losses, both phases, and the true error.

    The previous version plotted the Adam phase alone. For the 30000-iteration
    Case A run that is 2000 of 32000 iterations -- it showed 6% of the run and
    none of the phase that produced the result, while `relative_l2_vs_exact`
    and `validation_loss` sat unplotted in the same CSV and the drift trace sat
    unplotted in the file next to it.

    The second panel therefore carries BOTH phases on one axis, and the third
    carries the accuracy curve, which is the only line in this figure that is
    an accuracy claim at all.
    """
    if not history_path.is_file():
        return
    with history_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    trace = []
    if drift_path is not None and drift_path.is_file():
        with drift_path.open(encoding="utf-8") as handle:
            trace = list(csv.DictReader(handle))

    def column(source, name):
        return np.array([float(r[name]) for r in source if r.get(name) not in (None, "")])

    def aligned(source, name):
        """The column, or None when it is absent or partly empty.

        Older runs wrote drift traces with an `iteration,value` schema and no
        validation or accuracy columns at all, so a caller cannot assume a
        column exists just because a later run has it.
        """
        values = column(source, name)
        return values if len(values) == len(source) and len(values) else None

    steps = column(rows, "step")
    adam_end = float(steps[-1]) if len(steps) else 0.0
    term_columns = sorted(c for c in rows[0] if c.startswith("term_"))
    panels = 3 + (1 if weights_vary else 0)
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.2))

    # -- per-term Adam loss ------------------------------------------------ #
    axes[0].semilogy(steps, column(rows, "loss"), "k", label="total", linewidth=1.6)
    for name in term_columns:
        axes[0].semilogy(steps, np.maximum(column(rows, name), 1e-300),
                         label=name[5:], alpha=0.85, linewidth=1.1)
    axes[0].set_xlabel("Adam step")
    axes[0].set_ylabel("loss (diagnostic, not accuracy)")
    axes[0].set_title(f"{title}: Adam, per term")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # -- both phases on one axis ------------------------------------------- #
    axis = axes[1]
    axis.semilogy(steps, column(rows, "loss"), color="C0", label="Adam, train")
    held = [r for r in rows if r.get("validation_loss")]
    if held:
        axis.semilogy(column(held, "step"), column(held, "validation_loss"),
                      color="C0", linestyle="--", alpha=0.7, label="Adam, held-out")
    trace_loss = aligned(trace, "validation_loss") if trace else None
    if trace_loss is not None:
        iterations = column(trace, "iteration") + adam_end
        axis.semilogy(iterations, trace_loss,
                      color="C3", label="L-BFGS, held-out")
    if trace:
        axis.axvline(adam_end, color="k", linewidth=0.9, linestyle=":")
        axis.annotate("Adam -> L-BFGS", xy=(adam_end, axis.get_ylim()[1]),
                      xytext=(4, -10), textcoords="offset points",
                      fontsize=8, rotation=90, va="top")
    axis.set_xlabel("iteration (Adam steps, then L-BFGS)")
    axis.set_ylabel("loss (diagnostic)")
    axis.set_title("whole run")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    # -- the accuracy curve ------------------------------------------------ #
    axis = axes[2]
    accurate = [r for r in rows if r.get("relative_l2_vs_exact")]
    if accurate:
        axis.semilogy(column(accurate, "step"), column(accurate, "relative_l2_vs_exact"),
                      color="C2", label="Adam")
    trace_error = aligned(trace, "relative_l2_vs_exact") if trace else None
    if trace_error is not None:
        axis.semilogy(column(trace, "iteration") + adam_end, trace_error,
                      color="C2", linestyle="-", alpha=0.6, label="L-BFGS")
    if trace:
        axis.axvline(adam_end, color="k", linewidth=0.9, linestyle=":")
    axis.axhline(1e-5, color="k", linestyle="--", linewidth=1.0,
                 label="SPEC target 1e-5")
    axis.set_xlabel("iteration (Adam steps, then L-BFGS)")
    axis.set_ylabel("relative L2 vs analytic")
    axis.set_title("ACCURACY (the only such claim here)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    if not accurate and not trace:
        axis.text(0.5, 0.5, "no accuracy trace recorded\n(run predates the column)",
                  ha="center", va="center", transform=axis.transAxes, fontsize=9)

    # -- weights, only when they move -------------------------------------- #
    if weights_vary:
        for name in sorted(c for c in rows[0] if c.startswith("weight_")):
            axes[3].semilogy(steps, column(rows, name), label=name[7:])
        axes[3].set_xlabel("Adam step")
        axes[3].set_ylabel("loss weight")
        axes[3].set_title("loss weights")
        axes[3].legend(fontsize=8)
        axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_convergence(drift_path: Path, path: Path, title: str):
    """Mass drift and true error against L-BFGS iteration, log-log.

    This is the figure carrying the Block C result -- D fell monotonically
    with budget rather than plateauing, which is what settled the drift as
    under-convergence rather than formulation bias -- and it existed in no
    form until now, despite the data being written by every run.
    """
    if not drift_path.is_file():
        return
    with drift_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        return

    def column(name):
        return np.array([float(r[name]) for r in rows if r.get(name) not in (None, "")])

    iterations = column("iteration")
    drift = column("drift") if "drift" in rows[0] else column("value")

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.loglog(iterations, np.maximum(drift, 1e-300), "o-", color="C3",
                markersize=3.5, label="mass drift D")
    error = column("relative_l2_vs_exact") if "relative_l2_vs_exact" in rows[0] else np.array([])
    if len(error) == len(iterations) and len(error):
        axis.loglog(iterations, np.maximum(error, 1e-300), "s-",
                    color="C2", markersize=3.5, label="relative L2 vs analytic")
        axis.axhline(1e-5, color="C2", linestyle="--", linewidth=1.0, alpha=0.7,
                     label="target 1e-5 (accuracy)")
    axis.axhline(1e-6, color="C3", linestyle="--", linewidth=1.0, alpha=0.7,
                 label="target 1e-6 (drift)")
    axis.set_xlabel("L-BFGS iteration")
    axis.set_ylabel("magnitude")
    axis.set_title(f"{title}: convergence against budget")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_all(config: Config, model, params, directory: Path):
    plot_velocity_profiles(config, model, params, directory / "profiles.png")
    plot_residual_map(config, model, params, directory / "residuals.png")
    plot_loss_history(
        directory / "history.csv", directory / "loss_history.png", config.name,
        drift_path=directory / "drift_trace.csv",
        weights_vary=config.weights.scheme != "fixed",
    )
    plot_convergence(
        directory / "drift_trace.csv", directory / "convergence.png", config.name
    )
