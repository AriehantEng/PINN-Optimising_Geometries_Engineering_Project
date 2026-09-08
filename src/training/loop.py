"""The training loop: Adam with resampling, then L-BFGS on frozen points.

Determinism (SPEC test 8) is structural here: every random draw comes from a
PRNG key derived from `config.seed` or `config.sampling.seed`, there is no
wall-clock or hash-order dependence, and the L-BFGS phase is SciPy's
deterministic implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import Config
from ..geometry.domain import Channel
from ..geometry.sampling import sample_all
from ..models.build import build_model, count_parameters
from .losses import make_loss_terms
from .optimisers import adam_schedule, make_adam, run_lbfgs
from ..evaluation.drift import fit_drift, mass_drift
from ..evaluation.metrics import cross_channel_flux
from .weights import initial_weights, make_grad_norm_update, grad_norm_weights, total_loss

#: Target number of logged rows across the Adam phase. Logging cadence is
#: derived from this rather than configured, because it changes nothing about
#: the result -- only how densely the history is recorded.
TARGET_LOG_ROWS = 200


#: Points across the channel used by the live accuracy probe. Small on purpose:
#: it runs at every log interval, and the reported accuracy at the END of a run
#: is always recomputed on the full evaluation grid.
PROBE_NX, PROBE_NY = 81, 41


def _make_accuracy_probe(config, model):
    """Relative L2 against the analytic solution, as a function of parameters.

    This is the number that actually means something during training. The loss
    is a diagnostic and can fall while this rises -- that is the failure mode
    the whole project is built around -- so both are logged at every interval
    and printed side by side.

    Case A has an analytic solution everywhere. Case B does not, so the probe
    reports the outlet profile against the flux-consistent developed parabola,
    which is what SPEC test 9 measures.
    """
    from ..geometry.sampling import evaluation_grid
    from ..physics import analytic

    channel = Channel(config.domain.length)
    if config.case == "A":
        points, _, _ = evaluation_grid(channel, PROBE_NX, PROBE_NY)
        reference = analytic.u_nd(np.asarray(points[:, 1]))
    else:
        y = np.linspace(-1.0, 1.0, PROBE_NY * 4)
        points = jnp.stack(
            [jnp.full(len(y), config.domain.length), jnp.asarray(y)], axis=-1
        )
        reference = analytic.developed_profile_nd(y, config.inlet_flux)

    norm = float(np.linalg.norm(reference))

    def probe(params):
        u = np.asarray(
            jax.vmap(lambda xy: model.velocity(params, xy))(points)[:, 0]
        )
        return float(np.linalg.norm(u - reference) / norm)

    return probe


def _make_drift_probe(config, model):
    """D on the fixed evaluation stations, as a function of parameters alone."""
    length = config.domain.length
    stations = np.linspace(0.05 * length, 0.95 * length, 10)

    def probe(params):
        try:
            return mass_drift(cross_channel_flux(model.field, params, stations, 64))
        except (ValueError, FloatingPointError):
            return float("nan")

    return probe


@dataclass
class TrainingResult:
    params: Any
    model: Any
    history: list = field(default_factory=list)
    lbfgs_history: list = field(default_factory=list)
    drift_trace: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def train(config: Config, *, verbose: bool = True) -> TrainingResult:
    channel = Channel(config.domain.length)

    model_key = jax.random.PRNGKey(config.seed)
    params, model = build_model(config, model_key)

    loss_terms_fn = make_loss_terms(config, model)
    weights = initial_weights(config)

    def weighted_loss(p, points, w):
        return total_loss(loss_terms_fn(p, points), w)

    optimiser = make_adam(config)
    opt_state = optimiser.init(params)
    schedule = adam_schedule(config)

    @jax.jit
    def adam_step(p, state, points, w):
        def loss_and_terms(q):
            # `has_aux` returns the per-term losses from the SAME trace that
            # produced the gradient, instead of a second forward pass just to
            # log them.
            terms = loss_terms_fn(q, points)
            return total_loss(terms, w), terms

        (loss, terms), grads = jax.value_and_grad(loss_and_terms, has_aux=True)(p)
        updates, state = optimiser.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, terms

    term_grad_norms = None
    if config.weights.scheme == "grad_norm":
        term_grad_norms = jax.jit(make_grad_norm_update(loss_terms_fn))

    # Held-out validation points. Drawn from a PRNG stream that the training
    # sampler never reaches, so they are disjoint from every training draw.
    validation_points = sample_all(
        jax.random.PRNGKey(config.sampling.seed + 1_000_003),
        channel,
        config.sampling.n_validation,
        config.sampling.n_inlet,
        config.sampling.n_outlet,
    )
    validation_loss = jax.jit(
        lambda p: total_loss(loss_terms_fn(p, validation_points), weights)
    )

    best = {"validation": float("inf"), "params": params, "where": "init"}

    def consider(p, where):
        """Keep the parameters with the lowest HELD-OUT loss seen so far."""
        value = float(validation_loss(p))
        if value < best["validation"]:
            best.update(validation=value, params=jax.tree.map(lambda a: a, p), where=where)
        return value

    sample_key = jax.random.PRNGKey(config.sampling.seed)
    points = sample_all(
        sample_key,
        channel,
        config.sampling.n_interior,
        config.sampling.n_inlet,
        config.sampling.n_outlet,
    )

    history = []
    log_every = max(1, config.training.adam.steps // TARGET_LOG_ROWS)
    accuracy_probe = _make_accuracy_probe(config, model)
    resample_every = config.sampling.resample_every
    started = time.time()
    if verbose:
        print(
            f"  [adam] {config.training.adam.steps} steps, resampling every "
            f"{resample_every} (one 'epoch' = one resampling interval, "
            f"{config.training.adam.steps // max(resample_every, 1)} of them)",
            flush=True,
        )
        print(f"  {'step':>8} {'epoch':>6} {'train loss':>12} {'held-out':>12} "
              f"{'rel L2 vs exact':>16} {'lr':>9}", flush=True)

    for step in range(config.training.adam.steps):
        if step > 0 and step % config.sampling.resample_every == 0:
            sample_key, draw_key = jax.random.split(sample_key)
            points = sample_all(
                draw_key,
                channel,
                config.sampling.n_interior,
                config.sampling.n_inlet,
                config.sampling.n_outlet,
            )

        if (
            term_grad_norms is not None
            and step % config.weights.grad_norm.update_every == 0
        ):
            weights = grad_norm_weights(
                term_grad_norms(params, points), weights, config.weights.grad_norm.alpha
            )

        params, opt_state, loss, terms = adam_step(params, opt_state, points, weights)

        if step % log_every == 0 or step == config.training.adam.steps - 1:
            accuracy = accuracy_probe(params)
            held_out = consider(params, f"adam:{step}")
            row = {
                "phase": "adam",
                "step": step,
                "epoch": step // max(resample_every, 1),
                "loss": float(loss),
                "validation_loss": held_out,
                "relative_l2_vs_exact": accuracy,
                "lr": float(schedule(step)),
                **{f"term_{k}": float(v) for k, v in terms.items()},
                **{f"weight_{k}": float(v) for k, v in weights.items()},
            }
            history.append(row)
            if verbose and (step % (log_every * 10) == 0):
                print(
                    f"  {step:>8d} {step // max(resample_every, 1):>6d} "
                    f"{float(loss):>12.4e} {held_out:>12.4e} {accuracy:>16.4e} "
                    f"{float(schedule(step)):>9.2e}",
                    flush=True,
                )

    adam_seconds = time.time() - started

    diagnostics = {
        "parameter_count": count_parameters(params),
        "adam_seconds": adam_seconds,
        "final_weights": {k: float(v) for k, v in weights.items()},
    }

    # -- L-BFGS on FROZEN points -------------------------------------------- #
    lbfgs_history = []
    drift_trace = []
    if config.training.lbfgs.enabled:
        frozen = points
        loss_before = float(weighted_loss(params, frozen, weights))

        def objective(p):
            return weighted_loss(p, frozen, weights)

        # SPEC v2 S7: trace the mass drift D against L-BFGS iteration. Measured
        # on the FIXED evaluation stations, never on the training collocation
        # points (T-S7.1) -- the frozen set above is the training set.
        drift_fn = _make_drift_probe(config, model)

        def trace_fn(p):
            return {
                "drift": drift_fn(p),
                "relative_l2_vs_exact": accuracy_probe(p),
                "validation_loss": consider(p, "lbfgs"),
            }

        if verbose:
            print(
                f"  [lbfgs] up to {config.training.lbfgs.max_iterations} iterations",
                flush=True,
            )
            print(f"  {'iter':>8} {'':>6} {'train loss':>12} {'held-out':>12} "
                  f"{'rel L2 vs exact':>16} {'drift D':>9}", flush=True)

        def on_trace(n, values, loss):
            if verbose:
                print(
                    f"  {n:>8d} {'':>6} {loss:>12.4e} "
                    f"{values['validation_loss']:>12.4e} "
                    f"{values['relative_l2_vs_exact']:>16.4e} {values['drift']:>9.2e}",
                    flush=True,
                )

        started = time.time()
        result = run_lbfgs(
            objective,
            params,
            config.training.lbfgs.max_iterations,
            config.training.lbfgs.gtol,
            trace_fn=trace_fn,
            on_trace=on_trace,
        )
        lbfgs_seconds = time.time() - started

        # SPEC section 9: verify the loss is continuous across the handoff. The
        # first L-BFGS evaluation is at the Adam parameters on the same frozen
        # points, so any difference is a bug in the handoff, not optimisation.
        handoff_gap = abs(result.initial_loss - loss_before)
        handoff_relative = handoff_gap / max(abs(loss_before), 1e-300)

        final_validation = consider(result.params, "lbfgs:final")
        params = result.params
        diagnostics["validation_loss_final"] = final_validation
        lbfgs_history.append(
            {
                "phase": "lbfgs",
                "loss_at_handoff_adam_side": loss_before,
                "loss_at_handoff_lbfgs_side": result.initial_loss,
                "handoff_absolute_gap": handoff_gap,
                "handoff_relative_gap": handoff_relative,
                "final_loss": result.final_loss,
                "iterations": result.iterations,
                "function_evaluations": result.function_evaluations,
                "status": result.status,
                "message": result.message,
            }
        )
        drift_trace = result.trace
        if drift_trace:
            fit = fit_drift(
                [row["iteration"] for row in drift_trace],
                [row["drift"] for row in drift_trace],
            )
            diagnostics.update(
                {
                    "drift_d_infinity": fit.d_infinity,
                    "drift_d_infinity_stderr": fit.d_infinity_stderr,
                    "drift_exponent": fit.exponent,
                    "drift_fit_points": fit.points,
                    "drift_fit_converged": fit.converged,
                    "drift_verdict": fit.verdict(),
                }
            )

        diagnostics.update(
            {
                "lbfgs_seconds": lbfgs_seconds,
                "lbfgs_iterations": result.iterations,
                "lbfgs_status": result.status,
                "lbfgs_message": result.message,
                "handoff_relative_gap": handoff_relative,
                "loss_after_adam": loss_before,
                "loss_after_lbfgs": result.final_loss,
            }
        )
        if verbose:
            print(
                f"  lbfgs {result.iterations} iters, "
                f"loss {loss_before:.6e} -> {result.final_loss:.6e}, "
                f"handoff gap {handoff_relative:.2e} ({result.message})",
                flush=True,
            )
    else:
        diagnostics["loss_after_adam"] = float(weighted_loss(params, points, weights))

    # SPEC v2 does not cover this; it is the fix for the loss/accuracy
    # divergence. Selection uses the HELD-OUT loss, never the analytic
    # solution, so the same mechanism works in Phase 2.
    diagnostics["validation_loss_best"] = best["validation"]
    diagnostics["selected_from"] = best["where"]
    diagnostics["selection_rule"] = config.training.select
    if config.training.select == "best_validation":
        final_value = diagnostics.get("validation_loss_final")
        improved = final_value is None or best["validation"] < final_value
        if verbose and best["where"] != "lbfgs:final" and improved:
            print(
                f"  selected parameters from {best['where']} "
                f"(held-out loss {best['validation']:.4e}) rather than the final "
                f"iterate ({diagnostics.get('validation_loss_final', float('nan')):.4e})",
                flush=True,
            )
        params = best["params"]

    return TrainingResult(
        drift_trace=drift_trace if config.training.lbfgs.enabled else [],
        params=params,
        model=model,
        history=history,
        lbfgs_history=lbfgs_history,
        diagnostics=diagnostics,
    )
