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
from .weights import initial_weights, make_grad_norm_update, grad_norm_weights, total_loss

#: Target number of logged rows across the Adam phase. Logging cadence is
#: derived from this rather than configured, because it changes nothing about
#: the result -- only how densely the history is recorded.
TARGET_LOG_ROWS = 200


@dataclass
class TrainingResult:
    params: Any
    model: Any
    history: list = field(default_factory=list)
    lbfgs_history: list = field(default_factory=list)
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
    started = time.time()

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
            row = {
                "phase": "adam",
                "step": step,
                "loss": float(loss),
                "lr": float(schedule(step)),
                **{f"term_{k}": float(v) for k, v in terms.items()},
                **{f"weight_{k}": float(v) for k, v in weights.items()},
            }
            history.append(row)
            if verbose and (step % (log_every * 20) == 0):
                print(
                    f"  adam {step:>7d}  loss {float(loss):.6e}  lr {float(schedule(step)):.2e}",
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
    if config.training.lbfgs.enabled:
        frozen = points
        loss_before = float(weighted_loss(params, frozen, weights))

        def objective(p):
            return weighted_loss(p, frozen, weights)

        started = time.time()
        result = run_lbfgs(
            objective,
            params,
            config.training.lbfgs.max_iterations,
            config.training.lbfgs.gtol,
        )
        lbfgs_seconds = time.time() - started

        # SPEC section 9: verify the loss is continuous across the handoff. The
        # first L-BFGS evaluation is at the Adam parameters on the same frozen
        # points, so any difference is a bug in the handoff, not optimisation.
        handoff_gap = abs(result.initial_loss - loss_before)
        handoff_relative = handoff_gap / max(abs(loss_before), 1e-300)

        params = result.params
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

    return TrainingResult(
        params=params,
        model=model,
        history=history,
        lbfgs_history=lbfgs_history,
        diagnostics=diagnostics,
    )
