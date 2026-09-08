"""Optimiser schedule: Adam with exponential decay, then L-BFGS.

The handoff between the two is the delicate part. SPEC section 9 lists it as a
known trap: "Adam -> L-BFGS handoff frequently resets or mishandles state.
Verify the loss is continuous across the switch." Two things are done about it
here:

1. The collocation points are FROZEN for the whole L-BFGS phase. L-BFGS builds
   a curvature estimate from a sequence of gradients; resampling underneath it
   changes the objective between evaluations and the curvature estimate becomes
   meaningless. (This also means the loss values reported during L-BFGS are on
   a different point set from the Adam ones, which is why accuracy is always
   measured on the fixed evaluation grid instead.)

2. `run_lbfgs` returns the loss at its own first function evaluation, so the
   caller can compare it against the last Adam loss ON THE SAME POINTS and
   record the discontinuity. It should be zero to rounding.

L-BFGS itself is SciPy's, driven on a flattened float64 parameter vector.
SciPy's implementation is the well-tested Nocedal Fortran code and it is
deterministic, which SPEC test 8 depends on.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize


def adam_schedule(config):
    """Exponentially decaying learning rate, as configured."""
    return optax.exponential_decay(
        init_value=config.training.adam.lr_init,
        transition_steps=config.training.adam.decay_steps,
        decay_rate=config.training.adam.decay_rate,
        staircase=False,
    )


def make_adam(config):
    return optax.adam(learning_rate=adam_schedule(config))


class LBFGSResult(NamedTuple):
    params: object
    initial_loss: float
    final_loss: float
    iterations: int
    function_evaluations: int
    status: int
    message: str
    trace: list = []


def run_lbfgs(
    loss_fn: Callable, params, max_iterations: int, gtol: float,
    *, trace_fn=None, trace_every: int | None = None,
):
    """Minimise `loss_fn(params)` with SciPy L-BFGS-B on a flat float64 vector.

    `loss_fn` must be a pure function of the parameters alone; bind the
    collocation points before calling.

    `trace_fn(params) -> float` is sampled every `trace_every` iterations and
    recorded against the iteration count. SPEC v2 S7 uses this to trace the
    mass drift D against L-BFGS iteration and fit D(n) = D_inf + a n^-b, which
    is what discriminates formulation bias from under-convergence.
    """
    flat_initial, unflatten = ravel_pytree(params)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    evaluations = {"count": 0, "first_loss": None}

    def objective(flat):
        value, grad = value_and_grad(unflatten(jnp.asarray(flat)))
        evaluations["count"] += 1
        if evaluations["first_loss"] is None:
            evaluations["first_loss"] = float(value)
        flat_grad, _ = ravel_pytree(grad)
        # SciPy needs contiguous float64 numpy arrays, not JAX arrays.
        return float(value), np.asarray(flat_grad, dtype=np.float64)

    trace = []
    if trace_fn is not None:
        if trace_every is None:
            trace_every = max(1, max_iterations // 40)
        state = {"iteration": 0}

        def callback(flat):
            state["iteration"] += 1
            n = state["iteration"]
            if n % trace_every == 0 or n == 1:
                trace.append(
                    {"iteration": n, "value": float(trace_fn(unflatten(jnp.asarray(flat))))}
                )
    else:
        callback = None

    result = minimize(
        objective,
        np.asarray(flat_initial, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": max_iterations,
            "maxfun": max_iterations * 2,
            "gtol": gtol,
            # ftol = 0 so termination is governed by gtol alone; the default
            # ftol stops at ~1e-9 relative change, far short of the accuracy
            # this project targets.
            "ftol": 0.0,
            "maxcor": 100,
        },
    )

    return LBFGSResult(
        params=unflatten(jnp.asarray(result.x)),
        initial_loss=evaluations["first_loss"],
        final_loss=float(result.fun),
        iterations=int(result.nit),
        function_evaluations=int(evaluations["count"]),
        status=int(result.status),
        message=str(result.message),
        trace=trace,
    )
