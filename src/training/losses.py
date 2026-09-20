"""Per-term losses. Unweighted.

Each term is the mean squared residual of one governing equation or one
boundary condition, evaluated on the collocation points for that term. Nothing
here multiplies by a weight -- that happens in `weights.py` and nowhere else
(CLAUDE.md, "Code style").

There is deliberately no wall term. No-slip is exact by construction in the
ansatz, so a wall loss would be identically zero and would only dilute the
gradient of the terms that matter.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..config import Config
from ..physics import inlets
from ..physics import residuals, streamfunction
from ..physics.derivatives import (
    batched_field_and_jacobian,
    batched_field_derivatives,
    batched_stream_derivatives,
)


def inlet_profile(config: Config, y):
    """Prescribed streamwise velocity at x* = 0.

    Case A: the fully developed parabola u* = 1 - y*^2.
    Case B "uniform":  the top hat u* = `physics.inlet_uniform_value`.
    Case B "smoothed": U0 (1 - y^(2m)) from `physics.inlet_sharpness`, which
        satisfies no-slip exactly and so places a BOUNDED demand on the
        network, unlike the top hat. See src/physics/inlets.py.
    """
    if config.case == "A":
        return 1.0 - y**2
    if config.inlet_profile == "smoothed":
        return inlets.profile(y, config.inlet_sharpness)
    return jnp.full_like(y, config.inlet_uniform_value)


def make_loss_terms(config: Config, model):
    """Build `loss_terms(params, points) -> dict[str, scalar]`.

    The returned dict has exactly the keys in `config.loss_terms`, so a term
    can never be silently dropped or silently added.
    """
    reynolds = config.physics.reynolds
    pressure_scaling = config.physics.pressure_scaling
    convective_coefficient = config.physics.convective_coefficient
    formulation = config.physics.formulation

    def loss_terms(params, points):
        point_fn = lambda xy: model.field(params, xy)  # noqa: E731

        if formulation == "psi_omega":
            stream_point_fn = lambda xy: model.stream(params, xy)  # noqa: E731
            sd = batched_stream_derivatives(stream_point_fn, points.interior)
            r1, r2 = streamfunction.residuals(sd, reynolds)
            terms = {
                "vorticity": jnp.mean(r1**2),
                "vorticity_transport": jnp.mean(r2**2),
            }
        else:
            d = batched_field_derivatives(point_fn, points.interior)
            terms = {
                "momentum_x": jnp.mean(
                    residuals.momentum_x(
                        d, reynolds, pressure_scaling, convective_coefficient
                    )
                    ** 2
                ),
                "momentum_y": jnp.mean(
                    residuals.momentum_y(
                        d, reynolds, pressure_scaling, convective_coefficient
                    )
                    ** 2
                ),
                "continuity": jnp.mean(residuals.continuity(d) ** 2),
            }

        if config.bc.inlet == "soft":
            inlet_values, _ = batched_field_and_jacobian(point_fn, points.inlet)
            target = inlet_profile(config, points.inlet[:, 1])
            terms["inlet"] = jnp.mean((inlet_values[:, 0] - target) ** 2) + jnp.mean(
                inlet_values[:, 1] ** 2
            )

        needs_outlet = (
            config.bc.outlet_velocity == "soft"
            or config.bc.pressure_gauge == "soft"
        )
        if needs_outlet:
            outlet_values, outlet_jac = batched_field_and_jacobian(
                point_fn, points.outlet
            )
            if config.bc.outlet_velocity == "soft":
                # Zero streamwise gradient of velocity: du/dx = dv/dx = 0.
                terms["outlet_velocity"] = jnp.mean(
                    outlet_jac[:, 0, 0] ** 2
                ) + jnp.mean(outlet_jac[:, 1, 0] ** 2)
            if config.bc.pressure_gauge == "soft":
                # The gauge fix. Without it the pressure error is meaningless.
                terms["pressure_gauge"] = jnp.mean(outlet_values[:, 2] ** 2)

        return terms

    return loss_terms
