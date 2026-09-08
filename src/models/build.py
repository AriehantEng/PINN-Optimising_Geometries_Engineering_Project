"""Assemble a model from a validated `Config`.

Returns `(params, field_fn)` where `field_fn(params, xy) -> [u, v, p]` at a
single non-dimensional point. Everything that varies between experiments comes
from the config; nothing here has a default.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..config import Config
from .fourier import apply_fourier, init_fourier_matrix
from .hard_bc import (
    make_psi_omega_field,
    make_velocity_from_stream,
    make_velocity_pressure_field,
)
from .mlp import apply_mlp, apply_modified_mlp, init_mlp, init_modified_mlp

#: Raw network outputs per formulation.
#: velocity_pressure -> (N_u, N_v, N_p);  psi_omega -> (N_psi, N_omega).
_N_OUTPUTS = {"velocity_pressure": 3, "psi_omega": 2}


@dataclass(frozen=True)
class Model:
    """What a formulation exposes.

    `has_pressure` is False for the psi-omega system: pressure is absent from
    that system and is recovered post hoc by path integration, so any pressure
    metric taken from `field` there would be meaningless. Callers must check
    the flag rather than trusting the third slot of `field`.
    """

    formulation: str
    field: object          # (params, xy) -> [u, v, p]; p is 0 when has_pressure is False
    velocity: object       # (params, xy) -> [u, v]
    stream: object         # (params, xy) -> [psi, omega], or None
    has_pressure: bool
    lift: object = None


def build_model(config: Config, key):
    """Build parameters and the field function for `config`."""
    fourier_key, net_key = jax.random.split(key)

    n_out = _N_OUTPUTS[config.physics.formulation]

    if config.model.fourier.enabled:
        matrix = init_fourier_matrix(
            fourier_key,
            config.model.fourier.num_features,
            config.model.fourier.bandwidth,
        )
        n_in = 2 * config.model.fourier.num_features
        embed = lambda xi: apply_fourier(matrix, xi)  # noqa: E731
    else:
        n_in = 2
        embed = lambda xi: xi  # noqa: E731

    rwf = dict(
        rwf_enabled=config.model.rwf.enabled,
        rwf_mean=config.model.rwf.mean,
        rwf_stddev=config.model.rwf.stddev,
    )

    if config.model.arch == "mlp":
        params = init_mlp(net_key, n_in, n_out, config.model.width, config.model.depth, **rwf)
        core = apply_mlp
    elif config.model.arch == "modified_mlp":
        params = init_modified_mlp(
            net_key, n_in, n_out, config.model.width, config.model.depth, **rwf
        )
        core = apply_modified_mlp
    else:
        raise ValueError(f"unknown model.arch {config.model.arch!r}")

    def net_apply(p, xi):
        return core(p, embed(xi))

    if config.physics.formulation == "velocity_pressure":
        field_fn = make_velocity_pressure_field(
            net_apply,
            length=config.domain.length,
            inlet_mode=config.bc.inlet,
            lifting_length=config.bc.lifting_length,
            gauge_mode=config.bc.pressure_gauge,
        )
        model = Model(
            formulation="velocity_pressure",
            field=field_fn,
            velocity=lambda p, xy: field_fn(p, xy)[:2],
            stream=None,
            has_pressure=True,
        )
    elif config.physics.formulation == "psi_omega":
        if config.bc.inlet == "lifting":
            raise ValueError(
                "bc.inlet = 'lifting' is not implemented for the psi_omega "
                "formulation: the inlet condition there is a condition on "
                "dpsi/dy, and a lifting construction exact for both u and v at "
                "x* = 0 needs lambda'(0) = 0 as well. Use bc.inlet = 'soft'."
            )
        stream_fn = make_psi_omega_field(
            net_apply,
            length=config.domain.length,
            flux=config.inlet_flux,
            lift=config.model.lift,
        )
        velocity_fn = make_velocity_from_stream(stream_fn)
        model = Model(
            formulation="psi_omega",
            # Pressure is absent from the psi-omega system. The zero here is a
            # placeholder that `has_pressure=False` forbids anyone reporting.
            field=lambda p, xy: jnp.concatenate(
                [velocity_fn(p, xy), jnp.zeros((1,), dtype=jnp.float64)]
            ),
            velocity=velocity_fn,
            stream=stream_fn,
            has_pressure=False,
            lift=config.model.lift,
        )
    else:
        raise ValueError(f"unknown physics.formulation {config.physics.formulation!r}")

    return params, model


def count_parameters(params) -> int:
    return sum(int(leaf.size) for leaf in jax.tree.leaves(params))
