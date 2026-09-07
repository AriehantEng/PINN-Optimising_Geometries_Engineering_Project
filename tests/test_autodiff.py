"""SPEC test 5 - autodiff derivatives against central finite differences.

The reference is the FOURTH-ORDER central stencil, not the three-point one:

    f'  ~ [ f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h) ] / (12 h)
    f'' ~ [ -f(x-2h) + 16 f(x-h) - 30 f(x) + 16 f(x+h) - f(x+2h) ] / (12 h^2)

Why: the three-point second difference has a roundoff floor of ~4 eps |f| / h^2,
and its truncation error forces h <~ 1e-4, so its best achievable accuracy is
about 1e-7 |f|. On fields of magnitude ~5 that floor sits ABOVE the 1e-7
target, and a test written that way measures the finite-difference method
rather than the autodiff it is supposed to be checking. The five-point stencil
pushes truncation to O(h^4), which allows h = 1e-3, where the measured floor is
~2e-9 across every architecture in this suite -- about 50x below the target, so
a genuine derivative error is caught with margin.

Both stencils are central differences; only the order changed. The 1e-7
tolerance is the SPEC's and is not adjusted anywhere in this file. Step size
h = 1e-3 sits in the flat minimum of the measured error curve (error stays
below 1e-8 for h in [1e-3, 3e-3] on every architecture tested).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import parse_config
from src.geometry.sampling import sample_interior
from src.geometry.domain import Channel
from src.models.build import build_model
from src.physics import residuals
from src.physics.derivatives import batched_field_derivatives

H_FIRST = 1e-3
H_SECOND = 1e-3
FD_ATOL = 1e-7  # SPEC test 5. Never widen this.


def _base_config(**overrides):
    document = {
        "name": "fd-check",
        "seed": 3,
        "case": "A",
        "domain": {"length": 20.0},
        "physics": {
            "reynolds": 10.0,
            "pressure_scaling": "inertial",
            "formulation": "velocity_pressure",
        },
        "model": {
            "arch": "mlp",
            "width": 32,
            "depth": 3,
            "fourier": {"enabled": False},
            "rwf": {"enabled": False},
        },
        "bc": {"inlet": "soft", "outlet_velocity": "soft", "pressure_gauge": "soft"},
        "sampling": {
            "n_interior": 64,
            "n_inlet": 8,
            "n_outlet": 8,
            "resample_every": 10,
            "seed": 1,
        },
        "weights": {
            "scheme": "fixed",
            "values": {
                "momentum_x": 1.0,
                "momentum_y": 1.0,
                "continuity": 1.0,
                "inlet": 1.0,
                "outlet_velocity": 1.0,
                "pressure_gauge": 1.0,
            },
        },
        "training": {
            "adam": {"steps": 1, "lr_init": 1.0e-3, "decay_rate": 0.9, "decay_steps": 10},
            "lbfgs": {"enabled": False},
        },
        "evaluation": {"nx": 11, "ny": 11},
    }
    for path, value in overrides.items():
        node = document
        *parents, leaf = path.split(".")
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    return parse_config(document)


def _shifted(fn, points, axis, step, multiple):
    offset = np.zeros(2)
    offset[axis] = multiple * step
    return np.asarray(jax.vmap(fn)(jnp.asarray(points + offset)))


def _central_difference(fn, points, axis, step):
    """Fourth-order central first difference."""
    return (
        _shifted(fn, points, axis, step, -2)
        - 8.0 * _shifted(fn, points, axis, step, -1)
        + 8.0 * _shifted(fn, points, axis, step, +1)
        - _shifted(fn, points, axis, step, +2)
    ) / (12.0 * step)


def _central_second_difference(fn, points, axis, step):
    """Fourth-order central second difference."""
    return (
        -_shifted(fn, points, axis, step, -2)
        + 16.0 * _shifted(fn, points, axis, step, -1)
        - 30.0 * _shifted(fn, points, axis, step, 0)
        + 16.0 * _shifted(fn, points, axis, step, +1)
        - _shifted(fn, points, axis, step, +2)
    ) / (12.0 * step**2)


FORMULATIONS = [
    dict(),
    dict(**{"model.arch": "modified_mlp"}),
    dict(**{"model.fourier": {"enabled": True, "num_features": 16, "bandwidth": 1.0}}),
    dict(**{"model.rwf": {"enabled": True, "mean": 0.5, "stddev": 0.1}}),
    dict(
        **{
            "physics.formulation": "stream_function",
            "model.stream_base": "sine",
            "weights.values": {
                "momentum_x": 1.0,
                "momentum_y": 1.0,
                "inlet": 1.0,
                "outlet_velocity": 1.0,
                "pressure_gauge": 1.0,
            },
        }
    ),
]

FORMULATION_IDS = ["mlp", "modified_mlp", "fourier", "rwf", "stream_function"]


@pytest.mark.parametrize("overrides", FORMULATIONS, ids=FORMULATION_IDS)
def test_autodiff_matches_central_differences(overrides):
    """SPEC test 5, on every architecture and both formulations."""
    config = _base_config(**overrides)
    params, field_fn = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn = jax.jit(lambda xy: field_fn(params, xy))

    # Interior points, kept a stencil width clear of the boundaries.
    channel = Channel(config.domain.length)
    points = np.array(sample_interior(jax.random.PRNGKey(11), channel, 128))
    points[:, 0] = np.clip(points[:, 0], 0.05, config.domain.length - 0.05)
    points[:, 1] = np.clip(points[:, 1], -0.95, 0.95)

    d = batched_field_derivatives(point_fn, jnp.asarray(points))
    autodiff = {
        ("u", "x"): d.u_x, ("u", "y"): d.u_y,
        ("v", "x"): d.v_x, ("v", "y"): d.v_y,
        ("p", "x"): d.p_x, ("p", "y"): d.p_y,
    }
    component = {"u": 0, "v": 1, "p": 2}

    for (name, axis_name), value in autodiff.items():
        axis = 0 if axis_name == "x" else 1
        fd = _central_difference(point_fn, points, axis, H_FIRST)[:, component[name]]
        np.testing.assert_allclose(
            np.asarray(value), fd, rtol=0.0, atol=FD_ATOL,
            err_msg=f"d{name}/d{axis_name} disagrees with central differences",
        )

    for name, axis_name, value in (
        ("u", "x", d.u_xx), ("u", "y", d.u_yy),
        ("v", "x", d.v_xx), ("v", "y", d.v_yy),
    ):
        axis = 0 if axis_name == "x" else 1
        fd = _central_second_difference(point_fn, points, axis, H_SECOND)[
            :, component[name]
        ]
        np.testing.assert_allclose(
            np.asarray(value), fd, rtol=0.0, atol=FD_ATOL,
            err_msg=f"d2{name}/d{axis_name}2 disagrees with central differences",
        )


@pytest.mark.parametrize("overrides", FORMULATIONS, ids=FORMULATION_IDS)
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_residual_from_finite_differences_matches_autodiff_residual(
    overrides, pressure_scaling
):
    """The assembled residual, not just the individual derivatives.

    A pair of compensating sign errors would pass the derivative-by-derivative
    check above only if it also passed this one, and vice versa.
    """
    config = _base_config(**overrides, **{"physics.pressure_scaling": pressure_scaling})
    params, field_fn = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn = jax.jit(lambda xy: field_fn(params, xy))

    channel = Channel(config.domain.length)
    points = np.array(sample_interior(jax.random.PRNGKey(12), channel, 96))
    points[:, 0] = np.clip(points[:, 0], 0.05, config.domain.length - 0.05)
    points[:, 1] = np.clip(points[:, 1], -0.95, 0.95)

    d_auto = batched_field_derivatives(point_fn, jnp.asarray(points))

    values = np.asarray(jax.vmap(point_fn)(jnp.asarray(points)))
    first = {
        axis: _central_difference(point_fn, points, axis, H_FIRST) for axis in (0, 1)
    }
    second = {
        axis: _central_second_difference(point_fn, points, axis, H_SECOND)
        for axis in (0, 1)
    }
    d_fd = residuals.Derivatives(
        u=values[:, 0], v=values[:, 1], p=values[:, 2],
        u_x=first[0][:, 0], u_y=first[1][:, 0],
        v_x=first[0][:, 1], v_y=first[1][:, 1],
        p_x=first[0][:, 2], p_y=first[1][:, 2],
        u_xx=second[0][:, 0], u_yy=second[1][:, 0],
        v_xx=second[0][:, 1], v_yy=second[1][:, 1],
    )

    reynolds = config.physics.reynolds
    for auto, fd in zip(
        residuals.navier_stokes(d_auto, reynolds, pressure_scaling),
        residuals.navier_stokes(d_fd, reynolds, pressure_scaling),
    ):
        np.testing.assert_allclose(np.asarray(auto), np.asarray(fd), rtol=0.0, atol=FD_ATOL)


def test_finite_difference_check_would_fail_on_a_wrong_derivative():
    """A negative control: the check above must actually be able to fail.

    Without this, a tolerance loose enough to accept anything would look like a
    passing test.
    """
    config = _base_config()
    params, field_fn = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn = jax.jit(lambda xy: field_fn(params, xy))
    points = np.array(
        sample_interior(jax.random.PRNGKey(13), Channel(config.domain.length), 64)
    )
    points[:, 1] = np.clip(points[:, 1], -0.95, 0.95)

    d = batched_field_derivatives(point_fn, jnp.asarray(points))
    wrong = np.asarray(d.u_x) * (1.0 + 1e-5)  # a 10 ppm error in one derivative
    fd = _central_difference(point_fn, points, 0, H_FIRST)[:, 0]

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(wrong, fd, rtol=0.0, atol=FD_ATOL)


@pytest.mark.parametrize("overrides", FORMULATIONS, ids=FORMULATION_IDS)
def test_batched_derivatives_match_single_point(overrides):
    """The fast batched path must agree with the reference single-point path.

    `batched_field_derivatives` differentiates the batched field function along
    constant tangent directions; `field_derivatives` nests `jacfwd` at one
    point. They are the same mathematics by different routes, and the batched
    one is 2.4x faster, so it is the one training uses -- which makes this the
    test that stops it drifting.
    """
    from src.physics.derivatives import field_derivatives

    config = _base_config(**overrides)
    params, field_fn = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn = lambda xy: field_fn(params, xy)  # noqa: E731

    points = sample_interior(jax.random.PRNGKey(21), Channel(config.domain.length), 96)

    batched = batched_field_derivatives(point_fn, points)
    reference = jax.vmap(lambda xy: field_derivatives(point_fn, xy))(points)

    for name, fast, slow in zip(batched._fields, batched, reference):
        np.testing.assert_allclose(
            np.asarray(fast), np.asarray(slow), rtol=0.0, atol=1e-14,
            err_msg=f"batched and single-point derivative disagree on {name}",
        )
