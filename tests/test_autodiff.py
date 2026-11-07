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
    return parse_config(_base_document(**overrides))


def _base_document(**overrides):
    """The raw config dict, before parsing.

    Split out from `_base_config` so tests that need to DELETE or corrupt a key
    can do so: `_base_config` returns a frozen Config, which cannot be edited.
    """
    document = {
        "name": "fd-check",
        "seed": 3,
        "case": "A",
        "domain": {"alpha": 1.5, "outlet_margin": 5.0},
        "correlation": {
            "c0": 0.3155, "c1": 0.01105, "n": 1.6,
            "citation": "Durst et al. (2005) plane-channel fit, converted to "
                        "hydraulic-diameter units; see experiments/*.yaml",
        },
        "physics": {
            "reynolds": 10.0,
            "pressure_scaling": "inertial",
            "formulation": "velocity_pressure",
            "convective_coefficient": 1.0,
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
            "n_validation": 64,
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
            "select": "best_validation",
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
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return document


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
            "physics.formulation": "psi_omega",
            "model.lift": "sinusoidal",
            "bc.pressure_gauge": None,  # removed below: no pressure in the system
            "weights.values": {
                "vorticity": 1.0,
                "vorticity_transport": 1.0,
                "inlet": 1.0,
                "outlet_velocity": 1.0,
            },
        }
    ),
]

FORMULATION_IDS = ["mlp", "modified_mlp", "fourier", "rwf", "psi_omega"]


def _under_test(config, params, model):
    """The vector function whose derivatives the FD check compares against.

    For psi-omega that is (psi, omega): its `field` carries a placeholder zero
    in the pressure slot, so differentiating that would be a vacuous check.
    """
    if config.physics.formulation == "psi_omega":
        return jax.jit(lambda xy: model.stream(params, xy)), ("psi", "omega")
    return jax.jit(lambda xy: model.field(params, xy)), ("u", "v", "p")


@pytest.mark.parametrize("overrides", FORMULATIONS, ids=FORMULATION_IDS)
def test_autodiff_matches_central_differences(overrides):
    """SPEC test 5, on every architecture and both formulations."""
    config = _base_config(**overrides)
    params, model = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn, names = _under_test(config, params, model)

    # Interior points, kept a stencil width clear of the boundaries.
    channel = Channel(config.domain.length)
    points = np.array(sample_interior(jax.random.PRNGKey(11), channel, 128))
    points[:, 0] = np.clip(points[:, 0], 0.05, config.domain.length - 0.05)
    points[:, 1] = np.clip(points[:, 1], -0.95, 0.95)

    from src.physics.derivatives import batched_directional_derivatives

    value, d_dx, d_dy, d2_dx2, d2_dy2 = batched_directional_derivatives(
        point_fn, jnp.asarray(points)
    )

    for index, name in enumerate(names):
        for axis, axis_name, autodiff in (
            (0, "x", d_dx[:, index]), (1, "y", d_dy[:, index])
        ):
            fd = _central_difference(point_fn, points, axis, H_FIRST)[:, index]
            np.testing.assert_allclose(
                np.asarray(autodiff), fd, rtol=0.0, atol=FD_ATOL,
                err_msg=f"d{name}/d{axis_name} disagrees with central differences",
            )
        for axis, axis_name, autodiff in (
            (0, "x", d2_dx2[:, index]), (1, "y", d2_dy2[:, index])
        ):
            fd = _central_second_difference(point_fn, points, axis, H_SECOND)[:, index]
            np.testing.assert_allclose(
                np.asarray(autodiff), fd, rtol=0.0, atol=FD_ATOL,
                err_msg=f"d2{name}/d{axis_name}2 disagrees with central differences",
            )


@pytest.mark.parametrize("overrides", FORMULATIONS[:4], ids=FORMULATION_IDS[:4])
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_residual_from_finite_differences_matches_autodiff_residual(
    overrides, pressure_scaling
):
    """The assembled residual, not just the individual derivatives.

    A pair of compensating sign errors would pass the derivative-by-derivative
    check above only if it also passed this one, and vice versa.
    """
    config = _base_config(**overrides, **{"physics.pressure_scaling": pressure_scaling})
    params, model = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn = jax.jit(lambda xy: model.field(params, xy))

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
    params, model = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn = jax.jit(lambda xy: model.field(params, xy))
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
    params, model = build_model(config, jax.random.PRNGKey(config.seed))
    point_fn, _ = _under_test(config, params, model)

    points = sample_interior(jax.random.PRNGKey(21), Channel(config.domain.length), 96)

    batched = batched_field_derivatives(point_fn, points)
    reference = jax.vmap(lambda xy: field_derivatives(point_fn, xy))(points)

    for name, fast, slow in zip(batched._fields, batched, reference):
        np.testing.assert_allclose(
            np.asarray(fast), np.asarray(slow), rtol=0.0, atol=1e-14,
            err_msg=f"batched and single-point derivative disagree on {name}",
        )


# --------------------------------------------------------------------------- #
# Inertia ablation switch
# --------------------------------------------------------------------------- #


def test_convective_coefficient_zero_removes_the_inertial_terms():
    """coefficient 0 must leave exactly the Stokes residual, 1 the full NS one.

    The switch exists so that "the inertial terms are included" is a
    measurement rather than an assertion. If it does not actually remove
    u.grad(u), every Stokes-vs-Navier-Stokes comparison built on it is
    meaningless.
    """
    import jax
    import jax.numpy as jnp

    from src.physics import residuals
    from src.physics.derivatives import batched_field_derivatives

    def field(xy):
        x, y = xy[..., 0], xy[..., 1]
        return jnp.stack(
            [jnp.sin(x) * (1 - y**2), 0.3 * jnp.cos(x) * y, 0.2 * x - 0.1 * y], -1
        )

    points = jax.random.uniform(
        jax.random.PRNGKey(0), (64, 2), minval=-0.9, maxval=0.9, dtype=jnp.float64
    )
    d = batched_field_derivatives(field, points)

    full = residuals.momentum_x(d, 10.0, "inertial", 1.0)
    stokes = residuals.momentum_x(d, 10.0, "inertial", 0.0)
    convective = residuals.convective_x(d.u, d.v, d.u_x, d.u_y)

    # full - stokes must be exactly the convective term it claims to switch.
    np.testing.assert_allclose(np.asarray(full - stokes),
                               np.asarray(convective), rtol=0, atol=1e-14)
    # and the terms must be large enough here for that to mean something.
    assert np.max(np.abs(np.asarray(convective))) > 1e-3

    full_y = residuals.momentum_y(d, 10.0, "inertial", 1.0)
    stokes_y = residuals.momentum_y(d, 10.0, "inertial", 0.0)
    np.testing.assert_allclose(
        np.asarray(full_y - stokes_y),
        np.asarray(residuals.convective_y(d.u, d.v, d.v_x, d.v_y)),
        rtol=0, atol=1e-14,
    )


def test_the_switch_is_a_no_op_on_the_case_a_solution():
    """Case A annihilates u.grad(u) exactly, so the switch must do nothing.

    This is why no Case A run can demonstrate inertia, and it is the cheapest
    possible check that the coefficient is wired to the right term: on the
    exact Poiseuille field u = 1 - y^2, v = 0, both settings must give an
    identical residual.
    """
    import jax.numpy as jnp

    from src.physics import residuals
    from src.physics.derivatives import batched_field_derivatives

    def exact(xy):
        x, y = xy[..., 0], xy[..., 1]
        return jnp.stack([1.0 - y**2, jnp.zeros_like(y), -0.2 * x], -1)

    points = jnp.array(
        [[xx, yy] for xx in np.linspace(0, 5, 9) for yy in np.linspace(-0.95, 0.95, 9)]
    )
    d = batched_field_derivatives(exact, points)
    for name in ("momentum_x", "momentum_y"):
        fn = getattr(residuals, name)
        np.testing.assert_allclose(
            np.asarray(fn(d, 10.0, "inertial", 1.0)),
            np.asarray(fn(d, 10.0, "inertial", 0.0)),
            rtol=0, atol=1e-15, err_msg=f"{name} changed on the Case A solution",
        )


def test_convective_coefficient_is_a_required_config_key():
    """No silent default: a run's equations must be readable off its config."""
    import copy

    import pytest

    from src.config import ConfigError, parse_config

    document = copy.deepcopy(_base_document())
    del document["physics"]["convective_coefficient"]
    with pytest.raises(ConfigError, match="convective_coefficient"):
        parse_config(document)


def test_convective_coefficient_outside_zero_to_one_is_rejected():
    import copy

    import pytest

    from src.config import ConfigError, parse_config

    for bad in (-0.5, 1.5):
        document = copy.deepcopy(_base_document())
        document["physics"]["convective_coefficient"] = bad
        with pytest.raises(ConfigError, match=r"\[0, 1\]"):
            parse_config(document)
