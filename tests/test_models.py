"""The boundary-condition ansatz must hold for ARBITRARY parameters.

These are structural tests, not accuracy tests: they check properties that are
true by construction at initialisation and must stay true for every parameter
value the optimiser can reach. If one of these fails, no amount of training
will fix the resulting solution.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import parse_config
from src.geometry.domain import WALL, Channel, dphi_dy, normalise, phi
from src.geometry.sampling import evaluation_grid, sample_all, station_grid
from src.models.build import build_model, count_parameters
from src.physics import residuals
from src.physics.derivatives import batched_field_derivatives
from tests.test_autodiff import _base_config

ARCHS = [
    ("mlp", {}),
    ("modified_mlp", {"model.arch": "modified_mlp"}),
    ("fourier", {"model.fourier": {"enabled": True, "num_features": 16, "bandwidth": 2.0}}),
    ("rwf", {"model.rwf": {"enabled": True, "mean": 0.5, "stddev": 0.1}}),
]

STREAM_WEIGHTS = {
    "momentum_x": 1.0,
    "momentum_y": 1.0,
    "inlet": 1.0,
    "outlet_velocity": 1.0,
    "pressure_gauge": 1.0,
}


def _wall_points(length, n=64):
    y = np.concatenate([np.full(n, WALL), np.full(n, -WALL)])
    x = np.concatenate([np.linspace(0.0, length, n)] * 2)
    return jnp.asarray(np.stack([x, y], axis=-1))


# --------------------------------------------------------------------------- #
# phi
# --------------------------------------------------------------------------- #


def test_phi_vanishes_exactly_on_walls():
    assert float(phi(1.0)) == 0.0
    assert float(phi(-1.0)) == 0.0
    assert float(phi(0.0)) == 1.0


def test_phi_is_positive_inside_and_matches_its_derivative():
    y = np.linspace(-0.999, 0.999, 501)
    assert np.all(np.asarray(phi(y)) > 0.0)
    autodiff = jax.vmap(jax.grad(lambda t: phi(t)))(jnp.asarray(y))
    np.testing.assert_allclose(np.asarray(autodiff), np.asarray(dphi_dy(y)), atol=1e-15)


def test_normalise_maps_domain_corners_to_unit_square():
    length = 20.0
    corners = jnp.array([[0.0, -1.0], [length, -1.0], [0.0, 1.0], [length, 1.0]])
    expected = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(np.asarray(normalise(corners, length)), expected)


# --------------------------------------------------------------------------- #
# Hard wall boundary condition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,overrides", ARCHS, ids=[a[0] for a in ARCHS])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_slip_is_exactly_zero_on_both_walls(name, overrides, seed):
    """Not 'small'. Exactly zero, bitwise, for arbitrary parameters."""
    config = _base_config(**overrides, seed=seed)
    params, field_fn = build_model(config, jax.random.PRNGKey(seed))
    values = jax.vmap(lambda xy: field_fn(params, xy))(
        _wall_points(config.domain.length)
    )
    assert np.all(np.asarray(values[:, 0]) == 0.0), "u is not identically zero on walls"
    assert np.all(np.asarray(values[:, 1]) == 0.0), "v is not identically zero on walls"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_function_no_slip_is_exactly_zero_on_both_walls(seed):
    config = _base_config(
        seed=seed,
        **{
            "physics.formulation": "stream_function",
            "model.stream_base": "sine",
            "weights.values": STREAM_WEIGHTS,
        },
    )
    params, field_fn = build_model(config, jax.random.PRNGKey(seed))
    values = jax.vmap(lambda xy: field_fn(params, xy))(
        _wall_points(config.domain.length)
    )
    # psi_base has zero slope at the walls and phi^2 kills both derivatives of
    # the network term, so u and v vanish -- to rounding, since here they are
    # computed by differentiation rather than by multiplication.
    assert float(jnp.max(jnp.abs(values[:, 0]))) < 1e-15
    assert float(jnp.max(jnp.abs(values[:, 1]))) < 1e-15


# --------------------------------------------------------------------------- #
# Inlet lifting and pressure gauge
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lifting_inlet_reproduces_the_case_a_profile_exactly(seed):
    config = _base_config(
        seed=seed, **{"bc.inlet": "lifting", "bc.lifting_length": 1.0,
                      "weights.values": {"momentum_x": 1.0, "momentum_y": 1.0,
                                         "continuity": 1.0, "outlet_velocity": 1.0,
                                         "pressure_gauge": 1.0}}
    )
    assert "inlet" not in config.loss_terms, "lifting must remove the inlet loss term"

    params, field_fn = build_model(config, jax.random.PRNGKey(seed))
    y = jnp.linspace(-1.0, 1.0, 129)
    points = jnp.stack([jnp.zeros_like(y), y], axis=-1)
    values = jax.vmap(lambda xy: field_fn(params, xy))(points)

    np.testing.assert_allclose(np.asarray(values[:, 0]), 1.0 - np.asarray(y) ** 2,
                               rtol=0.0, atol=0.0)
    assert np.all(np.asarray(values[:, 1]) == 0.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_hard_pressure_gauge_is_exactly_zero_on_the_outlet(seed):
    config = _base_config(
        seed=seed,
        **{"bc.pressure_gauge": "hard",
           "weights.values": {"momentum_x": 1.0, "momentum_y": 1.0,
                              "continuity": 1.0, "inlet": 1.0,
                              "outlet_velocity": 1.0}},
    )
    assert "pressure_gauge" not in config.loss_terms

    params, field_fn = build_model(config, jax.random.PRNGKey(seed))
    y = jnp.linspace(-1.0, 1.0, 65)
    points = jnp.stack([jnp.full_like(y, config.domain.length), y], axis=-1)
    values = jax.vmap(lambda xy: field_fn(params, xy))(points)
    assert np.all(np.asarray(values[:, 2]) == 0.0)


# --------------------------------------------------------------------------- #
# Stream function
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("base", ["sine", "cubic"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_function_satisfies_continuity_identically(base, seed):
    """The whole point of the stream-function formulation.

    div u must be zero to rounding for ARBITRARY parameters, which is why the
    continuity loss term does not exist in that configuration.
    """
    config = _base_config(
        seed=seed,
        **{"physics.formulation": "stream_function", "model.stream_base": base,
           "weights.values": STREAM_WEIGHTS},
    )
    assert "continuity" not in config.loss_terms

    params, field_fn = build_model(config, jax.random.PRNGKey(seed))
    points, _, _ = evaluation_grid(Channel(config.domain.length), 41, 21)
    d = batched_field_derivatives(lambda xy: field_fn(params, xy), points)
    assert float(jnp.max(jnp.abs(residuals.continuity(d)))) < 1e-13


@pytest.mark.parametrize("base", ["sine", "cubic"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_function_carries_the_configured_flux_at_every_station(base, seed):
    """psi(+1) - psi(-1) = flux, so mass is conserved by construction.

    This is the property a psi = phi^2 * N ansatz cannot have: it would force
    equal psi on both walls and therefore zero net throughflow.
    """
    config = _base_config(
        seed=seed,
        **{"physics.formulation": "stream_function", "model.stream_base": base,
           "weights.values": STREAM_WEIGHTS},
    )
    params, field_fn = build_model(config, jax.random.PRNGKey(seed))

    points, x_stations, y = station_grid(Channel(config.domain.length), 10, 4001)
    u = jax.vmap(jax.vmap(lambda xy: field_fn(params, xy)[0]))(points)
    flux = np.trapezoid(np.asarray(u), np.asarray(y), axis=1)

    np.testing.assert_allclose(flux, config.inlet_flux, rtol=1e-6, atol=0.0)


def test_stream_function_rejects_lifting_inlet():
    """No silent fallbacks: an unimplemented combination raises."""
    config = _base_config(
        **{"physics.formulation": "stream_function", "model.stream_base": "sine",
           "bc.inlet": "lifting", "bc.lifting_length": 1.0,
           "weights.values": {"momentum_x": 1.0, "momentum_y": 1.0,
                              "outlet_velocity": 1.0, "pressure_gauge": 1.0}},
    )
    with pytest.raises(ValueError, match="not implemented for the"):
        build_model(config, jax.random.PRNGKey(0))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_sampling_is_a_pure_function_of_its_key():
    channel = Channel(20.0)
    a = sample_all(jax.random.PRNGKey(7), channel, 32, 8, 8)
    b = sample_all(jax.random.PRNGKey(7), channel, 32, 8, 8)
    c = sample_all(jax.random.PRNGKey(8), channel, 32, 8, 8)
    np.testing.assert_array_equal(np.asarray(a.interior), np.asarray(b.interior))
    assert not np.array_equal(np.asarray(a.interior), np.asarray(c.interior))


def test_samples_lie_in_the_domain():
    channel = Channel(20.0)
    points = sample_all(jax.random.PRNGKey(9), channel, 4096, 256, 256)
    interior = np.asarray(points.interior)
    assert interior[:, 0].min() >= 0.0 and interior[:, 0].max() <= 20.0
    assert interior[:, 1].min() >= -1.0 and interior[:, 1].max() <= 1.0
    assert np.all(np.asarray(points.inlet)[:, 0] == 0.0)
    assert np.all(np.asarray(points.outlet)[:, 0] == 20.0)


def test_evaluation_grid_is_not_the_training_set():
    """Guards SPEC section 9: error must be measured on a fixed grid."""
    channel = Channel(20.0)
    grid, _, _ = evaluation_grid(channel, 41, 21)
    training = sample_all(jax.random.PRNGKey(0), channel, 4096, 128, 128).interior
    grid_set = {tuple(map(float, row)) for row in np.asarray(grid)}
    train_set = {tuple(map(float, row)) for row in np.asarray(training)}
    assert grid_set.isdisjoint(train_set)


def test_parameter_count_is_reported():
    config = _base_config()
    params, _ = build_model(config, jax.random.PRNGKey(0))
    assert count_parameters(params) > 0
