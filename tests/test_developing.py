"""The standalone developing-flow PINN (developing_flow/pinn.py).

Structural checks that must hold for ANY network parameters, residuals checked
against hand-derived values (including the inertial terms, which the fully
developed case can never exercise because they vanish there), and one tiny
end-to-end run.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from jax.flatten_util import ravel_pytree

from developing_flow import pinn

CONFIG_PATH = "developing_flow/re10.yaml"


def base_document():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tiny_document():
    document = base_document()
    document["name"] = "tiny"
    document["model"] = {"width": 16, "depth": 2}
    document["sampling"].update(n_interior=256, n_inlet=32, n_outlet=32, n_validation=256)
    document["training"] = {
        "adam": {"steps": 60, "learning_rate": 1.0e-3, "decay_rate": 0.9,
                 "decay_steps": 1000, "resample_every": 20},
        "lbfgs": {"cycles": 2, "iterations_per_cycle": 20, "gtol": 1.0e-14},
        "check_every": 10,
    }
    document["evaluation"] = {"nx": 21, "ny": 11, "profile_stations": [0.0, 1.0],
                              "n_flux_stations": 5}
    return document


@pytest.fixture(scope="module")
def config():
    return pinn.load_config(CONFIG_PATH)


def random_params(config, seed):
    return pinn.init_network(jax.random.PRNGKey(seed), config.width, config.depth)


# -- configuration ---------------------------------------------------------- #


def test_shipped_config_loads(config):
    assert config.reynolds == 10.0
    assert config.inlet_sharpness == 8


def test_missing_key_raises_rather_than_defaulting():
    document = base_document()
    del document["physics"]["reynolds"]
    with pytest.raises(pinn.ConfigError, match="physics.reynolds"):
        pinn.parse_config(document)


def test_unknown_key_is_refused():
    document = base_document()
    document["physics"]["viscosity"] = 1.0e-3
    with pytest.raises(pinn.ConfigError, match="unknown"):
        pinn.parse_config(document)


def test_sharpness_one_is_refused_because_it_is_already_developed():
    document = base_document()
    document["physics"]["inlet_sharpness"] = 1
    with pytest.raises(pinn.ConfigError, match="developed"):
        pinn.parse_config(document)


def test_a_boolean_is_not_accepted_as_an_integer():
    document = base_document()
    document["seed"] = True
    with pytest.raises(pinn.ConfigError, match="seed"):
        pinn.parse_config(document)


# -- the ansatz holds for any parameters ------------------------------------ #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_slip_is_exactly_zero_on_both_walls(config, seed):
    params = random_params(config, seed)
    field = pinn.make_field(config.length)
    x = np.linspace(0.0, config.length, 17)
    for wall in (-1.0, 1.0):
        u, v, _ = pinn.predict(field, params, np.stack([x, np.full_like(x, wall)], axis=-1))
        assert np.all(u == 0.0) and np.all(v == 0.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_pressure_gauge_is_exactly_zero_at_the_outlet(config, seed):
    params = random_params(config, seed)
    field = pinn.make_field(config.length)
    y = np.linspace(-1.0, 1.0, 17)
    _, _, p = pinn.predict(field, params, np.stack([np.full_like(y, config.length), y], axis=-1))
    assert np.all(p == 0.0)


# -- the inlet ---------------------------------------------------------------- #


@pytest.mark.parametrize("m", [2, 4, 8, 16])
def test_every_inlet_carries_the_developed_flux_and_vanishes_at_the_walls(m):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    flux = float(pinn.inlet_profile(nodes, m) @ weights)
    assert abs(flux - pinn.INLET_FLUX) < 1e-13
    assert pinn.inlet_profile(np.array([-1.0, 1.0]), m).tolist() == [0.0, 0.0]


def test_the_inlet_is_not_the_developed_profile():
    y = np.linspace(-1.0, 1.0, 101)
    difference = pinn.inlet_profile(y, 8) - pinn.developed_profile(y, pinn.INLET_FLUX)
    assert np.max(np.abs(difference)) > 0.2


# -- residuals against hand-derived values ------------------------------------ #


@pytest.mark.parametrize("scaling, reynolds", [("inertial", 10.0), ("inertial", 100.0), ("viscous", 1.0)])
def test_residuals_vanish_on_the_exact_developed_solution(scaling, reynolds):
    """u = 1 - y^2, v = 0 and the matching linear pressure satisfy the full
    equations, inertia included (it vanishes identically there)."""
    _, viscous = pinn.scaling_coefficients(reynolds, scaling)

    def exact(xy):
        x, y = xy[0], xy[1]
        return jnp.stack([1.0 - y**2, 0.0 * x, -2.0 * viscous * x + 3.0])

    for point in ([0.3, -0.7], [4.0, 0.1], [9.5, 0.95]):
        d = pinn.point_derivatives(exact, jnp.array(point))
        assert abs(float(pinn.momentum_x(d, reynolds, scaling))) < 1e-13
        assert abs(float(pinn.momentum_y(d, reynolds, scaling))) < 1e-13
        assert abs(float(pinn.continuity(d))) < 1e-13


@pytest.mark.parametrize("scaling", ["inertial", "viscous"])
def test_x_inertia_matches_a_hand_derivation(scaling):
    """u = x (1 - y^2), v = 0, p = 0:
        u u_x + v u_y = x (1 - y^2)^2,  lap u = -2x,  continuity = 1 - y^2."""
    reynolds = 7.0
    inertia, viscous = pinn.scaling_coefficients(reynolds, scaling)

    def field(xy):
        x, y = xy[0], xy[1]
        return jnp.stack([x * (1.0 - y**2), 0.0 * x, 0.0 * x])

    for x, y in ((0.5, 0.2), (2.0, -0.6), (3.3, 0.9)):
        d = pinn.point_derivatives(field, jnp.array([x, y]))
        expected = inertia * x * (1.0 - y**2) ** 2 + viscous * 2.0 * x
        assert abs(float(pinn.momentum_x(d, reynolds, scaling)) - expected) < 1e-12
        assert abs(float(pinn.continuity(d)) - (1.0 - y**2)) < 1e-13


def test_y_inertia_matches_a_hand_derivation():
    """u = y, v = x y, p = 0:  u v_x + v v_y = y^2 + x^2 y,  lap v = 0."""

    def field(xy):
        x, y = xy[0], xy[1]
        return jnp.stack([y, x * y, 0.0 * x])

    for x, y in ((0.5, 0.2), (2.0, -0.6)):
        d = pinn.point_derivatives(field, jnp.array([x, y]))
        assert abs(float(pinn.momentum_y(d, 10.0, "inertial")) - (y**2 + x**2 * y)) < 1e-12


# -- the fast batched derivatives equal the single-point reference ----------- #


def test_batched_derivatives_match_the_point_reference(config):
    params = random_params(config, 3)
    field = pinn.make_field(config.length)
    f = lambda xy: field(params, xy)  # noqa: E731
    points = pinn.sample_batch(jax.random.PRNGKey(5), config, 32).interior
    batched = pinn.batched_derivatives(f, points)
    for i in range(0, 32, 7):
        reference = pinn.point_derivatives(f, points[i])
        for name in pinn.Derivatives._fields:
            a, b = float(getattr(batched, name)[i]), float(getattr(reference, name))
            assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), name


# -- sampling --------------------------------------------------------------- #


def test_samples_lie_in_the_domain_and_are_a_pure_function_of_the_key(config):
    key = jax.random.PRNGKey(11)
    first = pinn.sample_batch(key, config, 1000)
    again = pinn.sample_batch(key, config, 1000)
    other = pinn.sample_batch(jax.random.PRNGKey(12), config, 1000)
    x, y = np.asarray(first.interior[:, 0]), np.asarray(first.interior[:, 1])
    assert x.min() >= 0.0 and x.max() <= config.length
    assert y.min() >= -1.0 and y.max() <= 1.0
    assert np.all(np.asarray(first.inlet[:, 0]) == 0.0)
    assert np.all(np.asarray(first.outlet[:, 0]) == config.length)
    assert np.array_equal(np.asarray(first.interior), np.asarray(again.interior))
    assert not np.array_equal(np.asarray(first.interior), np.asarray(other.interior))


def test_focus_points_crowd_the_inlet(config):
    points = np.asarray(pinn.sample_batch(jax.random.PRNGKey(0), config, 4000).interior)
    near_inlet = np.mean(points[:, 0] < 1.0)
    uniform_share = 1.0 / config.length
    assert near_inlet > 2.0 * uniform_share


def test_held_out_points_are_a_different_draw_from_training(config):
    _, sample_key, heldout_key = pinn.root_keys(config.seed)
    heldout = pinn.sample_batch(heldout_key, config, 256).interior
    _, draw = jax.random.split(sample_key)
    training = pinn.sample_batch(draw, config, 256).interior
    assert not np.any(np.all(np.isclose(np.asarray(heldout)[:, None, :],
                                        np.asarray(training)[None, :, :]), axis=-1))


# -- the loss ----------------------------------------------------------------- #


def test_total_loss_is_the_weighted_sum_of_exactly_the_five_terms():
    terms = {name: jnp.asarray(float(i + 1)) for i, name in enumerate(pinn.LOSS_TERMS)}
    weights = {name: jnp.asarray(1.0) for name in pinn.LOSS_TERMS}
    assert float(pinn.total_loss(terms, weights)) == 15.0
    weights["inlet"] = jnp.asarray(0.0)
    assert float(pinn.total_loss(terms, weights)) == 11.0


# -- end to end ----------------------------------------------------------------- #


@pytest.mark.slow
def test_a_tiny_run_writes_everything_and_reloads_identically(tmp_path):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(tiny_document()), encoding="utf-8")
    directory = pinn.run(config_path, out=tmp_path / "run", verbose=False)

    for name in ("config.yaml", "params.npz", "history.csv", "metrics.json",
                 "profiles.png", "fields.png", "diagnostics.png"):
        assert (directory / name).is_file(), name

    import json
    metrics = json.loads((directory / "metrics.json").read_text())
    for key in ("entrance_length_correlation_flat_inlet", "loss_heldout", "flux_at_outlet"):
        assert np.isfinite(metrics[key])

    model = pinn.load_run(directory)
    config = pinn.parse_config(copy.deepcopy(tiny_document()))
    template = pinn.init_network(pinn.root_keys(config.seed)[0], config.width, config.depth)
    _, unflatten = ravel_pytree(template)
    params = unflatten(jnp.asarray(np.load(directory / "params.npz")["flat"]))
    x, y = np.array([0.0, 0.5, 2.0]), np.array([0.0, 0.3, -0.9])
    expected = pinn.predict(pinn.make_field(config.length), params, np.stack([x, y], axis=-1))
    for a, b in zip(model(x, y), expected):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="points must lie"):
        model(np.array([config.length + 1.0]), np.array([0.0]))
