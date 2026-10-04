"""Network 3 (bifurcation/geometric.py): the rules that hold for any parameters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from bifurcation import geometric as n3
from bifurcation.geometry import split_bounds


@pytest.fixture(scope="module")
def setup():
    config = n3.parse_config(yaml.safe_load(open("bifurcation/n3_re10_shares.yaml")))
    study_config, data = n3.load_study(n3.ROOT / config["data"])
    return config, study_config, data, n3.make_model(config, study_config)


def random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 179, n), rng.uniform(0, 179, n), rng.uniform(0.001, 0.999, n)


def test_swapping_the_daughters_mirrors_the_split_exactly(setup):
    config, _, _, predict = setup
    a1, a2, q = random_inputs(200)
    for seed in range(3):
        params = n3.init_params(config, jax.random.PRNGKey(seed))
        s, d = predict(params, a1, a2, q)
        sm, dm = predict(params, a2, a1, 1.0 - q)
        assert np.max(np.abs(np.asarray(s) + np.asarray(sm) - 1.0)) < 1e-14
        assert np.max(np.abs(np.asarray(d) / np.asarray(dm) - 1.0)) < 1e-13


def test_no_parameters_can_take_a_width_below_the_minimum(setup):
    config, study_config, _, predict = setup
    lo, hi = split_bounds(study_config["geometry"]["min_width"])
    a1, a2, q = random_inputs(500, seed=1)
    for seed in range(3):
        params = n3.init_params(config, jax.random.PRNGKey(seed))
        params = jax.tree_util.tree_map(lambda p: 50.0 * p, params)    # violently large
        s, _ = predict(params, a1, a2, q)
        assert np.all(np.asarray(s) >= lo) and np.all(np.asarray(s) <= hi)


def test_with_no_correction_it_is_the_friction_law(setup):
    config, study_config, data, predict = setup
    params = n3.init_params(config, jax.random.PRNGKey(0))
    params["output"] = jax.tree_util.tree_map(jnp.zeros_like, params["output"])
    s, d = predict(params, data["a1"], data["a2"], data["q"])
    assert np.allclose(np.asarray(s), np.asarray(n3.friction_split(jnp.asarray(data["q"]))),
                       rtol=0, atol=1e-13)
    assert np.allclose(np.asarray(d), np.asarray(n3.friction_drop(jnp.asarray(data["q"]),
                                                                    study_config)), rtol=1e-14)


def test_the_friction_drop_is_parallel_poiseuille():
    """Equal shares and widths at Re 10: daughters 12 nu L q / w^3 = 6.4, parent 0.6."""
    study = {"physics": {"reynolds": 10.0, "pressure_scaling": "inertial"},
             "geometry": {"daughter_length": 8.0, "parent_length": 3.0}}
    assert abs(float(n3.friction_drop(0.5, study)) - 7.0) < 1e-12


def test_the_study_data_are_what_the_report_says(setup):
    _, _, data, _ = setup
    assert len(data["q"]) == 66
    assert len({(a, b) for a, b in zip(data["a1"], data["a2"])}) == 6
