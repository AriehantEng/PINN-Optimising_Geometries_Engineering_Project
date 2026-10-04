"""Network 2, the pressure-drop PINN on the bifurcation (bifurcation/flow.py)."""

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from bifurcation import flow
from bifurcation.geometry import Geometry
from developing_flow import pinn

BASE = {
    "name": "test",
    "seed": 3,
    "physics": {"reynolds": 10.0, "pressure_scaling": "inertial"},
    "geometry": {"split": 0.4, "angle1": 30.0, "angle2": 70.0, "parent_length": 3.0,
                 "daughter_length": 4.0, "min_width": 0.2},
    "outlets": {"split": "prescribed", "outlet1_share": 0.5},
    "inlet": {"profile": "parabola"},
    "model": {"width": 8, "depth": 2, "activation": "tanh", "init_gain": 1.0,
              "input_scale": 2.0, "distance_order": 1},
    "sampling": {"n_interior": 64, "corner_fraction": 0.25, "corner_radius": 0.5,
                 "n_inlet": 16, "n_outlet": 16, "n_validation": 128, "n_flux": 32},
    "weights": {"momentum_x": 1.0, "momentum_y": 2.0, "continuity": 3.0, "inlet": 4.0,
                "outlet": 5.0, "flux": 6.0},
    "training": {
        "adam": {"steps": 0, "learning_rate": 1e-3, "decay_rate": 0.9, "decay_steps": 1000,
                 "resample_every": 10},
        "lbfgs": {"cycles": 0, "iterations_per_cycle": 10, "gtol": 1e-14},
        "lm": {"iterations": 0, "damping": 1e-4, "resample_every": 1, "average": 0},
        "check_every": 10,
    },
    "evaluation": {"max_area": 0.05, "corner_levels": 2, "corner_radius": 0.4, "n_line": 64},
}


def config(**changes):
    document = copy.deepcopy(BASE)
    for dotted, value in changes.items():
        node = document
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        if value is None:
            del node[parts[-1]]
        else:
            node[parts[-1]] = value
    return flow.parse_config(document)


def wall_points(geometry, n=200, seed=0):
    """Points along the walls. On an angled wall a + t (b - a) is on the line
    only to round-off (~1e-16 at these coordinates), and D measures exactly that."""
    return geometry.sample_boundary(np.random.default_rng(seed), "wall", n)[0]


def parent_wall_points(geometry, n=50):
    """Points EXACTLY on the walls: the parent's, y* = -1 and 1."""
    x = np.linspace(0.1, geometry.parent_length - 0.1, n)
    return np.concatenate([np.stack([x, -np.ones(n)], 1), np.stack([x, np.ones(n)], 1)])


#: Round-off in the position of a point sampled on an angled wall.
ON_WALL = 1e-15


# -- configuration ---------------------------------------------------------- #

def test_every_key_is_required_and_inapplicable_ones_are_refused():
    config()
    with pytest.raises(pinn.ConfigError, match="model.input_scale"):
        config(**{"model.input_scale": None})
    with pytest.raises(pinn.ConfigError, match="outlets.outlet1_share"):
        config(**{"outlets.outlet1_share": None})
    with pytest.raises(pinn.ConfigError, match="weights.flux"):
        config(**{"weights.flux": None})
    # a free split takes no share, no flux weight, no flux points
    with pytest.raises(pinn.ConfigError, match="no effect"):
        config(**{"outlets.split": "free"})
    free = config(**{"outlets.split": "free", "outlets.outlet1_share": None,
                     "weights.flux": None, "sampling.n_flux": None})
    assert not free.prescribed and "flux" not in free.weights
    # the parabola needs no Network 1 run; network1 needs one and a station
    with pytest.raises(pinn.ConfigError, match="no effect"):
        config(**{"inlet.run": "x"})
    with pytest.raises(pinn.ConfigError, match="inlet.run"):
        config(**{"inlet.profile": "network1"})
    with pytest.raises(pinn.ConfigError, match="outlet1_share"):
        config(**{"outlets.outlet1_share": 1.0})
    with pytest.raises(pinn.ConfigError, match="distance_order"):
        config(**{"model.distance_order": 3})


def test_an_impossible_geometry_is_refused_before_training():
    from bifurcation.geometry import GeometryError
    with pytest.raises(GeometryError):
        config(**{"geometry.split": 0.05})


# -- the wall distance ------------------------------------------------------ #

@pytest.mark.parametrize("angles", [(30.0, 70.0), (60.0, 60.0), (0.0, 120.0)])
@pytest.mark.parametrize("order", [1, 2])
def test_the_distance_vanishes_on_every_wall_and_only_there(angles, order):
    g = Geometry.from_split(0.4, *angles, 3.0, 4.0, min_width=0.2)
    segments = jnp.asarray(flow.wall_segments(g))
    distance = jax.vmap(lambda xy: flow.wall_distance(xy, segments, order))
    assert np.all(np.asarray(distance(parent_wall_points(g))) == 0.0)
    assert np.max(np.asarray(distance(wall_points(g)))) < ON_WALL
    inside = g.sample_interior(np.random.default_rng(1), 2000)
    assert np.all(np.asarray(distance(inside)) > 0.0)
    # inlet and outlets are not walls: the distance is positive along them
    for kind in ("inlet", "outlet1", "outlet2"):
        nodes, _, _ = g.line_quadrature(kind, 16)
        assert np.all(np.asarray(distance(nodes)) > 0.0)


def test_the_distance_is_normalised_to_first_order():
    """Near a wall, away from its ends, D ~ the distance to it (Sukumar &
    Srivastava's normalisation), so N_u is the wall-normal gradient there."""
    g = Geometry.from_split(0.5, 60.0, 60.0, 3.0, 4.0, min_width=0.2)
    segments = jnp.asarray(flow.wall_segments(g))
    for h in (1e-3, 1e-4, 1e-5):
        d = float(flow.wall_distance(jnp.array([1.5, -1.0 + h]), segments, 1))
        assert abs(d / h - 1.0) < 2.0 * h


def test_the_segment_distance_has_no_cancellation_inside_the_trimming_circle():
    a, b = jnp.array([0.0, 0.0]), jnp.array([2.0, 0.0])
    for h in (1e-3, 1e-6, 1e-9):
        d = float(flow.segment_distance(jnp.array([1.0, h]), a, b))
        assert abs(d / h - 1.0) < 1e-12


# -- the field and its residuals -------------------------------------------- #

def test_no_slip_holds_exactly_for_any_parameters():
    c = config()
    g = c.geometry()
    field = flow.make_field(c, g)
    for seed in range(3):
        params = flow.init_params(c, jax.random.PRNGKey(seed))
        exact = jax.vmap(lambda xy: field(params, xy))(parent_wall_points(g))
        assert np.all(np.asarray(exact[:, :2]) == 0.0)
        angled = jax.vmap(lambda xy: field(params, xy))(wall_points(g))
        assert np.max(np.abs(np.asarray(angled[:, :2]))) < 100 * ON_WALL


def test_point_and_batched_losses_agree():
    """Levenberg-Marquardt sums squares of per-point residuals; Adam and L-BFGS
    use batched derivatives. They must be the same loss."""
    c = config()
    g = c.geometry()
    field = flow.make_field(c, g)
    groups, batched_loss = flow.make_residuals(c, g, field, 4.0 / 3.0)
    params = flow.init_params(c, jax.random.PRNGKey(0))
    params["offset"] = jnp.asarray(0.37)
    flat, unflatten = ravel_pytree(params)
    _, lm_loss = pinn.gauss_newton(groups, unflatten, 16)
    batch = flow.sample_batch(np.random.default_rng(0), c, g, flow.inlet_function(c), 64)
    a, b = lm_loss(flat, batch), float(batched_loss(params, batch))
    assert abs(a - b) < 1e-11 * b


def test_the_outlet_residual_is_the_do_nothing_traction():
    """nu du/dn - p n (+ lam n on outlet 1), checked against a hand computation
    with a field whose gradient is known: Poiseuille, u = 1 - y^2,
    p = -2 nu x + 1/2, which solves Navier-Stokes everywhere."""
    c = config()
    g = c.geometry()
    nu = 1.0 / c.reynolds

    def poiseuille(params, xy):
        return jnp.stack([1.0 - xy[1] ** 2, 0.0 * xy[0], -2.0 * nu * xy[0] + 0.5])

    groups, _ = flow.make_residuals(c, g, poiseuille, 4.0 / 3.0)
    params = {"offset": jnp.asarray(0.25)}
    batch = flow.sample_batch(np.random.default_rng(0), c, g, flow.inlet_function(c), 64)
    found = {id(fn): (fn, pts) for fn, pts in groups(batch)}
    (interior, pts), (inlet, inlet_pts), (outlet1, o1), (outlet2, o2), _ = groups(batch)
    # Poiseuille solves the equations and matches the parabola inlet exactly
    assert np.max(np.abs(jax.vmap(lambda q: interior(params, q, 1))(jnp.asarray(pts)))) < 1e-13
    assert np.max(np.abs(jax.vmap(lambda q: inlet(params, q, 1))(jnp.asarray(inlet_pts)))) < 1e-15
    for fn, points, kind, offset in ((outlet1, o1, "outlet1", 0.25), (outlet2, o2, "outlet2", 0.0)):
        n = flow.outlet_normal(g, kind)
        got = np.asarray(jax.vmap(lambda q: fn(params, q, 1))(jnp.asarray(points)))
        x, y = points[:, 0], points[:, 1]
        p = -2.0 * nu * x + 0.5
        expected = np.stack([nu * (-2.0 * y * n[1]) - p * n[0] + offset * n[0],
                             -p * n[1] + offset * n[1]], axis=1) * math.sqrt(5.0)
        assert np.max(np.abs(got - expected)) < 1e-13
    assert len(found) == 5


def test_the_flux_residual_is_the_outlet_flux_minus_its_share():
    c = config()
    g = c.geometry()
    field = flow.make_field(c, g)
    groups, _ = flow.make_residuals(c, g, field, 4.0 / 3.0)
    params = flow.init_params(c, jax.random.PRNGKey(1))
    *_, (flux, row) = groups(flow.sample_batch(np.random.default_rng(0), c, g,
                                               flow.inlet_function(c), 64))
    nodes, weights, normal = g.line_quadrature("outlet1", 400)
    values = np.asarray(jax.vmap(lambda q: field(params, q))(jnp.asarray(nodes)))
    q1 = float(weights @ (values[:, :2] @ normal))
    got = float(flux(params, jnp.asarray(row[0]), 1)[0])
    assert abs(got - math.sqrt(6.0) * (q1 - 0.5 * 4.0 / 3.0)) < 1e-10


# -- sampling --------------------------------------------------------------- #

def test_every_sampled_point_lies_where_it_belongs():
    c = config()
    g = c.geometry()
    batch = flow.sample_batch(np.random.default_rng(2), c, g, flow.inlet_function(c), 4000)
    assert batch.interior.shape == (4000, 2)
    assert np.all(g.contains(batch.interior))
    corners = g.reentrant_corners()
    near = np.min(np.linalg.norm(batch.interior[:, None] - corners[None], axis=2), axis=1)
    assert np.mean(near < c.corner_radius) > c.corner_fraction    # the corners are favoured
    assert np.all(batch.inlet[:, 0] == 0.0) and np.all(np.abs(batch.inlet[:, 1]) < 1.0)
    assert np.allclose(batch.inlet[:, 2], 1.0 - batch.inlet[:, 1] ** 2, atol=0, rtol=1e-15)
    for kind, points in (("outlet1", batch.outlet1), ("outlet2", batch.outlet2)):
        (a, b), = [(p, q) for p, q, k in g.edges() if k == kind]
        cross = (points[:, 0] - a[0]) * (b[1] - a[1]) - (points[:, 1] - a[1]) * (b[0] - a[0])
        assert np.max(np.abs(cross)) < 1e-12


def test_the_draws_are_a_function_of_the_seed():
    c = config()
    g = c.geometry()
    one = flow.sample_batch(flow._streams(5)[1], c, g, flow.inlet_function(c), 100)
    two = flow.sample_batch(flow._streams(5)[1], c, g, flow.inlet_function(c), 100)
    held = flow.sample_batch(flow._streams(5)[2], c, g, flow.inlet_function(c), 100)
    assert all(np.array_equal(a, b) for a, b in zip(one, two))
    assert not np.array_equal(one.interior, held.interior)


# -- training --------------------------------------------------------------- #

def _tiny(lm_iterations=4):
    return config(**{"training.adam.steps": 20, "training.lbfgs.cycles": 1,
                     "training.lm.iterations": lm_iterations, "training.lm.average": 2})


@pytest.mark.slow
def test_training_reduces_the_held_out_loss():
    params, history, info = flow.train(_tiny(), verbose=False)
    held = [row["heldout_loss"] for row in history]
    assert held[-1] < 0.5 * held[0]
    assert info["selected"] != "initial"


@pytest.mark.slow
def test_a_resumed_run_is_the_uninterrupted_run(tmp_path):
    """Stopped after LM step 2 and resumed, the run must end bitwise where an
    uninterrupted one does: the checkpoint carries the draw generator's state."""
    full_params, full_history, _ = flow.train(_tiny(), verbose=False,
                                              checkpoint=tmp_path / "full.npz")
    flow.train(_tiny(lm_iterations=2), verbose=False, checkpoint=tmp_path / "part.npz")
    # the shorter run's config differs (iterations), so continue its state
    # under the full config by rewriting the fingerprint it was saved with
    import json
    with np.load(tmp_path / "part.npz") as stored:
        arrays = {k: stored[k] for k in stored.files if k != "meta"}
        meta = json.loads(str(stored["meta"]))
    meta["fingerprint"] = flow._fingerprint(_tiny())
    np.savez(tmp_path / "part.npz", meta=np.array(json.dumps(meta)), **arrays)
    resumed_params, resumed_history, _ = flow.train(_tiny(), verbose=False,
                                                    checkpoint=tmp_path / "part.npz", resume=True)
    a = ravel_pytree(full_params)[0]
    b = ravel_pytree(resumed_params)[0]
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert [r["heldout_loss"] for r in full_history] == [r["heldout_loss"] for r in resumed_history]
