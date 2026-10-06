"""Network 2, parametric: the bifurcation PINN with the width split as an input
(bifurcation/parametric.py)."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from bifurcation import flow
from bifurcation import parametric as P
from developing_flow import pinn

BASE = {
    "name": "test",
    "seed": 5,
    "physics": {"reynolds": 10.0, "pressure_scaling": "inertial"},
    "geometry": {"angle1": 30.0, "angle2": 70.0, "parent_length": 3.0, "daughter_length": 4.0,
                 "min_width": 0.2},
    "design": {"split_min": 0.35, "split_max": 0.6, "outlet1_share": 0.5},
    "inlet": {"profile": "parabola"},
    "model": {"width": 8, "depth": 2, "activation": "tanh", "init_gain": 1.0,
              "input_scale": 2.0, "distance_order": 1, "wall_extension": 0.5,
              "blend_exponent": 2.0, "offset_degree": 3},
    "sampling": {"designs": 3, "points_per_design": 32, "corner_fraction": 0.25,
                 "corner_radius": 0.5, "corner_exclusion": 0.05, "inlet_per_design": 8,
                 "outlet_per_design": 8, "validation_designs": 4, "n_flux": 32},
    "weights": {"momentum_x": 1.0, "momentum_y": 2.0, "inlet": 3.0, "outlet": 4.0,
                "corner_radius": 0.3},
    "training": {
        "adam": {"steps": 0, "learning_rate": 1e-3, "decay_rate": 0.9, "decay_steps": 1000,
                 "resample_every": 10},
        "lbfgs": {"cycles": 0, "iterations_per_cycle": 10, "gtol": 1e-14},
        "lm": {"iterations": 0, "damping": 1e-4, "resample_every": 1, "average": 0},
        "check_every": 10,
    },
    "evaluation": {"max_area": 0.05, "corner_levels": 2, "corner_radius": 0.4, "n_line": 64,
                   "splits": [0.35, 0.45, 0.6], "search_xatol": 0.01, "curve_points": 5},
}

Q_IN = 4.0 / 3.0                     # the parabola's flux
SPLITS = (0.35, 0.4123, 0.5, 0.6)    # both ends of the range and two inside it


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
    return P.parse_config(document)


def setup(c=None):
    c = c or config()
    st = P.structure(c)
    return c, st, P.make_field(c, st, Q_IN), P.init_params(c, jax.random.PRNGKey(1))


# -- configuration ---------------------------------------------------------- #

def test_every_key_is_required():
    config()
    for dotted in ("design.split_min", "model.offset_degree", "sampling.designs",
                   "evaluation.splits", "weights.corner_radius", "training.lm.average"):
        with pytest.raises(pinn.ConfigError, match=dotted.split(".")[-1]):
            config(**{dotted: None})


def test_an_impossible_range_is_refused_before_training():
    with pytest.raises(pinn.ConfigError, match="split_min"):
        config(**{"design.split_min": 0.6, "design.split_max": 0.35})
    with pytest.raises(pinn.ConfigError, match="split_min"):
        config(**{"design.split_max": 0.95})              # daughter 2 narrower than min_width
    with pytest.raises(pinn.ConfigError, match="outside the design range"):
        config(**{"evaluation.splits": [0.3, 0.5]})
    with pytest.raises(pinn.ConfigError, match="no effect"):
        config(**{"inlet.station": 1.0})                  # network1 keys under a parabola


def test_a_range_one_construction_cannot_serve_is_refused():
    """At a1 + a2 = 180 the septum is a face, not a tip: another polygon."""
    with pytest.raises(pinn.ConfigError, match="septum-tip"):
        config(**{"geometry.angle1": 90.0, "geometry.angle2": 90.0})
    # near 180 with unequal widths the inner walls meet past an outlet: no polygon
    with pytest.raises(pinn.ConfigError, match="split 0.3500 in the design range"):
        config(**{"geometry.angle1": 45.0, "geometry.angle2": 120.0})


# -- the geometry as a function of s ---------------------------------------- #

@pytest.mark.parametrize("angles", [(30.0, 70.0), (0.0, 60.0), (40.0, 100.0)])
def test_the_walls_are_the_fixed_networks_walls_at_every_split(angles):
    """Merged, extended segments, singular corners and outlet normals equal
    flow's construction for the geometry at that split (a straight daughter,
    0 degrees, merges two edges into one segment)."""
    c = config(**{"geometry.angle1": angles[0], "geometry.angle2": angles[1]})
    st = P.structure(c)
    _, segments_of, corners_of = P.make_geometry(c, st)
    n1, n2 = P.outlet_normals(c)
    for s in SPLITS:
        g = c.geometry(s)
        assert np.max(np.abs(np.asarray(segments_of(s))
                             - flow.wall_segments(g, c.wall_extension))) < 1e-14
        assert np.max(np.abs(np.asarray(corners_of(s)) - g.singular_corners())) < 1e-14
        assert np.max(np.abs(np.asarray(n1) - flow.outlet_normal(g, "outlet1"))) < 1e-14
        assert np.max(np.abs(np.asarray(n2) - flow.outlet_normal(g, "outlet2"))) < 1e-14


def test_the_offset_is_a_chebyshev_series_in_the_scaled_split():
    c, _, _, params = setup()
    params["offset"] = jnp.asarray([0.3, -0.2, 0.15, 0.05])
    for s in SPLITS:
        xi = (s - c.split_mid) / c.split_half
        expected = np.polynomial.chebyshev.chebval(xi, np.asarray(params["offset"]))
        assert abs(float(P.offset(c, params, s)) - expected) < 1e-15


# -- the field: exact constraints at every s --------------------------------- #

@pytest.mark.parametrize("s", SPLITS)
def test_mass_and_the_share_are_exact_at_any_split(s):
    """Continuity holds identically; outlet 2 carries (1 - share) Q_in, outlet 1
    the rest, whatever the network's parameters, at any split."""
    c, _, field, params = setup()
    g = c.geometry(s)
    inside = g.sample_interior(np.random.default_rng(0), 200)
    rows = jnp.asarray(np.column_stack([inside, np.full(len(inside), s)]))
    d = P._row_derivatives(field, params, rows)
    assert np.max(np.abs(np.asarray(pinn.continuity(d)))) < 1e-11
    expected2 = (1.0 - c.outlet1_share) * Q_IN
    for kind, expected in (("outlet2", expected2), ("outlet1", Q_IN - expected2),
                           ("inlet", -Q_IN)):
        nodes, weights, normal = g.line_quadrature(kind, 200)
        nodes = nodes - 1e-9 * normal
        values = np.asarray(jax.vmap(lambda q: field(params, q, s))(jnp.asarray(nodes)))
        assert abs(weights @ (values[:, :2] @ normal) - expected) < 1e-8


@pytest.mark.parametrize("s", SPLITS)
def test_no_slip_holds_at_any_split(s):
    """u, v ~ distance to the wall of THAT split's polygon."""
    c, _, field, params = setup()
    g = c.geometry(s)
    points, normals = g.sample_boundary(np.random.default_rng(1), "wall", 300)
    corners = g.vertices()
    keep = np.min(np.linalg.norm(points[:, None] - corners[None], axis=2), axis=1) > 0.05
    points, normals = points[keep], normals[keep]
    speeds = []
    for h in (1e-4, 1e-6):
        inside = jnp.asarray(points - h * normals)
        values = np.asarray(jax.vmap(lambda q: field(params, q, s))(inside))
        speeds.append(np.max(np.hypot(values[:, 0], values[:, 1])))
    assert speeds[1] < 2e-2 * speeds[0]
    assert speeds[1] < 1e-4


def test_the_field_depends_on_the_split():
    """The split enters the network, not only the walls: two splits differ at
    a point inside both, and the field is smooth in s."""
    c, _, field, params = setup()
    point = jnp.asarray([1.5, 0.1])
    a, b = (np.asarray(field(params, point, s)) for s in (0.45, 0.46))
    assert np.max(np.abs(a - b)) > 1e-6
    ds = np.asarray(jax.jacfwd(lambda s: field(params, point, s))(0.45))
    # five-point stencil (CLAUDE.md): s enters the network scaled by 1 / split_half
    # = 8, so its k-th derivative carries 8^k and the three-point stencil's
    # O(h^2) truncation reached 3e-5 at h = 1e-4. Here O(h^4) at h = 1e-3.
    h = 1e-3
    f = [np.asarray(field(params, point, 0.45 + k * h)) for k in (-2, -1, 1, 2)]
    five = (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h)
    assert np.allclose(five, ds, rtol=1e-6, atol=1e-9)


def _widen(tree):
    """A two-input network's parameters with a zero column for s in every input layer."""
    if isinstance(tree, dict) and set(tree) == {"W", "b"}:
        return dict(tree, W=jnp.concatenate([tree["W"], jnp.zeros((tree["W"].shape[0], 1))], 1))
    if isinstance(tree, dict):
        return {k: v if k == "output" else _widen(v) for k, v in tree.items()}
    return [_widen(v) if k == 0 else v for k, v in enumerate(tree)]


def test_at_a_fixed_split_it_is_the_fixed_geometry_network():
    """With its s-weights zero, the parametric network at split s IS
    flow.py's stream-function network on that geometry: the same field and the
    same residuals, point by point, so everything proved of the fixed network
    (no-slip, mass, the do-nothing traction, corner weighting) carries over."""
    s = 0.45
    c = config(**{"design.split_min": s - 1e-3, "design.split_max": s + 1e-3,
                  "evaluation.splits": [s]})
    fixed = copy.deepcopy(BASE)
    fixed["geometry"]["split"] = s
    fixed["outlets"] = {"split": "prescribed", "outlet1_share": 0.5}
    del fixed["design"]
    fixed["model"].update(formulation="stream_pressure")
    del fixed["model"]["offset_degree"]
    fixed["sampling"] = {"n_interior": 64, "corner_fraction": 0.25, "corner_radius": 0.5,
                         "corner_exclusion": 0.05, "n_inlet": 16, "n_outlet": 16,
                         "n_validation": 64, "n_flux": 32}
    fixed["evaluation"] = {k: BASE["evaluation"][k]
                           for k in ("max_area", "corner_levels", "corner_radius", "n_line")}
    fc = flow.parse_config(fixed)
    g = fc.geometry()
    q_in = flow.inlet_flux(flow.inlet_function(fc), fc.n_flux)
    fixed_field = flow.make_field(fc, g, q_in)
    fixed_params = dict(flow.init_params(fc, jax.random.PRNGKey(0)), offset=jnp.asarray(0.37))
    st, inlet, _, field = P.build(c)
    params = {"net": _widen(fixed_params["net"]), "offset": jnp.asarray([0.37, 0.0, 0.0, 0.0])}
    points = jnp.asarray(g.sample_interior(np.random.default_rng(0), 50))
    a = np.asarray(jax.vmap(lambda q: fixed_field(fixed_params, q))(points))
    b = np.asarray(jax.vmap(lambda q: field(params, q, s))(points))
    assert np.max(np.abs(a - b)) < 1e-14
    batch = flow.sample_batch(np.random.default_rng(1), fc, g, inlet, 64)
    rows = flow.Batch(*(np.insert(part, 2, s, axis=1) for part in batch))
    fixed_groups, fixed_loss = flow.make_residuals(fc, g, fixed_field, q_in)
    groups, loss = P.make_residuals(c, st, field)
    one, two = float(fixed_loss(fixed_params, batch)), float(loss(params, rows))
    assert abs(one - two) < 1e-13 * one
    for (f1, r1), (f2, r2) in zip(fixed_groups(batch), groups(rows)):
        x = np.asarray(jax.vmap(lambda r: f1(fixed_params, r, len(r1)))(jnp.asarray(r1)))
        y = np.asarray(jax.vmap(lambda r: f2(params, r, len(r2)))(jnp.asarray(r2)))
        assert np.max(np.abs(x - y)) < 1e-13


# -- residuals ------------------------------------------------------------- #

def test_batched_derivatives_are_the_point_derivatives_in_x_and_y():
    c, _, field, params = setup()
    rows = []
    for s in (0.38, 0.52):
        p = c.geometry(s).sample_interior(np.random.default_rng(2), 5)
        rows.append(np.column_stack([p, np.full(5, s)]))
    rows = jnp.asarray(np.concatenate(rows))
    batched = P._row_derivatives(field, params, rows)
    for k in range(rows.shape[0]):
        point = pinn.point_derivatives(lambda q: field(params, q, rows[k, 2]), rows[k, :2])
        for name in pinn.Derivatives._fields:
            assert abs(float(getattr(batched, name)[k]) - float(getattr(point, name))) \
                < 1e-12 * (1.0 + abs(float(getattr(point, name))))


def test_point_and_batched_losses_agree():
    c, st, field, params = setup()
    params["offset"] = jnp.asarray([0.2, 0.1, -0.05, 0.02])
    groups, batched_loss = P.make_residuals(c, st, field)
    flat, unflatten = ravel_pytree(params)
    _, lm_loss = pinn.gauss_newton(groups, unflatten, 16)
    batch = P.sample_batch(np.random.default_rng(0), c, flow.inlet_function(c), 3)
    a, b = lm_loss(flat, batch), float(batched_loss(params, batch))
    assert abs(a - b) < 1e-10 * b


def test_the_outlet_residual_uses_the_offset_at_the_rows_split():
    """Outlet 1's traction gains lam(s) n at the row's own split; outlet 2's does not."""
    c, st, field, params = setup()
    groups, _ = P.make_residuals(c, st, field)
    batch = P.sample_batch(np.random.default_rng(3), c, flow.inlet_function(c), 2)
    (_, _), (_, _), (outlet1, rows1), (outlet2, rows2) = groups(batch)
    shifted = dict(params, offset=jnp.asarray([0.2, 0.1, 0.0, 0.0]))
    n1, _ = P.outlet_normals(c)
    for row in rows1[:3]:
        row = jnp.asarray(row)
        change = np.asarray(outlet1(shifted, row, 1)) - np.asarray(outlet1(params, row, 1))
        expected = np.sqrt(c.weights["outlet"]) * float(P.offset(c, shifted, row[2])) * \
            np.asarray(n1)
        assert np.allclose(change, expected, rtol=1e-12, atol=1e-15)
    for row in rows2[:3]:
        row = jnp.asarray(row)
        assert np.array_equal(np.asarray(outlet2(shifted, row, 1)),
                              np.asarray(outlet2(params, row, 1)))


# -- collocation ----------------------------------------------------------- #

def test_every_sampled_point_lies_in_its_own_design():
    c = config()
    batch = P.sample_batch(np.random.default_rng(4), c, flow.inlet_function(c), 5)
    splits = np.unique(batch.interior[:, 2])
    assert len(splits) == 5
    # stratified: one split in each fifth of the range
    bins = np.floor((splits - c.split_min) / (c.split_max - c.split_min) * 5).astype(int)
    assert sorted(bins.tolist()) == [0, 1, 2, 3, 4]
    for s in splits:
        g = c.geometry(s)
        pts = batch.interior[batch.interior[:, 2] == s, :2]
        assert len(pts) == c.points_per_design
        assert g.contains(pts).all()
        assert flow.outside_corners(pts, g.singular_corners(), c.corner_exclusion).all()
        for kind, rows in (("outlet1", batch.outlet1), ("outlet2", batch.outlet2)):
            nodes, _, normal = g.line_quadrature(kind, 2)
            mine = rows[rows[:, 2] == s, :2]
            assert len(mine) == c.outlet_per_design
            assert np.max(np.abs((mine - nodes[0]) @ normal)) < 1e-12   # on that outlet's line
    assert np.all(batch.inlet[:, 0] == 0.0) and np.all(np.abs(batch.inlet[:, 1]) <= 1.0)
    u, v = flow.inlet_function(c)(batch.inlet[:, 1])
    assert np.array_equal(batch.inlet[:, 3], u) and np.array_equal(batch.inlet[:, 4], v)


def test_the_draws_are_a_function_of_the_seed():
    c = config()
    inlet = flow.inlet_function(c)
    a = P.sample_batch(np.random.default_rng(7), c, inlet, 3)
    b = P.sample_batch(np.random.default_rng(7), c, inlet, 3)
    other = P.sample_batch(np.random.default_rng(8), c, inlet, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a.interior, other.interior)


# -- training -------------------------------------------------------------- #

def _tiny(lm_iterations=3):
    return config(**{"training.adam.steps": 20, "training.lbfgs.cycles": 1,
                     "training.lm.iterations": lm_iterations, "training.lm.average": 2})


@pytest.mark.slow
def test_training_reduces_the_held_out_loss():
    params, history, info = P.train(_tiny(), verbose=False)
    held = [row["heldout_loss"] for row in history]
    assert held[-1] < 0.5 * held[0]
    assert info["selected"] != "initial"
    assert "offset(0.35)" in history[-1]


@pytest.mark.slow
def test_a_finished_run_loads_and_evaluates(tmp_path):
    """run -> load_run gives back the trained field; evaluate measures it
    against finite elements at every evaluation split and finds both optima."""
    import yaml
    document = copy.deepcopy(BASE)
    document["training"]["adam"]["steps"] = 20
    document["training"]["lbfgs"]["cycles"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    metrics = P.run(path, out=tmp_path / "run", verbose=False)
    model, c, params = P.load_run(tmp_path / "run")
    assert [r["split"] for r in metrics["splits"]] == list(c.eval_splits)
    for row in metrics["splits"]:
        assert all(np.isfinite(v) for v in row["relative_l2"].values())
        assert abs(row["pinn_mass_imbalance"]) < 1e-8          # exact by construction
    o = metrics["optimum"]
    assert c.split_min <= o["pinn_split"] <= c.split_max
    assert c.split_min <= o["reference_split"] <= c.split_max
    assert np.isfinite(o["regret"])
    assert len(o["curve"]["split"]) == c.curve_points
    assert (tmp_path / "run" / "report.png").is_file()
    # the loaded model is the trained one
    u, v, p = model(np.array([1.0]), np.array([0.2]), 0.45)
    expected = np.asarray(P.make_field(c, P.structure(c), P.flow.inlet_flux(
        flow.inlet_function(c), c.n_flux))(params, jnp.asarray([1.0, 0.2]), 0.45))
    assert np.allclose([u[0], v[0], p[0]], expected, rtol=1e-12, atol=1e-15)
