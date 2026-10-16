"""Network 2, parametric: the bifurcation PINN with the width split as an input
(bifurcation/parametric.py)."""

import copy
import math

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
              "blend_exponent": 2.0, "offset_degree": 3, "corner_modes": "none"},
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
        "init_from": "random",
    },
    "evaluation": {"max_area": 0.05, "corner_levels": 2, "corner_radius": 0.4, "n_line": 64,
                   "splits": [0.35, 0.45, 0.6], "search_xatol": 0.01, "curve_points": 5,
                   "continuation": []},
}

#: the corner-mode network: four modes on BASE's 30/70 layout (C 210, B 250,
#: T 260 degrees: T beyond 257.45 has an odd mode too), cutoff 0.4 < 0.7
CORNERS = {"model.corner_modes": "stokes", "model.corner_degree": 2, "model.corner_cutoff": 0.4,
           "model.corner_exponent_max": 1.0}

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


def setup(c=None, modes=False):
    """(config, structure, field, params); with `modes`, the corner-mode network
    with non-zero amplitudes, so that every exact constraint is tested with the
    modes switched on."""
    c = c or (config(**CORNERS) if modes else config())
    st = P.structure(c)
    params = P.init_params(c, jax.random.PRNGKey(1))
    if "corners" in params:
        params["corners"] = 0.3 * jax.random.normal(jax.random.PRNGKey(2),
                                                    params["corners"].shape)
    return c, st, P.make_field(c, st, Q_IN), params


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

@pytest.mark.parametrize("modes", [False, True])
@pytest.mark.parametrize("s", SPLITS)
def test_mass_and_the_share_are_exact_at_any_split(s, modes):
    """Continuity holds identically; outlet 2 carries (1 - share) Q_in, outlet 1
    the rest, whatever the network's parameters, at any split."""
    c, _, field, params = setup(modes=modes)
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


@pytest.mark.parametrize("modes", [False, True])
@pytest.mark.parametrize("s", SPLITS)
def test_no_slip_holds_at_any_split(s, modes):
    """u, v ~ distance to the wall of THAT split's polygon."""
    c, _, field, params = setup(modes=modes)
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
    del fixed["model"]["offset_degree"], fixed["model"]["corner_modes"]
    del fixed["training"]["init_from"]
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


# -- the corner modes --------------------------------------------------------- #

def test_the_corner_keys_are_required_with_the_modes_and_refused_without():
    config(**CORNERS)
    for key in ("model.corner_degree", "model.corner_cutoff"):
        with pytest.raises(pinn.ConfigError, match=key.split(".")[-1]):
            config(**{k: v for k, v in CORNERS.items() if k != key})
        with pytest.raises(pinn.ConfigError, match="no effect"):
            config(**{key: CORNERS[key]})
    with pytest.raises(pinn.ConfigError, match="corner_modes"):
        config(**{"model.corner_modes": "williams"})
    with pytest.raises(pinn.ConfigError, match="corner_modes"):
        config(**{"model.corner_modes": None})


def test_a_corner_cutoff_reaching_another_edge_is_refused():
    """T lies 0.7 from daughter 1's outer wall at split 0.35."""
    with pytest.raises(pinn.ConfigError, match="corner_cutoff at split 0.3500"):
        config(**{**CORNERS, "model.corner_cutoff": 0.75})


def test_every_singular_corner_gets_its_modes_with_z_below_one():
    c = config(**CORNERS)
    st = P.structure(c)
    found = P.singular_modes(c, st)
    assert [k.vertex for k in found] == list(st.singular)
    angles = sorted(round(math.degrees(k.angle), 9) for k in found)
    assert angles == [210.0, 250.0, 260.0]                       # C, B, T at 30/70
    assert [len(k.modes) for k in sorted(found, key=lambda k: k.angle)] == [1, 1, 2]
    assert P.init_params(c, jax.random.PRNGKey(0))["corners"].shape == (4, 3)
    assert P.singular_modes(config(), st) == ()
    assert "corners" not in P.init_params(config(), jax.random.PRNGKey(0))


def test_the_corner_amplitudes_are_chebyshev_series_in_the_scaled_split():
    c, _, _, params = setup(modes=True)
    for s in SPLITS:
        xi = (s - c.split_mid) / c.split_half
        expected = [np.polynomial.chebyshev.chebval(xi, row) for row in np.asarray(params["corners"])]
        assert np.allclose(np.asarray(P.corner_amplitudes(c, params, s)), expected,
                           rtol=0, atol=1e-15)


def test_with_zero_amplitudes_the_modes_change_nothing():
    """The original network is the corner-mode network with c = 0, exactly:
    stored runs keep their meaning and a run with modes starts where one
    without them would."""
    plain, with_modes = config(), config(**CORNERS)
    p0 = P.init_params(plain, jax.random.PRNGKey(3))
    p1 = dict(p0, corners=jnp.zeros_like(P.init_params(with_modes, jax.random.PRNGKey(3))
                                         ["corners"]))
    f0 = P.make_field(plain, P.structure(plain), Q_IN)
    f1 = P.make_field(with_modes, P.structure(with_modes), Q_IN)
    for s in (0.35, 0.5):
        points = jnp.asarray(plain.geometry(s).sample_interior(np.random.default_rng(0), 64))
        a = np.asarray(jax.vmap(lambda q: f0(p0, q, s))(points))
        b = np.asarray(jax.vmap(lambda q: f1(p1, q, s))(points))
        assert np.array_equal(a, b)


def test_the_modes_add_their_stream_function_and_pressure_partner():
    """field(with modes) - field(without) is, at a point within a corner's
    cutoff, the curl of c(s) chi S and c(s) chi nu P of that corner's modes,
    computed here from bifurcation/corners.py directly."""
    from bifurcation import corners as K
    c, st, field, params = setup(modes=True)
    plain = config()
    base = P.make_field(plain, P.structure(plain), Q_IN)
    base_params = {k: v for k, v in params.items() if k != "corners"}
    _, viscous = pinn.scaling_coefficients(c.reynolds, c.pressure_scaling)
    found = P.singular_modes(c, st)
    for s in (0.37, 0.55):
        v = c.geometry(s).vertices()
        amplitude = np.asarray(P.corner_amplitudes(c, params, s))
        first = 0
        for k in found:
            count = len(k.modes)

            def singular(q, k=k, first=first):
                r, theta = K.local_polar(q, jnp.asarray(v[k.vertex]), k.bisector)
                chi = K.cutoff(r, c.corner_cutoff)
                psi = sum(amplitude[first + j] * chi * K.stream(r, theta, m.z, jnp.asarray(
                    m.coefficients)) for j, m in enumerate(k.modes))
                p = sum(amplitude[first + j] * chi * viscous * K.pressure(
                    r, theta, m.z, jnp.asarray(m.coefficients)) for j, m in enumerate(k.modes))
                return psi, p

            for radius, along in ((0.05, 0.1), (0.2, -0.3)):
                b = np.asarray(k.bisector)
                direction = np.array([b[0] * math.cos(along) - b[1] * math.sin(along),
                                      b[0] * math.sin(along) + b[1] * math.cos(along)])
                q = jnp.asarray(v[k.vertex] + radius * direction)
                difference = np.asarray(field(params, q, s)) - np.asarray(base(base_params, q, s))
                g = np.asarray(jax.grad(lambda x: singular(x)[0])(q))
                expected = np.array([g[1], -g[0], float(singular(q)[1])])
                assert np.max(np.abs(difference - expected)) < 1e-12 * (1 + np.max(np.abs(expected)))
                assert np.max(np.abs(expected)) > 1e-3
            first += count


def test_the_migration_gives_old_parametric_runs_no_corner_modes(tmp_path):
    """A parametric run stored before model.corner_modes existed is migrated to
    'none', which is what it computed, and loads."""
    import yaml
    from bifurcation import migrate
    document = copy.deepcopy(BASE)
    del document["model"]["corner_modes"]
    run = tmp_path / "old"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(pinn.ConfigError, match="corner_modes"):
        P.load_config(run / "config.yaml")
    c = config()
    flat, _ = ravel_pytree(P.init_params(c, flow._streams(c.seed)[0]))
    pinn.save_params(run / "params.npz", P.init_params(c, flow._streams(c.seed)[0]))
    migrate.main(["--runs-root", str(tmp_path), "--write"])
    assert P.load_config(run / "config.yaml").corner_modes == "none"
    _, loaded, params = P.load_run(run)
    assert np.array_equal(np.asarray(ravel_pytree(params)[0]), np.asarray(flat))


# -- residuals ------------------------------------------------------------- #

@pytest.mark.parametrize("modes", [False, True])
def test_batched_derivatives_are_the_point_derivatives_in_x_and_y(modes):
    c, _, field, params = setup(modes=modes)
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


@pytest.mark.parametrize("modes", [False, True])
def test_point_and_batched_losses_agree(modes):
    c, st, field, params = setup(modes=modes)
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

def _tiny(lm_iterations=3, modes=False):
    return config(**{"training.adam.steps": 20, "training.lbfgs.cycles": 1,
                     "training.lm.iterations": lm_iterations, "training.lm.average": 2,
                     **(CORNERS if modes else {})})


@pytest.mark.slow
@pytest.mark.parametrize("modes", [False, True])
def test_training_reduces_the_held_out_loss(modes):
    params, history, info = P.train(_tiny(modes=modes), verbose=False)
    held = [row["heldout_loss"] for row in history]
    assert held[-1] < 0.5 * held[0]
    assert info["selected"] != "initial"
    assert "offset(0.35)" in history[-1]
    if modes:
        assert "corner0(0.47)" in history[-1]
        assert np.any(np.asarray(params["corners"]) != 0.0)      # the amplitudes are trained


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


# -- warm starts: the same function, grown ------------------------------------ #

def _finished(tmp_path, name, changes, params=None):
    """A 'finished' run directory: config and parameters (random, or given)."""
    import yaml
    document = copy.deepcopy(BASE)
    document["name"] = name
    for dotted, value in changes.items():
        node = document
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    c = P.load_config(directory / "config.yaml")
    if params is None:
        params = P.init_params(c, jax.random.PRNGKey(11))
        params["offset"] = jnp.asarray([0.3, -0.1, 0.05, 0.02])
        if "corners" in params:
            params["corners"] = 0.2 * jax.random.normal(jax.random.PRNGKey(12),
                                                        params["corners"].shape)
    pinn.save_params(directory / "params.npz", params)
    return directory, c, params


def _same_field(parent_dir, child_config, child_params, splits=(0.35, 0.47, 0.6)):
    model, c, params = P.load_run(parent_dir)
    _, _, _, child_field = P.build(child_config)
    for s in splits:
        pts = jnp.asarray(c.geometry(s).sample_interior(np.random.default_rng(3), 40))
        _, _, _, field = P.build(c)
        a = np.asarray(jax.vmap(lambda q: field(params, q, s))(pts))
        b = np.asarray(jax.vmap(lambda q: child_field(child_params, q, s))(pts))
        assert np.max(np.abs(a - b)) < 1e-13 * (1 + np.max(np.abs(a)))


@pytest.mark.parametrize("child", [
    {},                                                                  # continue as is
    {"model.width": 12},                                                 # wider
    {"model.offset_degree": 5},                                          # higher offset degree
    {**{k: v for k, v in CORNERS.items()}},                              # corner modes added
])
def test_a_warm_start_is_the_parent_function(tmp_path, child):
    parent_dir, _, _ = _finished(tmp_path, "parent", {})
    c = config(**{**child, "seed": 9, "training.init_from": "parent"})
    params, digest = P.warm_start(c, parent_dir, jax.random.PRNGKey(0))
    assert len(digest) == 64
    _same_field(parent_dir, c, params)
    if child.get("model.width"):
        # the new units are live: random input weights, zero effect on the output
        assert np.any(np.asarray(params["net"]["hidden"][1]["W"])[8:, :] != 0.0)
        assert np.all(np.asarray(params["net"]["output"]["W"])[:, 8:] == 0.0)


def test_a_warm_start_can_add_the_complex_corner_modes(tmp_path):
    parent_dir, _, _ = _finished(tmp_path, "parent", CORNERS)
    c = config(**{**CORNERS, "model.corner_exponent_max": 2.0, "model.corner_degree": 3,
                  "seed": 9, "training.init_from": "parent"})
    params, _ = P.warm_start(c, parent_dir, jax.random.PRNGKey(0))
    assert params["corners"].shape[0] > 4 and params["corners"].shape[1] == 4
    _same_field(parent_dir, c, params)


def test_a_warm_start_refuses_what_would_change_or_bias_it(tmp_path):
    parent_dir, _, _ = _finished(tmp_path, "parent", {})
    for changes, match in (({"seed": 5}, "seed 5"),                      # the parent's seed
                           ({"seed": 9, "geometry.angle2": 75.0}, "angle2"),
                           ({"seed": 9, "model.width": 6}, "never shrink")):
        with pytest.raises(pinn.ConfigError, match=match):
            P.warm_start(config(**{**changes, "training.init_from": "parent"}), parent_dir,
                         jax.random.PRNGKey(0))
    with pytest.raises(pinn.ConfigError, match="init_from"):
        config(**{"training.init_from": "../parent"})
