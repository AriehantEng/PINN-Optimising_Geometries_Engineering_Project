"""The bifurcation geometry (bifurcation/geometry.py).

Checked through exact invariants rather than pictures: outlets as wide as
their daughters, inner walls parallel to outer walls at that distance, the
septum tip on both inner walls, outward normals, mirror symmetry, and a
minimum width that no parameter can get under.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bifurcation.geometry import Geometry, GeometryError, split_bounds, vertices

jax.config.update("jax_enable_x64", True)

CASES = [
    dict(split=0.5, angle1=60.0, angle2=60.0),
    dict(split=0.3, angle1=0.0, angle2=60.0),
    dict(split=0.7, angle1=20.0, angle2=45.0),
    dict(split=0.5, angle1=5.0, angle2=0.0),
    dict(split=0.4, angle1=0.0, angle2=120.0),      # one straight, one turned back
    dict(split=0.6, angle1=30.0, angle2=90.0),      # 120 degrees, off-centre orientation
]


def make(split, angle1, angle2):
    return Geometry.from_split(split, angle1, angle2, parent_length=3.0, daughter_length=8.0,
                               min_width=0.2)


@pytest.mark.parametrize("case", CASES)
def test_the_daughters_are_strips_of_their_own_width(case):
    g = make(**case)
    A, B, E2o, E2i, T, E1i, E1o, C, D = g.vertices()
    assert abs(np.linalg.norm(E1o - E1i) - g.w1) < 1e-12
    assert abs(np.linalg.norm(E2o - E2i) - g.w2) < 1e-12
    assert abs(g.w1 + g.w2 - 2.0) < 1e-12                       # the parent's width, shared
    for outer_start, outer_end, inner_end, w in ((C, E1o, E1i, g.w1), (B, E2o, E2i, g.w2)):
        d = (outer_end - outer_start) / np.linalg.norm(outer_end - outer_start)
        # the inner wall is parallel to the outer one, at distance w, and passes through T
        cross = lambda a, b: a[0] * b[1] - a[1] * b[0]           # noqa: E731
        assert abs(cross(d, inner_end - T)) < 1e-12
        assert abs(abs(cross(d, T - outer_start)) - w) < 1e-12
    assert T[0] >= g.parent_length - 1e-12                      # the tip is not inside the parent


def test_one_straight_daughter_keeps_the_parent_wall_and_both_straight_is_refused():
    g = make(0.4, 0.0, 30.0)
    A, B, E2o, E2i, T, E1i, E1o, C, D = g.vertices()
    assert abs(E1o[1] - 1.0) < 1e-12 and abs(E1i[1] - (1.0 - g.w1)) < 1e-12
    assert abs(T[1] - (1.0 - g.w1)) < 1e-12                     # the tip on daughter 1's inner wall
    with pytest.raises(GeometryError, match="splitter plate"):
        make(0.5, 0.0, 0.0)


# -- a 180-degree bifurcation: parallel inner walls, a flat far wall ---------- #

STRAIGHT_ACROSS = [(90.0, 90.0), (60.0, 120.0), (30.0, 150.0), (120.0, 60.0), (45.0, 135.0)]


def daughters_are_strips(g):
    """Both daughters are strips of their own width, whatever the septum."""
    v = g.vertices()
    A, B, E2o, E2i = v[:4]
    E1i, E1o, C, D = v[-4:]
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]               # noqa: E731
    for outer_start, outer_end, inner_end, w in ((C, E1o, E1i, g.w1), (B, E2o, E2i, g.w2)):
        d = (outer_end - outer_start) / np.linalg.norm(outer_end - outer_start)
        assert abs(abs(cross(d, inner_end - outer_start)) - w) < 1e-12
        septum_end = [p for p in g.septum_polyline() if not np.array_equal(p, inner_end)]
        # every septum vertex next to this inner wall lies on its line
        nearest = min(septum_end, key=lambda p: np.linalg.norm(p - inner_end))
        assert abs(abs(cross(d, nearest - outer_start)) - w) < 1e-12


@pytest.mark.parametrize("angles", STRAIGHT_ACROSS)
@pytest.mark.parametrize("split", [0.3, 0.5, 0.7])
def test_a_180_degree_bifurcation_closes_with_a_flat_far_wall(angles, split):
    """The inner walls are parallel and meet nowhere: the junction is closed by
    a wall from start2 to start1. Until this shape existed, the tip was put at
    start1, bending daughter 2's inner wall whenever w1 != w2 (2.3 degrees at
    (90, 90), split 0.4; 5.9 degrees at (60, 120) even at split 0.5)."""
    g = make(split, *angles)
    assert g.septum == "face" and len(g.vertices()) == 10
    daughters_are_strips(g)
    assert [k for *_, k in g.edges()].count("wall") == 7
    # the same tiling and sampling guarantees as the tip shape
    tri = g.triangles()
    area = 0.5 * ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                  - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
    assert area.min() > -1e-12 and abs(area.sum() - g.area()) < 1e-10
    rng = np.random.default_rng(0)
    for kind in ("wall", "inlet", "outlet1", "outlet2"):
        p, n = g.sample_boundary(rng, kind, 400)
        assert not g.contains(p + 1e-6 * n).any()


def test_the_far_wall_is_the_limit_of_the_tip():
    """Approaching 180 degrees symmetrically, the tip flattens onto the far wall."""
    face = make(0.5, 90.0, 90.0)
    tip = make(0.5, 89.9999, 89.9999)
    assert tip.septum == "tip"
    assert abs(tip.area() - face.area()) < 1e-5
    assert np.linalg.norm(tip.vertices()[4] - 0.5 * (face.vertices()[4] + face.vertices()[5])) < 1e-5


def test_near_180_with_unequal_widths_the_walls_meet_too_far_away():
    """At 89 + 90 degrees, walls 0.8 and 1.2 wide meet 24 units up daughter 1:
    there is no tip near the junction, and the geometry says so."""
    with pytest.raises(GeometryError, match="meet beyond an outlet"):
        make(0.4, 89.0, 90.0)


def test_a_daughter_at_180_degrees_and_a_closing_entrance_are_refused():
    with pytest.raises(GeometryError, match="folds the daughter flat"):
        make(0.5, 0.0, 180.0)
    # (10, 170): full widths, but the entrance at B is 0.17 w2 = 0.174 < 0.2
    with pytest.raises(GeometryError, match="passage into daughter 2"):
        make(0.5, 10.0, 170.0)
    # a wider entrance makes it valid again, and the rule is the minimum width
    g = Geometry.from_split(0.5, 10.0, 170.0, 3.0, 8.0, min_width=0.15)
    assert g.septum == "face"


@pytest.mark.parametrize("case", CASES)
def test_the_reentrant_corners_are_b_t_and_c(case):
    g = make(**case)
    A, B, E2o, E2i, T, E1i, E1o, C, D = g.vertices()
    assert np.array_equal(g.reentrant_corners(), np.stack([B, T, C]))


def test_equal_angles_and_widths_give_a_mirror_symmetric_split():
    v = make(0.5, 50.0, 50.0).vertices()
    reflected = v * np.array([1.0, -1.0])
    # every reflected vertex is a vertex: the polygon is its own mirror image
    distance = np.linalg.norm(reflected[:, None, :] - v[None, :, :], axis=2).min(axis=1)
    assert distance.max() < 1e-12
    assert not np.allclose(make(0.3, 50.0, 50.0).vertices() * np.array([1.0, -1.0]),
                           make(0.3, 50.0, 50.0).vertices())


@pytest.mark.parametrize("case", CASES)
def test_interior_samples_fill_the_polygon_and_normals_point_out(case):
    g = make(**case)
    rng = np.random.default_rng(0)
    lo, hi = g.bounding_box()
    trial = lo + (hi - lo) * rng.random((200_000, 2))
    estimate = np.mean(g.contains(trial)) * np.prod(hi - lo)
    assert abs(estimate / g.area() - 1.0) < 0.02
    points = g.sample_interior(rng, 5000)
    assert len(points) == 5000 and g.contains(points).all()
    for kind in ("wall", "inlet", "outlet1", "outlet2"):
        p, n = g.sample_boundary(rng, kind, 400)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
        assert not g.contains(p + 1e-6 * n).any()
        assert g.contains(p - 1e-6 * n).mean() > 0.99            # corners aside


@pytest.mark.parametrize("kind, width", [("inlet", 2.0), ("outlet1", None), ("outlet2", None)])
def test_line_quadrature_measures_each_opening(kind, width):
    g = make(0.3, 20.0, 45.0)
    expected = {"inlet": 2.0, "outlet1": g.w1, "outlet2": g.w2}[kind]
    points, weights, normal = g.line_quadrature(kind, 16)
    assert abs(weights.sum() - expected) < 1e-12
    assert not g.contains(points[[0, -1]] + 1e-6 * normal).any()


def test_no_width_can_reach_zero():
    with pytest.raises(GeometryError, match=r"split must lie in \[0.1, 0.9\]"):
        make(0.05, 30.0, 30.0)
    with pytest.raises(GeometryError, match="minimum width"):
        Geometry(0.1, 1.9, 30.0, 30.0, 3.0, 8.0, min_width=0.2)
    with pytest.raises(GeometryError, match="min_width"):
        split_bounds(0.0)
    with pytest.raises(TypeError):
        Geometry.from_split(0.5, 30.0, 30.0, 3.0, 8.0)          # min_width is required
    assert split_bounds(0.2) == (0.1, 0.9)


@pytest.mark.parametrize("min_width", [0.2, 0.3, 0.1, 0.25])
def test_both_split_bounds_are_accepted(min_width):
    """The bounds are where an optimiser that wants unequal daughters ends up,
    and 2 (1 - 0.9) rounds below 0.2: the bound it was told to use was refused."""
    lo, hi = split_bounds(min_width)
    for split in (lo, hi):
        g = Geometry.from_split(split, 30.0, 60.0, 3.0, 8.0, min_width=min_width)
        assert min(g.w1, g.w2) == min_width
        assert abs(g.w1 + g.w2 - 2.0) < 1e-15


def test_impossible_geometries_are_refused():
    with pytest.raises(GeometryError, match="overlap"):
        Geometry(1.8, 1.8, 0.0, 5.0, 3.0, 8.0, min_width=0.2)  # far wider than the parent
    with pytest.raises(GeometryError, match="angle"):
        make(0.5, 180.0, 30.0)
    with pytest.raises(GeometryError, match="too short"):
        # the tip sits 0.061 past the junction; outlets 0.05 along are before it
        Geometry.from_split(0.3, 0.0, 5.0, 3.0, 0.05, min_width=0.2)


def test_every_vertex_is_differentiable_in_the_design_parameters():
    """Network 3 differentiates the pressure drop through the outlets' positions."""
    def outlet1_end(params):
        split, a1, a2 = params
        return vertices(2.0 * split, 2.0 * (1.0 - split), a1, a2, 3.0, 8.0, xp=jnp)[6]

    p = jnp.array([0.4, math.radians(30.0), math.radians(50.0)])
    jac = jax.jacfwd(outlet1_end)(p)
    # E1o = C + L (cos a1, sin a1): independent of the split, d/da1 = L(-sin a1, cos a1)
    assert np.allclose(jac[:, 0], 0.0)
    assert np.allclose(jac[:, 1], 8.0 * np.array([-math.sin(p[1]), math.cos(p[1])]))
    tip = jax.jacfwd(lambda q: vertices(2.0 * q[0], 2.0 * (1.0 - q[0]), q[1], q[2], 3.0, 8.0,
                                        xp=jnp)[4])(p)
    assert np.all(np.isfinite(np.asarray(tip)))


@pytest.mark.parametrize("angle1, angle2", [(60.0, 60.0), (30.0, 90.0), (0.0, 120.0)])
def test_one_bifurcation_angle_many_orientations_and_an_off_centre_tip(angle1, angle2):
    """The bifurcation angle is between the inner walls, at the tip; the tip
    moves off the centreline with unequal widths."""
    g = make(0.3, angle1, angle2)
    A, B, E2o, E2i, T, E1i, E1o, C, D = g.vertices()
    inner1, inner2 = E1i - T, E2i - T
    between = math.degrees(math.acos(np.dot(inner1, inner2)
                                     / np.linalg.norm(inner1) / np.linalg.norm(inner2)))
    assert abs(between - (angle1 + angle2)) < 1e-9
    assert abs(T[1]) > 1e-3                                     # w1 = 0.6, w2 = 1.4: off-centre



ORIENTATIONS = [(60.0, 60.0), (30.0, 90.0), (0.0, 120.0), (45.0, 75.0), (20.0, 45.0),
                (5.0, 0.0), (80.0, 80.0), (10.0, 150.0)]


@pytest.mark.parametrize("angles", ORIENTATIONS)
@pytest.mark.parametrize("split", [0.15, 0.5, 0.85])
def test_the_polygon_cuts_into_positive_triangles_that_tile_it(angles, split):
    from bifurcation.geometry import triangle_quadrature
    try:
        g = make(split, *angles)
    except GeometryError:
        pytest.skip("not a valid bifurcation")
    tri = g.triangles()
    area = 0.5 * ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                  - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
    assert area.min() > -1e-12
    assert abs(area.sum() - g.area()) < 1e-10
    points, weights = triangle_quadrature(tri, 2)
    assert abs(weights.sum() - g.area()) < 1e-10
    assert g.contains(points).all()


def test_the_triangle_rule_is_exact_to_degree_six():
    from bifurcation.geometry import triangle_quadrature
    square = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], float)
    for k in (1, 3):
        p, w = triangle_quadrature(square, k)
        for i, j in ((0, 0), (3, 3), (6, 0), (2, 4)):
            exact = 1.0 / ((i + 1) * (j + 1))
            assert abs(np.sum(w * p[:, 0] ** i * p[:, 1] ** j) - exact) < 1e-14


def test_domain_integrals_differentiate_through_the_moving_boundary():
    """Network 3 needs d(integral over the domain)/d(design): checked on the
    area, whose derivative is known by finite differences of the polygon."""
    from bifurcation.geometry import triangle_quadrature, triangles

    def area(params):
        split, a1, a2 = params
        tri = triangles(2.0 * split, 2.0 * (1.0 - split), a1, a2, 3.0, 8.0, "tip", xp=jnp)
        _, w = triangle_quadrature(tri, 1, xp=jnp)
        return jnp.sum(w)

    p = jnp.array([0.4, math.radians(30.0), math.radians(70.0)])
    grad = np.asarray(jax.grad(area)(p))
    for i in range(3):
        h = 1e-6
        dp = np.zeros(3)
        dp[i] = h
        g_plus = Geometry.from_split(float(p[0] + dp[0]), math.degrees(p[1] + dp[1]),
                                     math.degrees(p[2] + dp[2]), 3.0, 8.0, min_width=0.2)
        g_minus = Geometry.from_split(float(p[0] - dp[0]), math.degrees(p[1] - dp[1]),
                                      math.degrees(p[2] - dp[2]), 3.0, 8.0, min_width=0.2)
        assert abs(grad[i] - (g_plus.area() - g_minus.area()) / (2 * h)) < 1e-6


# -- the finite-element reference on the bifurcation ------------------------ #

COARSE = dict(max_area=0.04, corner_levels=3, corner_radius=0.4)


def coarse_solution(split, angle1, angle2):
    from bifurcation import reference as br
    g = make(split, angle1, angle2)
    return br.solve_bifurcation(g, 10.0, "inertial", br.parabola_inlet, None, **COARSE, continuation=())


def test_the_mesh_fills_the_polygon_and_is_refined_at_the_corners():
    from bifurcation import reference as br
    import skfem as fem
    g = make(0.4, 30.0, 60.0)
    m = br.mesh(g, max_area=0.04, corner_levels=4, corner_radius=0.4)
    area = fem.Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(fem.Basis(m, fem.ElementTriP1()))
    assert abs(area / g.area() - 1.0) < 1e-12
    facets = br.boundary_facets(m, g)
    assert set(facets) == {"wall", "inlet", "outlet1", "outlet2"}
    edge = np.linalg.norm(m.p[:, m.facets[0]] - m.p[:, m.facets[1]], axis=0)
    centre = m.p[:, m.facets].mean(axis=1).T
    tip = g.vertices()[4]
    near = np.linalg.norm(centre - tip, axis=1) < 0.02
    assert edge[near].max() < 0.25 * edge.max()


def test_mass_is_conserved_exactly_and_the_outlets_share_one_pressure():
    """Taylor-Hood conserves mass globally (the constant is in the P1 pressure
    space); the do-nothing outlets hold the same pressure, so the flow chooses
    its own split."""
    m = coarse_solution(0.4, 30.0, 70.0).measures()
    assert abs(m["mass_imbalance"]) < 1e-12
    assert abs(m["flux_in"] - 4.0 / 3.0) < 1e-12               # the parabola carries 4/3
    assert abs(m["static_pressure_out1"]) < 1e-3 and abs(m["static_pressure_out2"]) < 1e-3
    assert m["energy_consistency"] < 1e-3


def test_a_daughter_turned_back_across_the_inlet_line_keeps_no_slip():
    """At 0/120 daughter 2's outer wall crosses x* = 0 at y* = -6.2. The inlet
    profile was applied on the LINE x* = 0, so that wall node got u = -37 and
    mass imbalance was -152%."""
    m = coarse_solution(0.7, 0.0, 120.0).measures()
    assert abs(m["mass_imbalance"]) < 1e-12
    assert m["energy_consistency"] < 1e-3
    assert 0.0 < m["flux_out1"] and 0.0 < m["flux_out2"]


def test_a_mirror_symmetric_bifurcation_splits_the_flow_evenly():
    m = coarse_solution(0.5, 60.0, 60.0).measures()
    assert abs(m["split_flux"] - 0.5) < 1e-4                    # the mesh is not exactly symmetric
    lopsided = coarse_solution(0.5, 20.0, 80.0).measures()
    assert lopsided["split_flux"] > 0.5 + 1e-3                  # the straighter branch takes more


def prescribed(split, angle1, angle2, share):
    from bifurcation import reference as br
    g = make(split, angle1, angle2)
    return br.solve_bifurcation(g, 10.0, "inertial", br.parabola_inlet, share, **COARSE, continuation=())


def test_prescribing_the_free_split_changes_nothing():
    """The flux-defective outlet is the do-nothing outlet plus a pressure offset:
    asked for the split the flow chose by itself, it must need no offset."""
    free = coarse_solution(0.4, 30.0, 70.0).measures()
    held = prescribed(0.4, 30.0, 70.0, free["split_flux"]).measures()
    assert abs(held["outlet1_pressure_offset"]) < 1e-10
    for key in ("total_pressure_drop", "split_flux", "dissipation", "static_pressure_in"):
        assert abs(held[key] - free[key]) < 1e-10 * max(1.0, abs(free[key])), key


def test_a_prescribed_share_is_delivered_and_paid_for_in_outlet_pressure():
    free = coarse_solution(0.4, 30.0, 70.0).measures()
    more = prescribed(0.4, 30.0, 70.0, free["split_flux"] + 0.1).measures()
    assert abs(more["split_flux"] - (free["split_flux"] + 0.1)) < 1e-12
    assert abs(more["mass_imbalance"]) < 1e-12
    # more flow through outlet 1 than it takes freely: outlet 1 must sit lower
    assert more["outlet1_pressure_offset"] < -1e-3
    assert more["static_pressure_out1"] < more["static_pressure_out2"] - 1e-3
    assert more["energy_consistency"] < 1e-3
    with pytest.raises(ValueError, match="outlet1_share"):
        prescribed(0.4, 30.0, 70.0, 1.0)


def test_the_two_paths_total_pressure_drops_recombine_to_the_objective():
    """Q0 J = Q1 (Pt_in - Pt_1) + Q2 (Pt_in - Pt_2), with Pt the flux-weighted
    total pressure of each opening: the per-path drops behind the junction-loss
    coefficients add up to the objective."""
    m = prescribed(0.4, 30.0, 70.0, 0.3).measures()
    paths = ((m["flux_out1"] * (m["total_pressure_in"] - m["total_pressure_out1"])
              + m["flux_out2"] * (m["total_pressure_in"] - m["total_pressure_out2"]))
             / m["flux_in"])
    assert abs(paths - m["total_pressure_drop"]) < 1e-12 * m["total_pressure_drop"]


def test_equal_outlet_pressures_drive_the_optimum_to_a_width_bound():
    """Parallel daughters conduct like w^3, and w1^3 + w2^3 is convex under
    w1 + w2 = w0: with a free split the pressure drop falls toward the most
    unequal split, whatever the junction does."""
    drops = [coarse_solution(s, 60.0, 60.0).measures()["total_pressure_drop"]
             for s in (0.1, 0.3, 0.5)]
    assert drops[0] < drops[1] < drops[2]


def test_equal_prescribed_shares_have_an_interior_optimum():
    """Delivering half the flow to each daughter, the symmetric split costs least."""
    drops = [prescribed(s, 60.0, 60.0, 0.5).measures()["total_pressure_drop"]
             for s in (0.35, 0.5, 0.65)]
    assert drops[1] < drops[0] and drops[1] < drops[2]


@pytest.mark.slow
def test_the_pressure_drop_converges_as_the_mesh_is_refined():
    from bifurcation import reference as br
    g = make(0.4, 30.0, 70.0)
    drops = []
    for area in (0.08, 0.02, 0.005):
        sol = br.solve_bifurcation(g, 10.0, "inertial", br.parabola_inlet, None, max_area=area,
                                   corner_levels=4, corner_radius=0.4, continuation=())
        drops.append(sol.measures()["total_pressure_drop"])
    changes = np.abs(np.diff(drops))
    assert changes[1] < 0.5 * changes[0]


@pytest.mark.parametrize("angles", [[0.0, 150.0], [30.0, 120.0], [60.0, 60.0], [10.0, 80.0],
                                    [60.0, 120.0], [30.0, 150.0]])
@pytest.mark.parametrize("split", [0.2, 0.5, 0.8])
def test_equal_closed_lengths(angles, split):
    """from_split_closed: same widths as from_split, each daughter with both walls
    over exactly the closed length; the outer wall is never the shorter."""
    from bifurcation import sensitivity
    g = Geometry.from_split_closed(split, angles[0], angles[1], 3.0, 8.0, 0.2)
    ref = Geometry.from_split(split, angles[0], angles[1], 3.0, 8.0, 0.2)
    assert (g.w1, g.w2) == (ref.w1, ref.w2)
    assert max(abs(x - 8.0) for x in sensitivity.closed_lengths(g)) < 1e-12
    assert min(g.daughter_lengths) >= 8.0
    if g.septum == "face":                                   # square entrances: nothing to add
        assert g.daughter_lengths == (8.0, 8.0)


def test_a_pair_of_equal_lengths_is_the_single_length():
    a = Geometry.from_split(0.4, 30.0, 70.0, 3.0, 8.0, 0.2)
    b = Geometry.from_split(0.4, 30.0, 70.0, 3.0, (8.0, 8.0), 0.2)
    assert np.array_equal(a.vertices(), b.vertices())
    with pytest.raises(GeometryError, match="positive"):
        Geometry.from_split(0.4, 30.0, 70.0, 3.0, (8.0, -1.0), 0.2)


def test_continuation_reaches_the_same_solution():
    """Starting Newton from a lower-Re solution changes the path, not the answer."""
    from bifurcation import reference as br
    g = Geometry.from_split(0.5, 30.0, 90.0, 3.0, 4.0, 0.2)
    direct = br.solve_bifurcation(g, 25.0, "inertial", br.parabola_inlet, 0.4, **COARSE,
                                  continuation=())
    stepped = br.solve_bifurcation(g, 25.0, "inertial", br.parabola_inlet, 0.4, **COARSE,
                                   continuation=(5.0, 10.0))
    a, b = direct.measures(), stepped.measures()
    assert abs(a["total_pressure_drop"] / b["total_pressure_drop"] - 1.0) < 1e-11
    assert [s["reynolds"] for s in stepped.info["continuation"]] == [5.0, 10.0, 25.0]
    with pytest.raises(ValueError, match="continuation"):
        br.solve_bifurcation(g, 25.0, "inertial", br.parabola_inlet, 0.4, **COARSE,
                             continuation=(30.0,))
