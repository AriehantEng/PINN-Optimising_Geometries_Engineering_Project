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
