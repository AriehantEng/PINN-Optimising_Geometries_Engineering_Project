"""The 2D bifurcation geometry: a parent channel splitting into two daughters.

Units are those of developing_flow: lengths on the parent's half-width, so the
parent channel is y* in [-1, 1] and its width is w0 = 2.

    D ------------------------ C
    |                           \\  daughter 1, width w1, at angle a1 above the axis
    | inlet       parent         T (septum tip)
    |                           /   daughter 2, width w2, at angle a2 below the axis
    A ------------------------ B

The parent runs from the inlet x* = 0 to the junction x* = parent_length.
Each daughter is a straight strip: its outer wall leaves the parent's corner
(C or B) at its angle, its inner wall runs parallel at a distance of its width,
and the two inner walls meet at the septum tip T. Each daughter ends in an
outlet perpendicular to it, daughter_length along its outer wall.

Angles are DIRECTIONS, measured from the parent's axis, daughter 1 turning up
and daughter 2 down; 0 is straight on, above 90 turns back. The bifurcation
angle a1 + a2 is the angle between the two inner walls where they meet, at the
septum tip, and the tip sits wherever the widths put it: off the centreline
whenever w1 != w2. The angles need not be equal: a 120-degree bifurcation is
(60, 60), (30, 90) or (0, 120), among others, and which orientation loses the
least pressure is part of the design question.

Width rules, chosen by the caller:
  * `Geometry.from_split(split, ...)`: w1 + w2 = w0, the parent's width shared
    between the daughters (w1 = split * w0). The main design problem.
  * `Geometry(w1, w2, ...)`: independent widths, for user width limits.

No width may approach zero: a vanishing daughter is not a bifurcation. Every
geometry carries a REQUIRED `min_width` (no default) and refuses a narrower
daughter; `split_bounds` gives the split range it allows, which the geometric
network maps its output into, so no optimiser can produce a vanishing branch.

The corners B and C and the tip T are sharp: the flow has corner singularities
there (finite pressure drop, unbounded pressure gradient), which the
finite-element reference resolves by local refinement.

`vertices` accepts numpy or jax.numpy, so a network can differentiate any
boundary position with respect to the widths and angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The parent's width in these units (half-width 1).
PARENT_WIDTH = 2.0

#: Edge labels of the polygon, in vertex order A B E2o E2i T E1i E1o C D.
EDGE_KINDS = ("wall", "wall", "outlet2", "wall", "wall", "outlet1", "wall", "wall", "inlet")


class GeometryError(ValueError):
    pass


def vertices(w1, w2, angle1, angle2, parent_length, daughter_length, xp=np):
    """The polygon's 9 vertices, counter-clockwise, as an (9, 2) array:
    A, B, E2o, E2i, T, E1i, E1o, C, D. Angles in RADIANS here.

    Written with `xp` (numpy or jax.numpy) and no branching on values, so it
    is differentiable in every argument.
    """
    one = xp.ones(())
    A = xp.stack([0.0 * one, -one])
    B = xp.stack([parent_length * one, -one])
    C = xp.stack([parent_length * one, one])
    D = xp.stack([0.0 * one, one])
    d1 = xp.stack([xp.cos(angle1), xp.sin(angle1)])        # daughter 1 direction (up)
    d2 = xp.stack([xp.cos(angle2), -xp.sin(angle2)])       # daughter 2 direction (down)
    n1 = xp.stack([xp.sin(angle1), -xp.cos(angle1)])       # from outer wall 1 into daughter 1
    n2 = xp.stack([xp.sin(angle2), xp.cos(angle2)])        # from outer wall 2 into daughter 2
    start1 = C + w1 * n1                                   # inner wall 1 passes here
    start2 = B + w2 * n2                                   # inner wall 2 passes here
    # Septum tip: start1 + t1 d1 = start2 + t2 d2. The determinant is
    # sin(a1 + a2); when both daughters are straight the inner walls are
    # parallel and the tip sits on the junction line, at start1 (t1 = 0).
    rhs = start2 - start1
    det = xp.sin(angle1 + angle2)
    parallel = xp.abs(det) < 1e-12
    safe = xp.where(parallel, 1.0, det)
    t1 = xp.where(parallel, 0.0, (rhs[0] * (-d2[1]) - rhs[1] * (-d2[0])) / safe)
    T = start1 + t1 * d1
    E1o = C + daughter_length * d1
    E1i = E1o + w1 * n1
    E2o = B + daughter_length * d2
    E2i = E2o + w2 * n2
    return xp.stack([A, B, E2o, E2i, T, E1i, E1o, C, D])


def corner_points(w1, w2, angle1, angle2, parent_length, daughter_length, xp=np):
    """The named points the decomposition needs (angles in radians):
    A B C D, the daughters' entry points start1 = C + w1 n1 and start2 = B + w2 n2
    (each daughter starts at the cross-section through its outer corner), the
    tip T, and the outlet ends E1o E1i E2o E2i. Differentiable, like `vertices`."""
    v = vertices(w1, w2, angle1, angle2, parent_length, daughter_length, xp=xp)
    A, B, E2o, E2i, T, E1i, E1o, C, D = (v[i] for i in range(9))
    start1 = C + w1 * xp.stack([xp.sin(angle1), -xp.cos(angle1)])
    start2 = B + w2 * xp.stack([xp.sin(angle2), xp.cos(angle2)])
    return dict(A=A, B=B, C=C, D=D, T=T, start1=start1, start2=start2,
                E1o=E1o, E1i=E1i, E2o=E2o, E2i=E2i)


def triangles(w1, w2, angle1, angle2, parent_length, daughter_length, xp=np):
    """The polygon cut into 9 triangles, as a (9, 3, 2) array, each listed
    counter-clockwise (positive area):

        parent      A B C, A C D
        junction    a fan of 3 over B start2 T start1 C
        daughter 1  C start1 E1i, C E1i E1o
        daughter 2  B E2o E2i, B E2i start2

    The junction is the region between the parent's end and the daughters'
    entry sections, fanned from a vertex that sees the whole of it (a triangle
    may collapse to zero area where an entry point lies on the junction line).
    Every corner is a differentiable function of the design parameters, so
    Gauss points mapped onto these triangles integrate over a domain that
    moves with the widths and angles, and the integral can be differentiated.
    """
    q = corner_points(w1, w2, angle1, angle2, parent_length, daughter_length, xp=xp)
    fixed = [("A", "B", "C"), ("A", "C", "D"),
             ("C", "start1", "E1i"), ("C", "E1i", "E1o"),
             ("B", "E2o", "E2i"), ("B", "E2i", "start2")]
    tris = [xp.stack([q[a], q[b], q[c]]) for a, b, c in fixed]
    # The junction pentagon B start2 T start1 C is not always convex (T is a
    # reflex vertex when both daughters turn). Fan it from whichever vertex
    # sees all the others: the one whose smallest fan triangle is largest.
    # The choice switches between geometries, but the area it covers does not.
    ring = [q["B"], q["start2"], q["T"], q["start1"], q["C"]]
    fans = []
    for apex in range(5):
        others = [ring[(apex + k) % 5] for k in range(1, 5)]
        fans.append(xp.stack([xp.stack([ring[apex], others[k], others[k + 1]])
                              for k in range(3)]))
    fans = xp.stack(fans)                                         # (5, 3, 3, 2)
    a, b, c = fans[..., 0, :], fans[..., 1, :], fans[..., 2, :]
    area = 0.5 * ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                  - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    best = xp.argmax(xp.min(area, axis=1))
    junction = fans[best]
    return xp.concatenate([xp.stack(tris[:2]), junction, xp.stack(tris[2:])])


#: Symmetric Gauss rule on the reference triangle (Dunavant 1985, degree 6,
#: 12 points), as orbits: (a, b, b) with 3 points, (a, b, c) with 6. The
#: repeated coordinate is COMPUTED from the first, so equal values are equal
#: to the bit: typed separately they differed in the last digit, and a set of
#: permutations then counted one orbit twice (weights summed to 1.35).
_DUNAVANT6_AAB = ((0.873821971016996, 0.050844906370207),
                  (0.501426509658179, 0.116786275726379))
_DUNAVANT6_ABC = ((0.636502499121399, 0.310352451033785, 0.082851075618374),)


def _dunavant6():
    bary, weight = [], []
    for a, w in _DUNAVANT6_AAB:
        b = 0.5 * (1.0 - a)
        for p in ((a, b, b), (b, a, b), (b, b, a)):
            bary.append(p)
            weight.append(w)
    for a, b, w in _DUNAVANT6_ABC:
        c = 1.0 - a - b
        for p in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
            bary.append(p)
            weight.append(w)
    bary, weight = np.array(bary), np.array(weight)
    if abs(weight.sum() - 1.0) > 1e-14 or len(weight) != 12:
        raise AssertionError("the Dunavant rule is mis-specified")
    return bary, weight


def triangle_quadrature(tris, subdivisions, xp=np):
    """Gauss points and weights over a set of triangles: each triangle is cut
    into subdivisions^2 similar ones and a degree-6 rule is applied to each.
    Returns (points (N, 2), weights (N,)) with sum(weights) = total area."""
    bary, weight = _dunavant6()
    k = subdivisions
    sub = []                                   # sub-triangles in barycentric coordinates
    for i in range(k):
        for j in range(k - i):
            p0 = np.array([i, j]) / k
            sub.append((p0, p0 + np.array([1, 0]) / k, p0 + np.array([0, 1]) / k))
            if i + j < k - 1:
                sub.append((p0 + np.array([1, 0]) / k, p0 + np.array([1, 1]) / k,
                            p0 + np.array([0, 1]) / k))
    lam = []                                   # (s, t) of every point in the unit triangle
    for p0, p1, p2 in sub:
        for b in bary:
            lam.append(b[0] * p0 + b[1] * p1 + b[2] * p2)
    lam = np.array(lam)
    w_ref = np.tile(weight, len(sub)) / len(sub)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    pts = (a[:, None, :] + lam[None, :, 0:1] * (b - a)[:, None, :]
           + lam[None, :, 1:2] * (c - a)[:, None, :])
    return pts.reshape(-1, 2), (area[:, None] * xp.asarray(w_ref)[None, :]).reshape(-1)


@dataclass(frozen=True)
class Geometry:
    """One bifurcation. Widths in the parent's half-width units, angles in degrees."""

    w1: float
    w2: float
    angle1: float
    angle2: float
    parent_length: float
    daughter_length: float
    min_width: float

    def __post_init__(self):
        for name in ("min_width", "parent_length", "daughter_length"):
            if not getattr(self, name) > 0.0:
                raise GeometryError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("w1", "w2"):
            if not getattr(self, name) >= self.min_width:
                raise GeometryError(
                    f"{name} = {getattr(self, name)} is below the minimum width {self.min_width}: "
                    "a daughter may never close")
        for name in ("angle1", "angle2"):
            if not 0.0 <= getattr(self, name) < 180.0:
                raise GeometryError(f"{name} must lie in [0, 180) degrees, got {getattr(self, name)}")
        if self.angle1 + self.angle2 == 0.0:
            raise GeometryError(
                "both daughters straight is a zero-thickness splitter plate, not a bifurcation: "
                "give at least one daughter an angle")
        v = self.vertices()
        A, B, E2o, E2i, T, E1i, E1o, C, D = v
        a1, a2 = math.radians(self.angle1), math.radians(self.angle2)
        d1 = np.array([math.cos(a1), math.sin(a1)])
        d2 = np.array([math.cos(a2), -math.sin(a2)])
        start1 = C + self.w1 * np.array([math.sin(a1), -math.cos(a1)])
        start2 = B + self.w2 * np.array([math.sin(a2), math.cos(a2)])
        # Each daughter starts at the cross-section through its outer corner
        # (start_i on its inner wall). If the inner walls cross only DOWNSTREAM
        # of both starting sections, the two strips overlap between those
        # sections and the tip. In a valid bifurcation the tip lies at or
        # upstream of at least one of them: (60, 60) both, (0, 60) one.
        upstream1 = np.dot(T - start1, d1) <= 1e-12
        upstream2 = np.dot(T - start2, d2) <= 1e-12
        if not (upstream1 or upstream2):
            raise GeometryError(
                f"the daughters overlap after the junction: widths {self.w1:g} + {self.w2:g} are "
                f"too large for angles {self.angle1:g}, {self.angle2:g}")
        # The tip must not lie upstream of the junction: then the daughters
        # would overlap inside the parent.
        if T[0] < self.parent_length - 1e-12:
            raise GeometryError(
                f"the daughters overlap upstream of the junction (septum tip at x* = {T[0]:.4f} "
                f"< {self.parent_length}): widths {self.w1:g} + {self.w2:g} too large for "
                f"angles {self.angle1:g}, {self.angle2:g}")
        # Each inner wall must reach its outlet beyond the tip.
        for inner_end, d in ((E1i, v[6] - v[7]), (E2i, v[2] - v[1])):
            if np.dot(inner_end - T, d) <= 0.0:
                raise GeometryError("daughter_length is too short: an outlet lies upstream of the "
                                    "septum tip")
        if _self_intersects(v):
            raise GeometryError("the bifurcation polygon intersects itself for these parameters")

    @classmethod
    def from_split(cls, split, angle1, angle2, parent_length, daughter_length, min_width):
        """Daughters sharing the parent's width: w1 = split * w0, w2 = (1 - split) * w0."""
        lo, hi = split_bounds(min_width)
        if not lo <= split <= hi:
            raise GeometryError(f"split must lie in [{lo:g}, {hi:g}] (min_width {min_width:g} of "
                                f"the parent's {PARENT_WIDTH:g}), got {split}")
        return cls(PARENT_WIDTH * split, PARENT_WIDTH * (1.0 - split), angle1, angle2,
                   parent_length, daughter_length, min_width)

    def vertices(self):
        return vertices(self.w1, self.w2, math.radians(self.angle1), math.radians(self.angle2),
                        self.parent_length, self.daughter_length)

    def triangles(self):
        return triangles(self.w1, self.w2, math.radians(self.angle1), math.radians(self.angle2),
                         self.parent_length, self.daughter_length)

    def edges(self):
        """[(start, end, kind)] for the 9 edges, counter-clockwise."""
        v = self.vertices()
        return [(v[i], v[(i + 1) % len(v)], EDGE_KINDS[i]) for i in range(len(v))]

    def area(self):
        x, y = self.vertices().T
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def contains(self, points):
        """True for points strictly inside the polygon (even-odd rule)."""
        return _inside(self.vertices(), np.asarray(points, dtype=np.float64))

    def bounding_box(self):
        v = self.vertices()
        return v.min(axis=0), v.max(axis=0)

    def sample_interior(self, rng, n):
        """n points uniform over the interior, by rejection from the bounding box."""
        lo, hi = self.bounding_box()
        fraction = self.area() / float(np.prod(hi - lo))
        out = []
        need = n
        while need > 0:
            trial = lo + (hi - lo) * rng.random((int(need / fraction * 1.3) + 16, 2))
            keep = trial[self.contains(trial)]
            out.append(keep[:need])
            need -= len(out[-1])
        return np.concatenate(out)[:n]

    def sample_boundary(self, rng, kind, n):
        """(points, outward unit normals) uniform along the edges of one kind."""
        chosen = [(a, b) for a, b, k in self.edges() if k == kind]
        if not chosen:
            raise GeometryError(f"no edges of kind {kind!r}")
        lengths = np.array([np.linalg.norm(b - a) for a, b in chosen])
        which = rng.choice(len(chosen), size=n, p=lengths / lengths.sum())
        t = rng.random(n)
        starts = np.array([chosen[i][0] for i in which])
        ends = np.array([chosen[i][1] for i in which])
        points = starts + t[:, None] * (ends - starts)
        tangent = (ends - starts) / np.linalg.norm(ends - starts, axis=1, keepdims=True)
        normals = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)   # right of a CCW edge = out
        return points, normals

    def line_quadrature(self, kind, n):
        """Gauss-Legendre nodes, weights (by length) and outward normal on ONE edge."""
        chosen = [(a, b) for a, b, k in self.edges() if k == kind]
        if len(chosen) != 1:
            raise GeometryError(f"line_quadrature needs exactly one {kind!r} edge")
        a, b = chosen[0]
        g, w = np.polynomial.legendre.leggauss(n)
        length = float(np.linalg.norm(b - a))
        points = 0.5 * (a + b) + 0.5 * g[:, None] * (b - a)
        tangent = (b - a) / length
        return points, 0.5 * length * w, np.array([tangent[1], -tangent[0]])


def split_bounds(min_width):
    """The splits w1 / w0 that keep BOTH daughters at least min_width wide."""
    if not 0.0 < min_width < 0.5 * PARENT_WIDTH:
        raise GeometryError(f"min_width must lie in (0, {0.5 * PARENT_WIDTH:g}) so both daughters "
                            f"fit, got {min_width}")
    return min_width / PARENT_WIDTH, 1.0 - min_width / PARENT_WIDTH


def _inside(v, p):
    x, y = p[:, 0:1], p[:, 1:2]
    x1, y1 = v[:, 0][None, :], v[:, 1][None, :]
    x2, y2 = np.roll(v[:, 0], -1)[None, :], np.roll(v[:, 1], -1)[None, :]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    return (np.sum(crosses & (x < at), axis=1) % 2) == 1


def _segments_cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, q1) * orient(p1, p2, q2) < 0) and (orient(q1, q2, p1) * orient(q1, q2, p2) < 0)


def _self_intersects(v):
    n = len(v)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue   # adjacent through the wrap-around
            if _segments_cross(v[i], v[(i + 1) % n], v[j], v[(j + 1) % n]):
                return True
    return False
