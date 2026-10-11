"""Stokes corner modes: the singular part of the flow at a reflex corner.

Where two straight no-slip walls meet with a fluid-side angle Theta > 180
degrees (the bifurcation's corners B, C and the septum tip T), Stokes flow near
the corner is a sum of separable modes (Dean & Montagnon 1949; Moffatt 1964;
the paper's Appendix A.8). In polar coordinates (r, theta) about the corner,
theta measured from the bisector of the fluid sector |theta| <= Theta / 2,

    psi = r^(1+z) f(theta),
    f   = a1 cos((1+z) theta) + a2 sin((1+z) theta)
        + a3 cos((z-1) theta) + a4 sin((z-1) theta),

solves the biharmonic equation, and no-slip on both walls (psi = d(psi)/d(theta)
= 0 at theta = +-Theta/2) separates into two families:

    even   psi even about the bisector   sin(z Theta) = -z sin(Theta)
    odd    psi odd about the bisector    sin(z Theta) =  z sin(Theta)

A root z < 1 makes the velocity gradients, the vorticity and the pressure
singular, like r^(z-1); a root with 1 < Re z < 2 leaves the gradients bounded
but the second derivatives, and so the momentum residual, singular. Beyond the
first roots they come in complex pairs z, conj(z) (Moffatt 1964): the mode is
then Re[c r^(1+z) f(theta)] with a complex amplitude c, and z, f and the
pressure below are the same formulae in complex arithmetic. The bifurcation PINN writes psi = G + D^2 N with smooth
G, D and network N, and D ~ r at the corner, so N would have to grow like
r^(z-1): no network can, and the residual grows like 1/r near every reflex
corner (bifurcation/flow.py). Adding each mode with z < 1 explicitly, with a
trained amplitude, leaves N a bounded remainder.

Each mode's pressure partner follows from Stokes' equations, grad p = nu
lap(u) with u = d(psi)/dy, v = -d(psi)/dx: p and nu lap(psi) are harmonic
conjugates, and only the (z-1) terms of f have a non-zero Laplacian,
4z r^(z-1) (a3 cos((z-1) theta) + a4 sin((z-1) theta)), so

    p = -4 nu z r^(z-1) (a3 sin((z-1) theta) - a4 cos((z-1) theta)).

A mode is used inside a cutoff radius R only, multiplied by
chi(r) = (1 - (r/R)^2)^4, which is a polynomial in x and y, so smooth at the
corner, and whose first three derivatives vanish at r = R, so psi stays C^3.
Within R of the corner only its own two walls may lie (`check_cutoff`): there
chi * mode keeps psi = d(psi)/dn = 0, since a radial chi has no normal
derivative along a straight wall through the corner.

Every function here is pure and takes non-dimensional inputs.
"""

from __future__ import annotations

import cmath
import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

FAMILIES = ("even", "odd")

#: Points per unit of z on which the eigenvalue condition is scanned for sign changes.
SCAN = 20000

#: Starting points of the complex search: real parts per unit of z, imaginary parts.
COMPLEX_STARTS = (40, np.linspace(0.02, 1.5, 30))


def condition(z, angle, family):
    """sin(z Theta) +- z sin(Theta): zero at the family's eigenvalues."""
    sign = 1.0 if family == "even" else -1.0
    return np.sin(z * angle) + sign * z * np.sin(angle)


def eigenvalues(angle, family, upper):
    """The family's roots with 0 < Re z < `upper` for fluid-side angle `angle`
    (radians), by ascending real part: real roots as floats, complex ones as
    complex with Im z > 0 (each stands for its conjugate pair). z = 0, and
    z = 1 for the odd family, are trivial (f = 0)."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if not math.pi < angle < 2.0 * math.pi:
        raise ValueError("a reflex corner has a fluid-side angle between 180 and 360 degrees")
    sign = 1.0 if family == "even" else -1.0

    def trivial(z):
        return family == "odd" and abs(z - 1.0) < 1e-9

    z = np.linspace(1e-6, upper, int(SCAN * upper))
    g = condition(z, angle, family)
    roots = []
    for k in np.nonzero(np.sign(g[:-1]) != np.sign(g[1:]))[0]:
        root = brentq(condition, z[k], z[k + 1], args=(angle, family), xtol=1e-15,
                      rtol=4 * np.finfo(float).eps)
        if not trivial(root):
            roots.append(root)
    per_unit, imaginary = COMPLEX_STARTS
    for re in np.linspace(0.02, upper, max(2, int(per_unit * upper))):
        for im in imaginary:
            w = complex(re, im)
            for _ in range(80):
                step = ((cmath.sin(w * angle) + sign * w * math.sin(angle))
                        / (angle * cmath.cos(w * angle) + sign * math.sin(angle)))
                w -= step
                if abs(step) < 1e-15 or abs(w) > 1e3:
                    break
            if (abs(cmath.sin(w * angle) + sign * w * math.sin(angle)) < 1e-12
                    and w.imag > 1e-9 and 0.0 < w.real < upper
                    and not any(isinstance(r, complex) and abs(r - w) < 1e-8 for r in roots)):
                roots.append(w)
    return sorted(roots, key=lambda r: (r.real, abs(r.imag) if isinstance(r, complex) else 0.0))


def mode_coefficients(z, angle, family):
    """(a1, a2, a3, a4) of the mode with eigenvalue z, real or complex, scaled so
    that max |f| over the fluid sector is 1."""
    half = 0.5 * angle
    if family == "even":
        a = np.array([cmath.cos((z - 1.0) * half), 0.0, -cmath.cos((1.0 + z) * half), 0.0])
    else:
        a = np.array([0.0, cmath.sin((z - 1.0) * half), 0.0, -cmath.sin((1.0 + z) * half)])
    if not isinstance(z, complex):
        a = a.real
    theta = np.linspace(-half, half, 4001)
    return a / np.max(np.abs(angular(a, z, theta, np)))


def angular(a, z, theta, xp=jnp):
    """f(theta) for coefficients a."""
    return (a[0] * xp.cos((1.0 + z) * theta) + a[1] * xp.sin((1.0 + z) * theta)
            + a[2] * xp.cos((z - 1.0) * theta) + a[3] * xp.sin((z - 1.0) * theta))


def stream(r, theta, z, a):
    """The mode's stream function r^(1+z) f(theta)."""
    return r ** (1.0 + z) * angular(a, z, theta)


def pressure(r, theta, z, a):
    """The mode's Stokes pressure per unit viscosity, -4 z r^(z-1) (a3 sin((z-1)theta)
    - a4 cos((z-1)theta)): grad of it equals lap(u) for u = (d psi/dy, -d psi/dx)."""
    return -4.0 * z * r ** (z - 1.0) * (a[2] * jnp.sin((z - 1.0) * theta)
                                         - a[3] * jnp.cos((z - 1.0) * theta))


def cutoff(r, radius):
    """chi = (1 - (r/R)^2)^4 inside R, 0 outside: C^3 everywhere, smooth at r = 0."""
    q = 1.0 - (r / radius) ** 2
    return jnp.where(q > 0.0, q, 0.0) ** 4


class Mode(NamedTuple):
    z: float | complex        # complex: Re[c r^(1+z) f], c = two trained amplitudes
    family: str
    coefficients: tuple

    @property
    def amplitudes(self):
        """Trained amplitudes it takes: 1 for a real eigenvalue, 2 for a complex pair."""
        return 2 if isinstance(self.z, complex) else 1


class Corner(NamedTuple):
    """One reflex corner: its vertex index in the polygon, the unit bisector of
    its fluid sector, its fluid-side angle (radians) and its modes."""
    vertex: int
    bisector: tuple
    angle: float
    modes: tuple


def sector(vertices, i):
    """(unit bisector, fluid-side angle) at vertex i of a counter-clockwise
    polygon, whose fluid lies to the left of every edge."""
    v = np.asarray(vertices, dtype=np.float64)
    before = v[i] - v[i - 1]
    after = v[(i + 1) % len(v)] - v[i]
    start = math.atan2(after[1], after[0])            # the outgoing wall
    end = math.atan2(-before[1], -before[0])          # the incoming wall, seen from the corner
    angle = (end - start) % (2.0 * math.pi)           # counter-clockwise, through the fluid
    middle = start + 0.5 * angle
    return (math.cos(middle), math.sin(middle)), angle


def corner(vertices, i, upper):
    """The Corner at vertex i, with every mode of either family that has Re z < upper."""
    bisector, angle = sector(vertices, i)
    modes = tuple(Mode(z, family, tuple(mode_coefficients(z, angle, family)))
                  for family in FAMILIES for z in eigenvalues(angle, family, upper))
    return Corner(i, bisector, angle, modes)


def local_polar(xy, centre, bisector):
    """(r, theta) of a point about `centre`, theta from the bisector, in (-pi, pi]."""
    d = xy - centre
    along = d[0] * bisector[0] + d[1] * bisector[1]
    across = bisector[0] * d[1] - bisector[1] * d[0]
    return jnp.sqrt(d[0] ** 2 + d[1] ** 2), jnp.arctan2(across, along)


def check_cutoff(vertices, i, radius):
    """Raise ValueError unless, within `radius` of vertex i, the polygon's
    boundary is the two edges meeting there: every other edge farther away,
    both of those longer."""
    v = np.asarray(vertices, dtype=np.float64)
    n = len(v)
    for j in range(n):
        a, b = v[j], v[(j + 1) % n]
        if j in (i, (i - 1) % n):
            if np.linalg.norm(b - a) <= radius:
                raise ValueError(f"an edge at vertex {i} is shorter than the cutoff {radius:g}")
            continue
        t = np.clip(np.dot(v[i] - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        gap = np.linalg.norm(v[i] - (a + t * (b - a)))
        if gap <= radius:
            raise ValueError(f"edge {j} lies {gap:.4f} from vertex {i}, within the corner "
                             f"cutoff {radius:g}")
