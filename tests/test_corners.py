"""Stokes corner modes (bifurcation/corners.py): the eigenvalues, the modes'
wall conditions, their pressure partners and the cutoff."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bifurcation import corners as C

jax.config.update("jax_enable_x64", True)

CENTRE = jnp.array([0.3, -0.2])
TURN = 0.7                     # the bisector's direction: any rotation must work


def _mode_functions(z, angle, family):
    a = jnp.asarray(C.mode_coefficients(z, angle, family))
    bisector = (math.cos(TURN), math.sin(TURN))

    def psi(q):
        r, theta = C.local_polar(q, CENTRE, bisector)
        return C.stream(r, theta, z, a)

    def pressure(q):
        r, theta = C.local_polar(q, CENTRE, bisector)
        return C.pressure(r, theta, z, a)

    def velocity(q):
        g = jax.grad(psi)(q)
        return jnp.array([g[1], -g[0]])

    return psi, pressure, velocity


def _point(r, theta):
    return CENTRE + r * jnp.array([math.cos(theta + TURN), math.sin(theta + TURN)])


def _modes():
    out = []
    for degrees in (195.0, 210.0, 240.0, 270.0, 300.0):
        angle = math.radians(degrees)
        out += [(angle, family, z) for family in C.FAMILIES for z in C.eigenvalues(angle, family, 1.0)]
    return out


def test_the_leading_exponents_are_the_papers_table_1():
    """Table 1 of the paper (smallest root, three decimals), and the L-shaped
    corner's two classical roots, 0.5444837 and 0.9085292."""
    table = {195.0: 0.857, 210.0: 0.752, 240.0: 0.616, 270.0: 0.544, 300.0: 0.512}
    for degrees, expected in table.items():
        angle = math.radians(degrees)
        leading = min(z for family in C.FAMILIES for z in C.eigenvalues(angle, family, 1.0))
        assert abs(leading - expected) < 5e-4
    right = math.radians(270.0)
    assert abs(C.eigenvalues(right, "even", 1.0)[0] - 0.5444837) < 1e-7
    assert abs(C.eigenvalues(right, "odd", 1.0)[0] - 0.9085292) < 1e-7


def test_the_odd_family_is_regular_below_257_degrees():
    """The odd family's first root crosses z = 1 where tan(Theta) = Theta,
    Theta = 257.45 degrees: below it every odd mode has bounded gradients."""
    assert C.eigenvalues(math.radians(255.0), "odd", 1.0) == []
    assert len(C.eigenvalues(math.radians(260.0), "odd", 1.0)) == 1


def test_a_non_reflex_corner_is_refused():
    with pytest.raises(ValueError, match="reflex"):
        C.eigenvalues(math.radians(150.0), "even", 1.0)


@pytest.mark.parametrize("angle, family, z", _modes())
def test_each_mode_is_a_root_and_meets_no_slip_on_both_walls(angle, family, z):
    assert abs(C.condition(z, angle, family)) < 1e-14
    psi, _, velocity = _mode_functions(z, angle, family)
    for side in (-1.0, 1.0):
        for r in (0.05, 0.3):
            q = _point(r, side * 0.5 * angle)
            assert abs(float(psi(q))) < 1e-15
            assert float(jnp.linalg.norm(velocity(q))) < 1e-14
    inside = _point(0.3, 0.1 * angle)
    assert float(jnp.linalg.norm(velocity(inside))) > 0.1        # not the trivial mode


@pytest.mark.parametrize("angle, family, z", _modes())
def test_each_mode_with_its_pressure_solves_stokes(angle, family, z):
    """lap(u) = grad(p) at points across the sector, for any rotation of it."""
    _, pressure, velocity = _mode_functions(z, angle, family)

    def laplacian(q):
        h = jax.jacfwd(jax.jacfwd(velocity))(q)
        return h[:, 0, 0] + h[:, 1, 1]

    rng = np.random.default_rng(0)
    for _ in range(12):
        q = _point(rng.uniform(0.01, 0.5), rng.uniform(-0.49, 0.49) * angle)
        lap, grad_p = np.asarray(laplacian(q)), np.asarray(jax.grad(pressure)(q))
        assert np.max(np.abs(lap - grad_p)) < 1e-11 * np.max(np.abs(lap))


@pytest.mark.parametrize("angle, family, z", _modes())
def test_the_velocity_goes_like_r_to_the_z(angle, family, z):
    _, _, velocity = _mode_functions(z, angle, family)
    theta = 0.2 * angle
    near, far = (float(jnp.linalg.norm(velocity(_point(r, theta)))) for r in (0.01, 0.02))
    assert abs(far / near - 2.0 ** z) < 1e-12


def test_the_modes_are_scaled_to_unit_peak():
    for angle, family, z in _modes():
        a = C.mode_coefficients(z, angle, family)
        theta = np.linspace(-0.5 * angle, 0.5 * angle, 20001)
        assert abs(np.max(np.abs(C.angular(a, z, theta, np))) - 1.0) < 1e-6


def test_the_cutoff_is_one_at_the_corner_and_c3_at_its_radius():
    radius = 0.4
    radial = [lambda r: C.cutoff(r, radius)]
    for _ in range(3):
        radial.append(jax.grad(radial[-1]))
    assert float(radial[0](0.0)) == 1.0
    for k, f in enumerate(radial):
        assert abs(float(f(radius))) < 1e-15, k
        assert float(f(radius + 0.01)) == 0.0
    # in x and y it is a polynomial: smooth through the corner itself
    chi = lambda q: C.cutoff(jnp.sqrt(q @ q), radius)                     # noqa: E731
    hessian = np.asarray(jax.hessian(chi)(jnp.array([1e-9, 0.0])))
    assert np.allclose(hessian, -8.0 / radius**2 * np.eye(2), rtol=1e-6)


def test_the_sector_of_an_l_shaped_corner():
    """Counter-clockwise L: the reflex vertex (1, 1) has 270 degrees of fluid,
    its bisector pointing into the L's elbow, away from the missing square."""
    v = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    bisector, angle = C.sector(v, 3)
    assert abs(angle - 1.5 * math.pi) < 1e-14
    assert np.allclose(bisector, -np.array([1.0, 1.0]) / math.sqrt(2.0), atol=1e-14)
    _, convex = C.sector(v, 0)
    assert abs(convex - 0.5 * math.pi) < 1e-14
    corner = C.corner(v, 3, 1.0)
    assert [round(m.z, 4) for m in corner.modes] == [0.5445, 0.9085]


def test_a_cutoff_reaching_another_edge_is_refused():
    v = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    C.check_cutoff(v, 3, 0.9)
    with pytest.raises(ValueError, match="within the corner cutoff"):
        C.check_cutoff(v, 3, 1.1)                 # the far walls x = 0 and y = 0
    w = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [1.0, 4.0], [1.0, 4.5], [0.0, 4.5]])
    with pytest.raises(ValueError, match="shorter than the cutoff"):
        C.check_cutoff(w, 3, 0.6)                 # its own edge to (1, 4.5) is 0.5 long


# -- beyond z = 1: the next real roots and the first complex pairs ------------ #

def test_the_next_eigenvalues_up_to_two():
    """Every root with Re z < 2 at the bifurcation's three corner angles: real
    ones bracketed, complex ones by Newton, each a root to round-off."""
    expected = {210.0: {"even": [0.752], "odd": [1.4858]},
                240.0: {"even": [0.6157, complex(1.8335, 0.2523)], "odd": [1.1489]},
                270.0: {"even": [0.5445, complex(1.6293, 0.2313)], "odd": [0.9085]}}
    for degrees, families in expected.items():
        angle = math.radians(degrees)
        for family, roots in families.items():
            found = C.eigenvalues(angle, family, 2.0)
            assert len(found) == len(roots), (degrees, family, found)
            for z, ref in zip(found, roots):
                assert isinstance(z, complex) == isinstance(ref, complex)
                assert abs(z - ref) < 1e-4
                assert abs(C.condition(z, angle, family)) < 1e-12


@pytest.mark.parametrize("degrees, family", [(240.0, "even"), (270.0, "even")])
@pytest.mark.parametrize("amplitude", [1.0, 1j, 0.6 - 0.8j])
def test_a_complex_mode_is_a_real_stokes_flow_with_no_slip(degrees, family, amplitude):
    """Re[c r^(1+z) f] and Re[c P] for a complex root z: no-slip on both walls,
    and lap(u) = grad(p), for any complex amplitude c."""
    angle = math.radians(degrees)
    z = C.eigenvalues(angle, family, 2.0)[1]
    assert isinstance(z, complex)
    a = jnp.asarray(C.mode_coefficients(z, angle, family))
    bisector = (math.cos(TURN), math.sin(TURN))

    def psi(q):
        r, theta = C.local_polar(q, CENTRE, bisector)
        return jnp.real(amplitude * C.stream(r, theta, z, a))

    def pressure(q):
        r, theta = C.local_polar(q, CENTRE, bisector)
        return jnp.real(amplitude * C.pressure(r, theta, z, a))

    def velocity(q):
        g = jax.grad(psi)(q)
        return jnp.array([g[1], -g[0]])

    for side in (-1.0, 1.0):
        q = _point(0.2, side * 0.5 * angle)
        assert abs(float(psi(q))) < 1e-15 and float(jnp.linalg.norm(velocity(q))) < 1e-14
    rng = np.random.default_rng(1)
    for _ in range(8):
        q = _point(rng.uniform(0.01, 0.5), rng.uniform(-0.49, 0.49) * angle)
        h = jax.jacfwd(jax.jacfwd(velocity))(q)
        lap = np.asarray(h[:, 0, 0] + h[:, 1, 1])
        assert np.max(np.abs(lap - np.asarray(jax.grad(pressure)(q)))) < 1e-11 * np.max(np.abs(lap))


def test_a_complex_mode_takes_two_amplitudes():
    v = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    corner = C.corner(v, 3, 2.0)
    assert [m.amplitudes for m in corner.modes] == [1, 2, 1]        # 0.54, 1.63+0.23i, 0.91
