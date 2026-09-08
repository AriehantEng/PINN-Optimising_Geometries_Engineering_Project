"""SPEC v2 S4 - the coupled psi-omega system and pressure recovery."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.geometry import lifts
from src.physics import analytic, streamfunction
from src.physics.derivatives import batched_stream_derivatives

RE_VALUES = [1.0, 10.0, 50.0, 100.0]


def analytic_stream(flux=analytic.FLUX_ND):
    """Case A as (psi, omega).

    psi = (3Q/4)(y - y^3/3), so u = psi_y = (3Q/4)(1 - y^2) and for Q = 4/3 that
    is 1 - y^2. omega = -lap psi = -psi_yy = (3Q/4)(2y), which is 2y at Q = 4/3.
    """

    def stream(xy):
        y = xy[1]
        return jnp.array([lifts.cubic(y, flux), 0.75 * flux * 2.0 * y])

    return stream


@pytest.mark.parametrize("reynolds", RE_VALUES)
def test_s4_1_residuals_vanish_on_the_analytic_solution(reynolds):
    """T-S4.1: r_1 and r_2 reduce to the analytic Case A solution to 1e-12."""
    rng = np.random.default_rng(4242)
    points = jnp.asarray(rng.uniform([0.0, -1.0], [8.0, 1.0], size=(512, 2)))
    d = batched_stream_derivatives(analytic_stream(), points)
    r1, r2 = streamfunction.residuals(d, reynolds)
    assert float(jnp.max(jnp.abs(r1))) < 1e-12
    assert float(jnp.max(jnp.abs(r2))) < 1e-12


def test_s4_1_residuals_are_not_vacuous():
    """Negative control: a wrong field must produce a non-zero residual."""
    def wrong(xy):
        return jnp.array([xy[1] - xy[1] ** 3 / 3.0, 2.0 * xy[1] + 0.3])

    points = jnp.asarray([[1.0, 0.2], [3.0, -0.5]])
    r1, _ = streamfunction.residuals(batched_stream_derivatives(wrong, points), 10.0)
    assert float(jnp.max(jnp.abs(r1))) > 0.29


@pytest.mark.parametrize("reynolds", RE_VALUES)
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_s4_2_pressure_recovery_is_path_independent(reynolds, pressure_scaling):
    """T-S4.2: two distinct paths must agree to 1e-8.

    Checked on the analytic solution, where grad p genuinely IS a gradient. On
    a trained network the disagreement measures how far the solution is from
    satisfying momentum, which is why it is also logged for every psi-omega run
    rather than only asserted here.
    """
    length = 8.0
    velocity = lambda xy: jnp.array([1.0 - xy[1] ** 2, 0.0])  # noqa: E731
    gauge = (length, 0.0)

    for probe in ([1.0, 0.5], [4.0, -0.7], [7.5, 0.9], [0.2, 0.0]):
        x_first = streamfunction.recover_pressure(
            velocity, probe, gauge, reynolds, pressure_scaling, path="x_first"
        )
        y_first = streamfunction.recover_pressure(
            velocity, probe, gauge, reynolds, pressure_scaling, path="y_first"
        )
        assert abs(x_first - y_first) < 1e-8


@pytest.mark.parametrize("reynolds", RE_VALUES)
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_s4_2_recovered_pressure_matches_the_analytic_field(reynolds, pressure_scaling):
    """Path independence alone would be satisfied by recovering the wrong field."""
    length = 8.0
    velocity = lambda xy: jnp.array([1.0 - xy[1] ** 2, 0.0])  # noqa: E731

    for probe in ([1.0, 0.5], [4.0, -0.7], [7.5, 0.9]):
        recovered = streamfunction.recover_pressure(
            velocity, probe, (length, 0.0), reynolds, pressure_scaling
        )
        exact = float(analytic.p_nd(probe[0], length, reynolds, pressure_scaling))
        assert abs(recovered - exact) < 1e-10


# --------------------------------------------------------------------------- #
# Lifts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["cubic", "sinusoidal"])
@pytest.mark.parametrize("flux", [4.0 / 3.0, 2.0, 0.5])
def test_lifts_satisfy_their_boundary_conditions(name, flux):
    """psi_lift(+-1) = +-Q/2 and psi_lift'(+-1) = 0, for both lifts."""
    lift = lifts.get(name)
    for wall in (1.0, -1.0):
        value = float(lift(jnp.asarray(wall), flux))
        slope = float(jax.grad(lambda t: lift(t, flux))(jnp.asarray(wall)))
        np.testing.assert_allclose(value, wall * flux / 2.0, rtol=1e-14, atol=1e-15)
        assert abs(slope) < 1e-15


def test_cubic_lift_is_the_exact_case_a_solution_and_sinusoidal_is_not():
    """The degeneracy S4 warns about, asserted rather than assumed.

    Under the cubic lift the correct Case A answer is N_psi = 0, so Case A
    stops testing expressivity. The sinusoidal lift carries the same flux with
    the wrong shape, so N_psi must learn a non-zero correction.
    """
    y = jnp.linspace(-1.0, 1.0, 401)
    flux = analytic.FLUX_ND

    cubic_u = jax.vmap(jax.grad(lambda t: lifts.cubic(t, flux)))(y)
    np.testing.assert_allclose(
        np.asarray(cubic_u), analytic.u_nd(np.asarray(y)), rtol=1e-13, atol=1e-14
    )

    sine_u = jax.vmap(jax.grad(lambda t: lifts.sinusoidal(t, flux)))(y)
    assert not np.allclose(
        np.asarray(sine_u), analytic.u_nd(np.asarray(y)), rtol=1e-3, atol=1e-3
    )

    assert "cubic" in lifts.DEGENERATE_LIFTS
    assert "sinusoidal" not in lifts.DEGENERATE_LIFTS


def test_unknown_lift_raises():
    with pytest.raises(ValueError, match="unknown lift"):
        lifts.get("parabolic")
