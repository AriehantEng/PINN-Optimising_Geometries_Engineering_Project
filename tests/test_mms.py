"""SPEC test 6 - method of manufactured solutions.

The residual operator is applied to a known smooth field and compared against a
forcing derived analytically BY HAND below. The two paths share nothing: the
code under test reaches the answer by forward-mode autodiff, this file reaches
it by pencil-and-paper differentiation transcribed into closed form.

Manufactured field
------------------
    u(x, y) =  A sin(x) cos(y)
    v(x, y) = -B cos(x) sin(y)
    p(x, y) =  C cos(x) sin(y)

with A != B, so the field is deliberately NOT divergence free and the
continuity residual is exercised with a non-zero forcing too.

Derivatives (by hand):
    u_x =  A cos x cos y      u_y = -A sin x sin y      lap u = -2A sin x cos y
    v_x =  B sin x sin y      v_y = -B cos x cos y      lap v =  2B cos x sin y
    p_x = -C sin x sin y      p_y =  C cos x cos y

Forcing, inertial scaling (R = conv + grad p - lap/Re):
    f_x = A^2 sin x cos x cos^2 y + A B sin x cos x sin^2 y
          - C sin x sin y + (2A/Re) sin x cos y
    f_y = A B sin^2 x sin y cos y + B^2 cos^2 x sin y cos y
          + C cos x cos y - (2B/Re) cos x sin y
    f_c = (A - B) cos x cos y

Forcing, viscous scaling (R = Re*conv + grad p - lap):
    f_x = Re (A^2 sin x cos x cos^2 y + A B sin x cos x sin^2 y)
          - C sin x sin y + 2A sin x cos y
    f_y = Re (A B sin^2 x sin y cos y + B^2 cos^2 x sin y cos y)
          + C cos x cos y - 2B cos x sin y
    f_c = (A - B) cos x cos y
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.physics import residuals
from src.physics.derivatives import batched_field_derivatives

A, B, C = 1.3, 0.7, 0.9

#: Fields and forcings are O(1); float64 autodiff against a closed form should
#: agree to a few ulp. 1e-13 leaves room for the ~10 rounding operations in the
#: closed-form expressions without being loose enough to hide a real error.
MMS_ATOL = 1e-13


def manufactured_field(xy):
    x, y = xy[0], xy[1]
    return jnp.array(
        [
            A * jnp.sin(x) * jnp.cos(y),
            -B * jnp.cos(x) * jnp.sin(y),
            C * jnp.cos(x) * jnp.sin(y),
        ]
    )


def expected_forcing(x, y, reynolds, pressure_scaling):
    """Hand-derived forcing. Deliberately written out in full, not factored."""
    sx, cx, sy, cy = np.sin(x), np.cos(x), np.sin(y), np.cos(y)

    convective_x = A**2 * sx * cx * cy**2 + A * B * sx * cx * sy**2
    convective_y = A * B * sx**2 * sy * cy + B**2 * cx**2 * sy * cy
    grad_p_x = -C * sx * sy
    grad_p_y = C * cx * cy
    lap_u = -2.0 * A * sx * cy
    lap_v = 2.0 * B * cx * sy

    if pressure_scaling == "inertial":
        f_x = convective_x + grad_p_x - lap_u / reynolds
        f_y = convective_y + grad_p_y - lap_v / reynolds
    elif pressure_scaling == "viscous":
        f_x = reynolds * convective_x + grad_p_x - lap_u
        f_y = reynolds * convective_y + grad_p_y - lap_v
    else:
        raise ValueError(pressure_scaling)

    f_c = (A - B) * cx * cy
    return f_x, f_y, f_c


@pytest.mark.parametrize("reynolds", [1.0, 10.0, 50.0, 100.0])
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_residual_operator_reproduces_manufactured_forcing(
    reynolds, pressure_scaling
):
    """SPEC test 6."""
    rng = np.random.default_rng(76543)
    points = rng.uniform([0.0, -1.0], [20.0, 1.0], size=(512, 2))

    d = batched_field_derivatives(manufactured_field, jnp.asarray(points))
    got_x, got_y, got_c = residuals.navier_stokes(d, reynolds, pressure_scaling)

    want_x, want_y, want_c = expected_forcing(
        points[:, 0], points[:, 1], reynolds, pressure_scaling
    )

    np.testing.assert_allclose(np.asarray(got_x), want_x, rtol=0.0, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(got_y), want_y, rtol=0.0, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(got_c), want_c, rtol=0.0, atol=MMS_ATOL)


@pytest.mark.parametrize("reynolds", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_individual_terms_match_hand_derivatives(reynolds, pressure_scaling):
    """Each term is tested on its own, not only the assembled residual.

    A sign error in one term that is cancelled by a sign error in another would
    survive the assembled test above; this one catches it.
    """
    rng = np.random.default_rng(24680)
    points = rng.uniform([0.0, -1.0], [20.0, 1.0], size=(256, 2))
    x, y = points[:, 0], points[:, 1]
    sx, cx, sy, cy = np.sin(x), np.cos(x), np.sin(y), np.cos(y)

    d = batched_field_derivatives(manufactured_field, jnp.asarray(points))

    np.testing.assert_allclose(np.asarray(d.u), A * sx * cy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.v), -B * cx * sy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.p), C * cx * sy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.u_x), A * cx * cy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.u_y), -A * sx * sy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.v_x), B * sx * sy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.v_y), -B * cx * cy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.p_x), -C * sx * sy, atol=MMS_ATOL)
    np.testing.assert_allclose(np.asarray(d.p_y), C * cx * cy, atol=MMS_ATOL)

    np.testing.assert_allclose(
        np.asarray(residuals.convective_x(d.u, d.v, d.u_x, d.u_y)),
        A**2 * sx * cx * cy**2 + A * B * sx * cx * sy**2,
        atol=MMS_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(residuals.convective_y(d.u, d.v, d.v_x, d.v_y)),
        A * B * sx**2 * sy * cy + B**2 * cx**2 * sy * cy,
        atol=MMS_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(residuals.laplacian(d.u_xx, d.u_yy)),
        -2.0 * A * sx * cy,
        atol=MMS_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(residuals.laplacian(d.v_xx, d.v_yy)),
        2.0 * B * cx * sy,
        atol=MMS_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(residuals.divergence(d.u_x, d.v_y)),
        (A - B) * cx * cy,
        atol=MMS_ATOL,
    )


@pytest.mark.parametrize("reynolds", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_analytic_poiseuille_has_zero_residual(reynolds, pressure_scaling):
    """The oracle must satisfy the operator exactly. Zero forcing, zero slack."""
    from src.physics.analytic import p_nd, pressure_gradient_nd

    length = 20.0
    dpdx = pressure_gradient_nd(reynolds, pressure_scaling)

    def field(xy):
        return jnp.array([1.0 - xy[1] ** 2, 0.0, dpdx * (xy[0] - length)])

    rng = np.random.default_rng(13579)
    points = jnp.asarray(rng.uniform([0.0, -1.0], [length, 1.0], size=(512, 2)))
    d = batched_field_derivatives(field, points)
    r_x, r_y, r_c = residuals.navier_stokes(d, reynolds, pressure_scaling)

    assert float(jnp.max(jnp.abs(r_x))) < 1e-14
    assert float(jnp.max(jnp.abs(r_y))) < 1e-14
    assert float(jnp.max(jnp.abs(r_c))) < 1e-14

    # And the gauge: p*(L*) = 0.
    np.testing.assert_allclose(
        p_nd(length, length, reynolds, pressure_scaling), 0.0, atol=0.0
    )


def test_unknown_pressure_scaling_raises():
    """No silent fallbacks."""
    d = batched_field_derivatives(
        manufactured_field, jnp.asarray([[0.3, 0.4]])
    )
    with pytest.raises(ValueError, match="unknown pressure_scaling"):
        residuals.momentum_x(d, 10.0, "isothermal")
