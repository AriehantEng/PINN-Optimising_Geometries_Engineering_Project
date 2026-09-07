"""SPEC test 7 - non-dimensionalisation round trip, plus scaling consistency."""

import numpy as np
import pytest

from src.physics import analytic
from src.physics.scaling import (
    ROUNDTRIP_RTOL,
    Scales,
    recommended_pressure_scaling,
    scales_from_reynolds,
)

# Deliberately awkward SI values: nothing here is 1.0, so a dropped or
# duplicated factor cannot cancel itself out.
SI_CASES = [
    dict(h=2.5e-4, u_max=1.37e-2, rho=998.2, mu=1.002e-3),
    dict(h=1.0, u_max=1.0, rho=1.0, mu=1.0),
    dict(h=7.3e-6, u_max=4.1e-1, rho=1.225, mu=1.81e-5),
    dict(h=3.0, u_max=250.0, rho=13593.0, mu=1.526e-3),
]


@pytest.mark.parametrize("si", SI_CASES)
@pytest.mark.parametrize("pressure_scaling", ["viscous", "inertial"])
def test_roundtrip_si_to_nondim_to_si_is_machine_exact(si, pressure_scaling):
    """SPEC test 7: SI -> non-dim -> SI is exact to machine precision."""
    scales = Scales(pressure_scaling=pressure_scaling, **si)
    rng = np.random.default_rng(20240501)

    lengths = rng.uniform(-5.0, 5.0, 64) * si["h"]
    velocities = rng.uniform(-3.0, 3.0, 64) * si["u_max"]
    pressures = rng.uniform(-100.0, 100.0, 64) * scales.pressure_scale
    gradients = rng.uniform(-100.0, 100.0, 64) * scales.pressure_scale / si["h"]

    for values, to_nd, to_si in (
        (lengths, scales.length_to_nondim, scales.length_to_si),
        (velocities, scales.velocity_to_nondim, scales.velocity_to_si),
        (pressures, scales.pressure_to_nondim, scales.pressure_to_si),
        (
            gradients,
            scales.pressure_gradient_to_nondim,
            scales.pressure_gradient_to_si,
        ),
    ):
        np.testing.assert_allclose(
            to_si(to_nd(values)), values, rtol=ROUNDTRIP_RTOL, atol=0.0
        )


@pytest.mark.parametrize("si", SI_CASES)
def test_nondim_analytic_profile_matches_si_analytic_profile(si):
    """The oracle is the same solution in both worlds.

    Builds the SI Poiseuille profile from its SI pressure gradient, maps it
    through the transform, and checks it lands on u* = 1 - y*^2. This catches a
    transform that round-trips but is the wrong transform.
    """
    scales = Scales(pressure_scaling="viscous", **si)
    gradient = analytic.pressure_gradient_si(scales.u_max, scales.h, scales.mu)

    # The SI profile built from that gradient must have the stated U_max.
    np.testing.assert_allclose(
        analytic.u_max_si(gradient, scales.h, scales.mu),
        scales.u_max,
        rtol=ROUNDTRIP_RTOL,
        atol=0.0,
    )

    y_si = np.linspace(-scales.h, scales.h, 101)
    u_si_values = analytic.u_si(y_si, gradient, scales.h, scales.mu)

    np.testing.assert_allclose(
        scales.velocity_to_nondim(u_si_values),
        analytic.u_nd(scales.length_to_nondim(y_si)),
        rtol=1e-14,
        atol=1e-15,
    )


@pytest.mark.parametrize("si", SI_CASES)
def test_flow_rate_matches_integral_of_profile(si):
    """Q = 2 G h^3 / (3 mu) is the integral of the profile, not a pipe formula.

    Guards the h^3 (2D slot) versus r^4 (circular pipe) trap called out in the
    SPEC as a Phase 3 hazard.
    """
    scales = Scales(pressure_scaling="viscous", **si)
    gradient = analytic.pressure_gradient_si(scales.u_max, scales.h, scales.mu)

    y = np.linspace(-scales.h, scales.h, 2_000_001)
    quadrature = np.trapezoid(analytic.u_si(y, gradient, scales.h, scales.mu), y)

    np.testing.assert_allclose(
        analytic.flow_rate_si(gradient, scales.h, scales.mu),
        quadrature,
        rtol=1e-10,
        atol=0.0,
    )


@pytest.mark.parametrize("reynolds", [0.1, 1.0, 10.0, 50.0, 100.0])
def test_pressure_scales_differ_by_exactly_reynolds(reynolds):
    """p*_viscous = Re * p*_inertial, and the analytic gradients agree."""
    inertial = scales_from_reynolds(reynolds, "inertial")
    viscous = Scales(
        h=inertial.h,
        u_max=inertial.u_max,
        rho=inertial.rho,
        mu=inertial.mu,
        pressure_scaling="viscous",
    )
    np.testing.assert_allclose(
        inertial.pressure_scale / viscous.pressure_scale,
        reynolds,
        rtol=1e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        analytic.pressure_gradient_nd(reynolds, "viscous"),
        analytic.pressure_gradient_nd(reynolds, "inertial") * reynolds,
        rtol=1e-14,
        atol=0.0,
    )


@pytest.mark.parametrize("reynolds", [0.1, 1.0, 10.0, 100.0])
def test_scales_from_reynolds_reproduces_reynolds(reynolds):
    scales = scales_from_reynolds(reynolds, "inertial")
    np.testing.assert_allclose(scales.reynolds, reynolds, rtol=1e-14, atol=0.0)


def test_recommended_pressure_scaling_follows_spec():
    assert recommended_pressure_scaling(0.5) == "viscous"
    assert recommended_pressure_scaling(1.0) == "viscous"
    assert recommended_pressure_scaling(1.0001) == "inertial"
    assert recommended_pressure_scaling(100.0) == "inertial"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(h=0.0, u_max=1.0, rho=1.0, mu=1.0, pressure_scaling="viscous"),
        dict(h=1.0, u_max=-1.0, rho=1.0, mu=1.0, pressure_scaling="viscous"),
        dict(h=1.0, u_max=1.0, rho=1.0, mu=1.0, pressure_scaling="mixed"),
        dict(h=np.nan, u_max=1.0, rho=1.0, mu=1.0, pressure_scaling="viscous"),
    ],
)
def test_invalid_scales_raise(kwargs):
    """No silent fallbacks: a bad reference quantity raises immediately."""
    with pytest.raises(ValueError):
        Scales(**kwargs)
