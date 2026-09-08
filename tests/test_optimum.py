"""SPEC v2 S8 - the Phase 3 analytic optimum.

Pure analysis, no training. Phase 3 previously had no oracle; it now has one,
valid in the fully developed limit. The DEVIATION of the PINN optimum from
sqrt(2) is the Phase 3 result -- the law itself is not.
"""

import numpy as np
import pytest

from src.physics import analytic
from src.physics.optimum import (
    CIRCULAR_PIPE,
    OPTIMUM_HEIGHT_RATIO,
    PLANE_CHANNEL,
    area_law_exponent,
    flow_exponent,
    flow_rate_plane_channel,
    optimum_ratio_by_flow_exponent,
    optimum_ratio_by_lagrange,
    resistance_plane_channel,
)


def test_s8_1_both_derivation_routes_agree():
    """T-S8.1: Lagrange stationarity and the flow-exponent route give sqrt(2)."""
    lagrange = optimum_ratio_by_lagrange()
    exponent = optimum_ratio_by_flow_exponent(*PLANE_CHANNEL)
    np.testing.assert_allclose(lagrange, np.sqrt(2.0), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(exponent, np.sqrt(2.0), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(lagrange, exponent, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(OPTIMUM_HEIGHT_RATIO, np.sqrt(2.0), rtol=1e-15)


@pytest.mark.parametrize("mu", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("l0", [0.1, 1.0, 25.0])
@pytest.mark.parametrize("l1", [0.1, 1.0, 25.0])
def test_s8_1_optimum_is_independent_of_viscosity_and_lengths(mu, l0, l1):
    """The ratio must not depend on mu, lambda or either length.

    If it did, the 'analytic optimum' would be a function of the test case
    rather than an oracle.
    """
    np.testing.assert_allclose(
        optimum_ratio_by_lagrange(mu=mu, length_parent=l0, length_daughter=l1),
        np.sqrt(2.0),
        rtol=1e-12,
        atol=0.0,
    )


def test_s8_2_circular_pipe_specialisation_recovers_murray():
    """T-S8.2: guards against silently mixing 2D and 3D formulas.

    Plane channel R ~ h^-3, area ~ h  -> Q ~ h^2, ratio sqrt(2), h0^2 = h1^2 + h2^2.
    Circular pipe R ~ r^-4, volume ~ r^2 -> Q ~ r^3, ratio 2^(1/3), Murray's law.
    """
    np.testing.assert_allclose(flow_exponent(*PLANE_CHANNEL), 2.0, rtol=1e-15)
    np.testing.assert_allclose(flow_exponent(*CIRCULAR_PIPE), 3.0, rtol=1e-15)
    np.testing.assert_allclose(
        optimum_ratio_by_flow_exponent(*CIRCULAR_PIPE), 2.0 ** (1.0 / 3.0), rtol=1e-12
    )
    # The two must NOT be equal -- that is the whole point of the guard.
    assert not np.isclose(
        optimum_ratio_by_flow_exponent(*PLANE_CHANNEL),
        optimum_ratio_by_flow_exponent(*CIRCULAR_PIPE),
    )


def test_s8_2_area_law_exponent_is_two_not_three():
    """h_0^2 = h_1^2 + h_2^2 in 2D. The cube law is circular-pipe only."""
    assert area_law_exponent(*PLANE_CHANNEL) == 2.0
    assert area_law_exponent(*CIRCULAR_PIPE) == 3.0

    # Symmetric split, stated as the conserved quantity rather than the ratio.
    h1 = 1.0
    h0 = OPTIMUM_HEIGHT_RATIO * h1
    np.testing.assert_allclose(h0**2, h1**2 + h1**2, rtol=1e-12, atol=0.0)


def test_resistance_and_flow_rate_are_mutually_consistent():
    """R = dp/Q must reproduce the h^3 flux law from analytic.py."""
    mu, length, h, gradient = 1.7e-3, 3.2, 4.5e-4, 12.0
    flux = flow_rate_plane_channel(gradient, mu, h)
    np.testing.assert_allclose(
        flux, analytic.flow_rate_si(gradient, h, mu), rtol=1e-14, atol=0.0
    )
    pressure_drop = gradient * length
    np.testing.assert_allclose(
        resistance_plane_channel(mu, length, h), pressure_drop / flux,
        rtol=1e-14, atol=0.0,
    )


def test_parallel_daughters_at_the_optimum_beat_a_perturbed_split():
    """The stationary point is a MINIMUM of resistance at fixed area.

    Asserting stationarity alone would be satisfied by a maximum, which would
    make the Phase 3 oracle point the wrong way.
    """
    mu, l0, l1, area = 1.0, 1.0, 1.0, 10.0

    def total_resistance(ratio):
        # h_0 = ratio * h_1, with area 2 h_0 l_0 + 4 h_1 l_1 held fixed.
        h1 = area / (2.0 * ratio * l0 + 4.0 * l1)
        h0 = ratio * h1
        return 1.5 * mu * (l0 / h0**3 + l1 / (2.0 * h1**3))

    best = total_resistance(float(OPTIMUM_HEIGHT_RATIO))
    for perturbation in (0.7, 0.9, 1.1, 1.4, 2.0):
        assert total_resistance(float(OPTIMUM_HEIGHT_RATIO) * perturbation) > best


@pytest.mark.phase3
@pytest.mark.xfail(
    strict=True,
    reason=(
        "T-S8.3: as L_1/h_1 -> infinity the PINN optimum must approach sqrt(2). "
        "Requires the Phase 3 optimisation loop over the bifurcation geometry, "
        "which does not exist. Strict, so it fails loudly if it starts passing "
        "without this marker being removed."
    ),
)
def test_s8_3_pinn_optimum_approaches_the_analytic_value():
    raise NotImplementedError(
        "Phase 3 requirement. Once the parametric bifurcation solve exists, "
        "this must sweep L_1/h_1 upward and assert the recovered h_0/h_1 "
        "converges to sqrt(2) = 1.41421356..., reporting the deviation as the "
        "result. Junction losses make the deviation nonzero at finite L_1/h_1; "
        "that deviation IS the Phase 3 finding."
    )
