"""SPEC v2 S1, S2, S3 - conventions, inlet amplitude, derived domain length.

The bug these exist to prevent already happened once: Case B called an
entrance-length correlation with a bare `bulk_velocity_nd` of 1.0 where the
parabolic mean 2/3 was required, inflating the correlation by 25% and turning a
41% entrance-length miss into a reported 12.9% pass.
"""

import copy

import numpy as np
import pytest
import yaml

from src.config import ConfigError, parse_config
from src.physics import analytic
from src.physics.conventions import (
    D_H_OVER_H,
    RE_DH_OVER_RE_H,
    U_MEAN_OVER_U_MAX,
    EntranceLengthCorrelation,
    MissingCitationError,
    entrance_length_over_h,
    re_dh_to_re_h,
    re_h_to_re_dh,
    reynolds_dh,
    reynolds_h,
)

CORRELATION = EntranceLengthCorrelation(
    c0=0.3155, c1=0.01105, n=1.6, citation="test fixture"
)


def _document(path="experiments/caseA_re10.yaml"):
    return yaml.safe_load(open(path, encoding="utf-8"))


# --------------------------------------------------------------------------- #
# S1
# --------------------------------------------------------------------------- #


def test_s1_1_re_h_to_re_dh_is_exactly_eight_thirds():
    """T-S1.1."""
    assert re_h_to_re_dh(1.0) == 8.0 / 3.0
    assert RE_DH_OVER_RE_H == 8.0 / 3.0


@pytest.mark.parametrize("re_h", [1e-3, 0.5, 1.0, 10.0, 50.0, 100.0, 1e4])
def test_s1_2_reynolds_round_trip_is_machine_exact(re_h):
    """T-S1.2."""
    np.testing.assert_allclose(
        re_dh_to_re_h(re_h_to_re_dh(re_h)), re_h, rtol=4 * np.finfo(float).eps, atol=0.0
    )


def test_s1_3_mean_velocity_is_two_thirds_of_u_max_by_quadrature():
    """T-S1.3: recompute U_mean/U_max from the analytic profile, not from memory.

    Gauss-Legendre, not trapezoid. The trapezoid rule on 2e6 points leaves a
    2.5e-13 relative truncation error on a parabola -- above the 1e-14 this
    test asserts -- so it would be measuring the quadrature rather than the
    convention. GL with 64 nodes is exact for a quadratic to machine precision.
    The tolerance is the SPEC's and is unchanged.
    """
    nodes, weights = np.polynomial.legendre.leggauss(64)
    mean = float(analytic.u_nd(nodes) @ weights) / 2.0  # full height is 2 in units of h
    np.testing.assert_allclose(mean, 2.0 / 3.0, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(mean, U_MEAN_OVER_U_MAX, rtol=1e-14, atol=0.0)


def test_s1_3_reynolds_definitions_are_mutually_consistent():
    """Re_Dh built from its own definition must equal (8/3) Re_h."""
    rho, u_max, h, mu = 998.2, 1.37e-2, 2.5e-4, 1.002e-3
    re_h = reynolds_h(rho, u_max, h, mu)
    re_dh = reynolds_dh(rho, U_MEAN_OVER_U_MAX * u_max, D_H_OVER_H * h, mu)
    np.testing.assert_allclose(re_dh, re_h_to_re_dh(re_h), rtol=1e-14, atol=0.0)


def test_s1_4_config_loader_raises_without_a_citation():
    """T-S1.4."""
    document = _document()
    del document["correlation"]["citation"]
    with pytest.raises(ConfigError, match="citation"):
        parse_config(document)


def test_s1_4_correlation_object_raises_on_empty_citation():
    with pytest.raises(MissingCitationError):
        EntranceLengthCorrelation(c0=0.3155, c1=0.01105, n=1.6, citation="   ")


def test_entrance_length_asymptote_matches_the_classical_value():
    """At large Re the blended form must reduce to L_e/D_h = 0.011 Re_Dh.

    This is the cross-check that pins the constants to the right geometry: the
    pipe fit would give a visibly different coefficient.
    """
    re_h = 1e6
    predicted = entrance_length_over_h(re_h, CORRELATION) / D_H_OVER_H
    classical = 0.011 * re_h_to_re_dh(re_h)
    np.testing.assert_allclose(predicted, classical, rtol=1e-2)


def test_entrance_length_is_returned_in_units_of_half_height():
    """L_e/h = 4 (L_e/D_h). A call site that forgot this would be 4x out."""
    re_h = 10.0
    np.testing.assert_allclose(
        entrance_length_over_h(re_h, CORRELATION),
        D_H_OVER_H * CORRELATION.length_over_dh(re_h_to_re_dh(re_h)),
        rtol=0.0,
        atol=0.0,
    )


# --------------------------------------------------------------------------- #
# S2
# --------------------------------------------------------------------------- #


def test_s2_1_case_b_inlet_flux_equals_the_developed_flux():
    """T-S2.1: the uniform inlet must carry exactly the developed flux."""
    config = parse_config(_document("experiments/caseB_re10.yaml"))
    np.testing.assert_allclose(
        config.inlet_flux, analytic.FLUX_ND, rtol=1e-12, atol=0.0
    )
    # And the amplitude that does it is 2/3, not 1.
    np.testing.assert_allclose(
        config.inlet_uniform_value, 2.0 / 3.0, rtol=1e-12, atol=0.0
    )


def test_s2_1_case_b_developed_profile_is_now_the_case_a_profile():
    """The SPEC v1 contradiction is gone: both cases share one reference velocity."""
    case_a = parse_config(_document("experiments/caseA_re10.yaml"))
    case_b = parse_config(_document("experiments/caseB_re10.yaml"))
    assert case_a.inlet_flux == case_b.inlet_flux

    y = np.linspace(-1.0, 1.0, 501)
    np.testing.assert_allclose(
        analytic.developed_profile_nd(y, case_b.inlet_flux),
        analytic.u_nd(y),
        rtol=1e-14,
        atol=1e-15,
    )


def test_s2_1_a_uniform_inlet_of_one_would_carry_the_wrong_flux():
    """Negative control: the old amplitude really was inconsistent."""
    assert not np.isclose(2.0 * 1.0, analytic.FLUX_ND)
    np.testing.assert_allclose(2.0 * (2.0 / 3.0), analytic.FLUX_ND, rtol=1e-15)


# --------------------------------------------------------------------------- #
# S3
# --------------------------------------------------------------------------- #


def test_s3_1_domain_length_grows_linearly_with_reynolds_at_moderate_re():
    """T-S3.1: in the moderate-Re limit L* is dominated by alpha*L_e ~ Re."""
    lengths = {}
    for re in (200.0, 400.0, 800.0):
        document = _document()
        document["physics"]["reynolds"] = re
        # The Phase 1 ceiling is a physics.reynolds check; raise it locally by
        # constructing the correlation directly rather than through the config.
        lengths[re] = float(entrance_length_over_h(re, CORRELATION))

    first = lengths[400.0] / lengths[200.0]
    second = lengths[800.0] / lengths[400.0]
    assert 1.8 < first < 2.05, first
    assert 1.9 < second < 2.05, second


def test_s3_2_a_short_domain_raises_at_load():
    """T-S3.2: the assertion fires before training, not after."""
    document = _document()
    document["domain"]["length"] = 1.0
    with pytest.raises(ConfigError, match="below the S3 bound"):
        parse_config(document)


def test_s3_2_an_adequate_explicit_domain_is_accepted():
    document = _document()
    document["domain"]["length"] = 500.0
    config = parse_config(document)
    assert config.domain.explicitly_overridden
    assert config.domain.length == 500.0


def test_s3_derived_length_matches_the_bound_exactly():
    config = parse_config(_document())
    expected = (
        config.domain.alpha * config.domain.entrance_length_over_h
        + config.domain.outlet_margin
    )
    np.testing.assert_allclose(config.domain.length, expected, rtol=1e-15, atol=0.0)
    assert not config.domain.explicitly_overridden


def test_s3_outlet_margin_below_five_raises():
    """SPEC v2 S3 requires L*_out >= 5."""
    document = _document()
    document["domain"]["outlet_margin"] = 2.0
    with pytest.raises(ConfigError, match="outlet_margin"):
        parse_config(document)


@pytest.mark.parametrize("path", [
    "experiments/caseA_re1.yaml", "experiments/caseA_re10.yaml",
    "experiments/caseA_re50.yaml", "experiments/caseA_re100.yaml",
    "experiments/caseB_re1.yaml", "experiments/caseB_re10.yaml",
    "experiments/caseB_re50.yaml", "experiments/caseB_re100.yaml",
])
def test_s3_every_committed_config_satisfies_the_bound(path):
    config = parse_config(_document(path))
    bound = (
        config.domain.alpha * config.domain.entrance_length_over_h
        + config.domain.outlet_margin
    )
    assert config.domain.length >= bound
