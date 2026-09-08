"""SPEC v2 S7 - mass drift D and its convergence fit."""

import numpy as np
import pytest

from src.evaluation.drift import DriftFit, fit_drift, mass_drift


def test_s7_d_is_referenced_to_the_first_station():
    """A monotone drift must report its full size, not half of it."""
    fluxes = np.linspace(2.0, 2.2, 10)
    np.testing.assert_allclose(mass_drift(fluxes), 0.1, rtol=1e-12)

    # A spread-about-the-mean measure would report half.
    spread = float(np.max(np.abs(fluxes - fluxes.mean())) / fluxes.mean())
    assert spread < mass_drift(fluxes)


def test_s7_d_is_zero_for_a_perfectly_conserved_flux():
    assert mass_drift(np.full(12, 1.3333)) == 0.0


def test_s7_requires_at_least_ten_stations():
    """SPEC v2 S7 says K >= 10; fewer must raise rather than under-sample."""
    with pytest.raises(ValueError, match="at least 10"):
        mass_drift(np.linspace(2.0, 2.1, 9))


def test_s7_zero_reference_flux_raises():
    with pytest.raises(ValueError, match="undefined"):
        mass_drift(np.zeros(10))


@pytest.mark.parametrize("d_inf,exponent", [(3e-6, 0.7), (1e-5, 0.4), (2e-7, 1.1)])
def test_s7_fit_recovers_the_exponent_from_a_clean_trace(d_inf, exponent):
    """The decay exponent is recoverable across the whole range tested."""
    n = np.arange(50, 6001, 50)
    fit = fit_drift(n, d_inf + 0.02 * n ** (-exponent))
    assert fit.converged
    np.testing.assert_allclose(fit.exponent, exponent, rtol=0.1)


@pytest.mark.parametrize("d_inf,exponent", [(3e-6, 0.7), (2e-7, 1.1)])
def test_s7_fit_recovers_d_infinity_when_it_is_identifiable(d_inf, exponent):
    n = np.arange(50, 6001, 50)
    fit = fit_drift(n, d_inf + 0.02 * n ** (-exponent))
    assert fit.identifiable
    np.testing.assert_allclose(fit.d_infinity, d_inf, rtol=0.25)


@pytest.mark.parametrize("d_inf,exponent,expected_error", [(1e-5, 0.4, 0.3), (1e-7, 0.3, 20.0)])
def test_s7_fit_refuses_to_report_an_unidentifiable_asymptote(
    d_inf, exponent, expected_error
):
    """A slow exponent leaves D_inf as an extrapolation, not a measurement.

    At b = 0.4 the fitted D_inf is 33% out and at b = 0.3 it is wrong by a
    factor of 25, while the exponent is recovered to three digits in both
    cases. Reporting a bias verdict there would be reporting the
    extrapolation, so the fit declines instead.
    """
    n = np.arange(50, 6001, 50)
    fit = fit_drift(n, d_inf + 0.02 * n ** (-exponent))
    assert fit.converged
    assert not fit.identifiable
    assert "NOT identifiable" in fit.verdict()
    assert "formulation bias" not in fit.verdict()
    # And the error really is as large as the guard implies.
    assert abs(fit.d_infinity - d_inf) / d_inf > expected_error * 0.8


def test_s7_fit_distinguishes_bias_from_under_convergence():
    """The whole point: the verdict must differ between the two cases."""
    n = np.arange(50, 6001, 50)
    biased = fit_drift(n, 3e-5 + 0.02 * n**-0.7)
    clean = fit_drift(n, 0.02 * n**-0.7)

    assert biased.d_infinity > 1e-6
    assert biased.identifiable
    assert "formulation bias" in biased.verdict()
    assert clean.d_infinity < biased.d_infinity
    assert "formulation bias" not in clean.verdict()


def test_s7_fit_reports_failure_rather_than_guessing():
    fit = fit_drift([1, 2, 3], [1e-3, 1e-4, 1e-5])
    assert not fit.converged
    assert isinstance(fit, DriftFit)
    assert "fewer than 5" in fit.message
