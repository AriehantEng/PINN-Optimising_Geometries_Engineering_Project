"""Mass-drift metric and its convergence fit (SPEC v2, S7).

With `M(x*) = integral u* dy*` on a FIXED evaluation grid at `K >= 10` stations:

    D = max_x | M(x*) - M(0) | / M(0)

Test 3 asserts `D <= 1e-6` and the tolerance stays. Test 1 measures a pointwise
field error; test 3 measures the variation of an integral, and integration
normally averages errors down, so a tighter bound on the integral is the
correct default. `D` exceeding the pointwise error means the error is
SYSTEMATIC AND ACCUMULATING DOWNSTREAM rather than cancelling -- which is the
finding, and relaxing the bound would erase it.

The discriminating measurement is the trace of `D` against L-BFGS iteration,
fitted to

    D(n) = D_inf + a n^-b

* `D_inf` significantly above 1e-6 -> formulation bias. Report it with a
  confidence interval and close the test as a characterised limitation.
* `D_inf` consistent with zero -> under-convergence, and a longer run clears it.

Either outcome is a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mass_drift(fluxes) -> float:
    """D, referenced to the FIRST station as SPEC v2 S7 defines it.

    Note this differs from a spread-about-the-mean measure: referencing the
    inlet station makes a monotonic downstream drift show at its full size
    rather than half of it.
    """
    fluxes = np.asarray(fluxes, dtype=np.float64)
    if fluxes.size < 10:
        raise ValueError(
            f"SPEC v2 S7 requires at least 10 stations, got {fluxes.size}"
        )
    reference = float(fluxes[0])
    if reference == 0.0:
        raise ValueError("flux at the first station is zero; D is undefined")
    return float(np.max(np.abs(fluxes - reference)) / abs(reference))


@dataclass(frozen=True)
class DriftFit:
    """Result of fitting D(n) = D_inf + a n^-b."""

    d_infinity: float
    amplitude: float
    exponent: float
    d_infinity_stderr: float
    points: int
    converged: bool
    message: str
    at_machine_precision: bool = False
    tail_at_last_point: float = float("nan")
    last_iteration: float = float("nan")
    decay_range: float = float("nan")
    max_relative_residual: float = float("nan")
    moved_off_guess: bool = True

    #: Largest allowed |model - D| / D across the trace. A power law fitted to
    #: a noisy tail does not track it tightly -- the real caseA traces reach
    #: 0.85 and 1.15 with genuine fits -- so this is a gross-failure detector,
    #: not a quality bar. The degenerate fits it exists to catch sit at 5.8,
    #: 14.5 and 6e5, so a cut at 2.0 separates them with room on both sides.
    MAXIMUM_RELATIVE_RESIDUAL = 2.0

    #: The fitted transient must decay by at least this factor across the
    #: traced range for D_inf to be a measurement rather than an extrapolation.
    #: Recalibrated on synthetic traces carrying 10% multiplicative scatter,
    #: since identifiability is a property of noisy data (the earlier noiseless
    #: calibration was measuring the unweighted fitter's own error, and became
    #: meaningless once `fit_drift` weighted its residuals relatively). Median
    #: relative error in D_inf over 20 seeds, at D_inf = 1e-5: decay 4.5 is
    #: 801% out and 7.1 is 218% out, against 33% at decay 29 and 1.6% at 197.
    #: One decade is the conservative cut between them. Note decay range alone
    #: is not sufficient -- at decay 29 with D_inf = 1e-7 the error is still
    #: 1040% -- which is what `resolves` catches downstream.
    MINIMUM_DECAY_RANGE = 10.0

    @property
    def identifiable(self) -> bool:
        """Whether the trace actually pins D_inf down.

        D_inf is the asymptote of D(n) = D_inf + a n^-b. What determines
        whether it can be measured is not how big the tail is, but how much the
        tail DECAYED across the traced range: without that leverage the fit
        cannot separate the constant from the transient.

        The fitted standard error is not usable for this. On synthetic traces
        it underestimates the true error by roughly an order of magnitude
        (0.023 reported against 0.33 actual, 0.089 against 25), which is
        typical for a weakly constrained parameter in a nonlinear fit. The
        decay range separates the same cases cleanly.
        """
        if not self.converged or not np.isfinite(self.decay_range):
            return False
        return self.decay_range >= self.MINIMUM_DECAY_RANGE

    @property
    def tracks_data(self) -> bool:
        """Whether the fitted curve actually follows the trace.

        `curve_fit` reporting success is not evidence that a fit happened. On
        an unscaled least-squares problem whose residuals are O(1e-7), the cost
        at the initial guess is already below the default `ftol`, so the
        optimiser returns `p0` untouched and `converged=True`. That produced
        "D_inf = 1.885e-07 +- 9.1e-07, consistent with zero" for the 30000
        iteration run -- a verdict with the right conclusion drawn from a
        model that missed its own final data point by 9x.

        Relative weighting in `fit_drift` removes the cause; this checks the
        symptom directly, and catches any other way the fit can fail to
        describe the data.
        """
        if not self.converged or not np.isfinite(self.max_relative_residual):
            return False
        if not self.moved_off_guess:
            return False
        return self.max_relative_residual <= self.MAXIMUM_RELATIVE_RESIDUAL

    @property
    def consistent_with_zero(self) -> bool:
        """True when D_inf is within two standard errors of zero."""
        if not np.isfinite(self.d_infinity_stderr):
            return False
        return self.d_infinity <= 2.0 * self.d_infinity_stderr

    def resolves(self, tolerance=1e-6) -> bool:
        """Whether the fit can say anything AT the tolerance scale.

        Separate from `identifiable`, which asks whether the trace has enough
        decay leverage. This asks whether the resulting uncertainty is small
        enough to matter: a short trace can fit a clean power law (good decay
        range) and still leave D_inf uncertain by far more than the tolerance
        being tested, in which case "consistent with zero" is true but says
        nothing. Measured on a 400-iteration run: D_inf = 2.2e-21 +- 9.8e-3
        against a 1e-6 tolerance.
        """
        if not np.isfinite(self.d_infinity_stderr):
            return False
        return self.d_infinity_stderr <= tolerance

    def verdict(self, tolerance=1e-6) -> str:
        if self.at_machine_precision:
            return (
                "D is at machine precision throughout: this formulation "
                "conserves mass by construction, so there is no drift to "
                "characterise and no convergence claim to make. Test 3 is "
                "VACUOUS here (see docs/falsifiability-matrix.md), not passed"
            )
        if not self.converged:
            return "fit did not converge; too few points or too little variation"
        if not self.tracks_data:
            if not self.moved_off_guess:
                return (
                    "the fit is degenerate: curve_fit returned the initial "
                    "guess unchanged, so no fit was performed. No verdict can "
                    "be drawn from it"
                )
            return (
                f"the fitted curve does not describe the trace: it misses the "
                f"data by up to {self.max_relative_residual:.1f}x its own "
                f"value, against the {self.MAXIMUM_RELATIVE_RESIDUAL:.0f}x "
                "cut. D_inf is not a measurement of anything here"
            )
        if not self.identifiable:
            return (
                f"D_inf is NOT identifiable from this trace: the fitted "
                f"transient decayed by only {self.decay_range:.1f}x across it, "
                f"against the {self.MINIMUM_DECAY_RANGE:.0f}x needed to "
                "separate the constant from the transient. No bias or "
                "under-convergence verdict can be drawn -- extend the L-BFGS "
                "budget and refit"
            )
        if not self.resolves(tolerance):
            return (
                f"the trace cannot resolve D_inf at the {tolerance:.0e} scale: "
                f"the fitted uncertainty is +-{self.d_infinity_stderr:.1e}, "
                "larger than the tolerance being tested. Any verdict here would "
                "be an over-claim -- run longer before concluding anything"
            )
        if self.consistent_with_zero:
            return (
                "D_inf is consistent with zero: the residual drift is "
                "under-convergence, and a longer run should clear it"
            )
        if self.d_infinity > tolerance:
            return (
                f"D_inf = {self.d_infinity:.3e} +- {self.d_infinity_stderr:.1e} "
                f"exceeds {tolerance:.0e}: this is formulation bias, and test 3 "
                "should be closed as a characterised limitation rather than "
                "chased with more iterations"
            )
        return (
            f"D_inf = {self.d_infinity:.3e} +- {self.d_infinity_stderr:.1e} "
            f"lies below {tolerance:.0e}: more iterations should reach the target"
        )


def fit_drift(iterations, drifts) -> DriftFit:
    """Least-squares fit of D(n) = D_inf + a n^-b."""
    from scipy.optimize import curve_fit

    iterations = np.asarray(iterations, dtype=np.float64)
    drifts = np.asarray(drifts, dtype=np.float64)

    # A formulation that conserves mass by construction traces D at the
    # rounding floor. Fitting D(n) = D_inf + a n^-b to that is fitting noise,
    # and it produced the confident nonsense "D_inf = 1.0e-10, more iterations
    # should reach the target" for a run whose D was already exactly zero.
    finite = drifts[np.isfinite(drifts)]
    if finite.size and float(np.max(finite)) < 1e-12:
        return DriftFit(
            d_infinity=float(np.max(finite)), amplitude=0.0, exponent=float("nan"),
            d_infinity_stderr=0.0, points=int(finite.size), converged=False,
            message="D is at machine precision; nothing to fit",
            at_machine_precision=True,
        )

    keep = (iterations > 0) & np.isfinite(drifts) & (drifts > 0)
    iterations, drifts = iterations[keep], drifts[keep]

    if iterations.size < 5:
        return DriftFit(np.nan, np.nan, np.nan, np.nan, int(iterations.size), False,
                        "fewer than 5 usable trace points")

    def model(n, d_inf, a, b):
        return d_inf + a * n ** (-b)

    guess = (float(drifts[-1]), float(max(drifts[0] - drifts[-1], drifts[0])), 0.5)
    try:
        # Weight each residual by its own D. Unweighted, this problem spans
        # three and a half decades with a cost of O(1e-9) at the initial
        # guess -- already under curve_fit's default ftol, so the optimiser
        # returns p0 and reports success. Relative residuals are O(1), which
        # is also the right statistical choice: the trace is meaningful to
        # constant relative, not constant absolute, precision.
        popt, pcov = curve_fit(
            model, iterations, drifts, p0=guess, sigma=drifts, absolute_sigma=False,
            bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, 5.0]), maxfev=20000,
        )
    except Exception as error:  # noqa: BLE001 - a failed fit is a reportable state
        return DriftFit(np.nan, np.nan, np.nan, np.nan, int(iterations.size), False,
                        f"curve_fit failed: {error}")

    stderr = float(np.sqrt(np.diag(pcov))[0]) if np.all(np.isfinite(pcov)) else np.nan
    first, last = float(iterations[0]), float(iterations[-1])
    tail = float(popt[1] * last ** (-popt[2]))
    decay = float((last / first) ** popt[2]) if first > 0 else float("nan")
    residual = float(np.max(np.abs(model(iterations, *popt) - drifts) / drifts))
    return DriftFit(
        d_infinity=float(popt[0]), amplitude=float(popt[1]), exponent=float(popt[2]),
        d_infinity_stderr=stderr, points=int(iterations.size), converged=True,
        message="ok", tail_at_last_point=tail, last_iteration=last,
        decay_range=decay, max_relative_residual=residual,
        moved_off_guess=not bool(np.allclose(popt, guess)),
    )
