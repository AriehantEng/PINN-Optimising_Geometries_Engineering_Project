"""Case-specific accuracy evaluation on the fixed evaluation grid.

Never on the training points (SPEC section 9). The grid is rebuilt from the
config, so the same config always evaluates on the same points regardless of
what the sampler did.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..config import Config
from ..geometry.domain import Channel
from ..geometry.sampling import evaluation_grid
from ..physics import analytic
from ..physics.conventions import entrance_length_over_h
from ..physics.derivatives import batched_field_derivatives
from ..physics import residuals
from .metrics import (
    centreline_velocity,
    cross_channel_flux,
    entrance_length,
    evaluate_fields,
    flux_conservation_error,
    max_abs,
    relative_l2,
)

#: SPEC test 3 asks for ten streamwise stations.
N_FLUX_STATIONS = 10

#: Stations are trimmed 5% away from the inlet and outlet planes so that a
#: boundary-condition error is not counted as a mass-conservation error.
FLUX_STATION_MARGIN = 0.05


def _residual_norms(config, field_fn, params, points):
    """Max and RMS of each unweighted residual on the evaluation grid.

    Reported as a DIAGNOSTIC. A small residual is not an accuracy claim: the
    failure mode this project exists to catch is exactly a low residual with a
    wrong solution.
    """
    d = batched_field_derivatives(lambda xy: field_fn(params, xy), points)
    out = {}
    names = ("momentum_x", "momentum_y", "continuity")
    values = residuals.navier_stokes(
        d, config.physics.reynolds, config.physics.pressure_scaling
    )
    for name, value in zip(names, values):
        value = np.asarray(value)
        out[f"residual_{name}_max"] = float(np.max(np.abs(value)))
        out[f"residual_{name}_rms"] = float(np.sqrt(np.mean(value**2)))
    return out


def _flux_metrics(config, field_fn, params):
    length = config.domain.length
    stations = np.linspace(
        FLUX_STATION_MARGIN * length,
        (1.0 - FLUX_STATION_MARGIN) * length,
        N_FLUX_STATIONS,
    )
    fluxes = cross_channel_flux(field_fn, params, stations)
    return {
        "flux_stations": stations.tolist(),
        "flux_values": np.asarray(fluxes).tolist(),
        "flux_mean": float(np.mean(fluxes)),
        "flux_conservation_relative_error": flux_conservation_error(fluxes),
        "flux_relative_error_vs_inlet": float(
            abs(np.mean(fluxes) - config.inlet_flux) / config.inlet_flux
        ),
    }


def evaluate(config: Config, field_fn, params) -> dict:
    """All reported metrics for a finished run."""
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)

    u, v, p = evaluate_fields(field_fn, params, points)

    metrics = {
        "case": config.case,
        "reynolds": config.physics.reynolds,
        "pressure_scaling": config.physics.pressure_scaling,
        "formulation": config.physics.formulation,
        "evaluation_points": int(points.shape[0]),
        "v_max_abs": max_abs(v),
        "u_max": float(np.max(u)),
    }
    metrics.update(_flux_metrics(config, field_fn, params))
    metrics.update(_residual_norms(config, field_fn, params, points))

    grid_y = np.asarray(points[:, 1])
    grid_x = np.asarray(points[:, 0])

    if config.case == "A":
        # The exact solution is known everywhere.
        u_exact = analytic.u_nd(grid_y)
        p_exact = analytic.p_nd(
            grid_x,
            config.domain.length,
            config.physics.reynolds,
            config.physics.pressure_scaling,
        )
        metrics["u_relative_l2"] = relative_l2(u, u_exact)
        metrics["p_relative_l2"] = relative_l2(p, p_exact)
        metrics["u_max_abs_error"] = max_abs(u - u_exact)
        metrics["p_max_abs_error"] = max_abs(p - p_exact)
    else:
        # No closed-form solution. Two checks apply (SPEC section 2, Case B):
        # the downstream profile, and the entrance length.
        #
        # A uniform inlet u* = 1 is DISCONTINUOUS at the corners against hard
        # no-slip walls: the ansatz u = phi(y) N_u would need N_u = 1/(1-y^2)
        # at the inlet, which is unbounded. The network therefore delivers
        # slightly less than the nominal flux, and that deficit is an inlet
        # boundary-condition artefact, not a solver error.
        #
        # So the developed profile is compared two ways, and both are reported:
        #   * against the NOMINAL flux from the config -- includes the deficit;
        #   * against the MEASURED flux -- a pure shape test, which is the
        #     question Phase 2 actually cares about ("does the flow redevelop
        #     into the right profile?"). Mass conservation is checked separately
        #     and independently by the flux metrics above.
        measured_flux = float(metrics["flux_mean"])
        nominal = analytic.developed_profile_nd(np.asarray(y), config.inlet_flux)
        shaped = analytic.developed_profile_nd(np.asarray(y), measured_flux)

        outlet_points = jnp.stack(
            [jnp.full_like(y, config.domain.length), y], axis=-1
        )
        u_outlet, _, _ = evaluate_fields(field_fn, params, outlet_points)
        metrics["outlet_profile_relative_l2_vs_nominal_flux"] = relative_l2(
            u_outlet, nominal
        )
        metrics["outlet_profile_relative_l2"] = relative_l2(u_outlet, shaped)
        metrics["developed_centreline_nominal"] = float(0.75 * config.inlet_flux)
        metrics["developed_centreline_measured"] = float(0.75 * measured_flux)
        metrics["inlet_flux_deficit_relative"] = float(
            (config.inlet_flux - measured_flux) / config.inlet_flux
        )

        stations = np.linspace(0.0, config.domain.length, 2001)
        centreline = centreline_velocity(field_fn, params, stations)
        # The 99% threshold is against the flux the solution actually carries.
        measured = entrance_length(stations, centreline, 0.75 * measured_flux)
        # SPEC v2 S1: the correlation consumes Re_Dh, not Re_h, and returns a
        # length in D_h. `entrance_length_over_h` converts both internally, so
        # this call site cannot get the factor of 8/3 or the factor of 4 wrong.
        correlation = float(
            entrance_length_over_h(config.physics.reynolds, config.correlation)
        )
        metrics["entrance_length_measured"] = measured
        metrics["entrance_length_correlation"] = correlation
        metrics["entrance_length_relative_error"] = (
            float(abs(measured - correlation) / correlation)
            if np.isfinite(measured)
            else float("nan")
        )
        # T-S2.2: the coupling between profile error and mass drift stays
        # visible in every run. A ratio near 1 means the profile error IS the
        # mass drift wearing an amplitude disguise. Undefined once the drift is
        # at machine zero (stream function), where it is reported as such.
        drift = metrics["flux_conservation_relative_error"]
        metrics["profile_error_over_mass_error"] = (
            float(metrics["outlet_profile_relative_l2"] / drift)
            if drift > 1e-12
            else None
        )
        metrics["centreline_stations"] = stations[::100].tolist()
        metrics["centreline_values"] = np.asarray(centreline)[::100].tolist()

    return metrics
