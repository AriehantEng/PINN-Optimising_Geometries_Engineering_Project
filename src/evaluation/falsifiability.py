"""Falsifiability matrix (SPEC v2, S5).

A test `T` with assertion `A(theta)` over network parameters is VACUOUS under
ansatz `F` if `A(F(theta))` holds for all `theta`. Then `T` reports a property
of `F`, not of the trained solution.

Vacuity is decidable by sampling: if a randomly initialised, UNTRAINED network
passes a test, that test cannot distinguish a trained solution from noise.

    LIVE      the untrained model fails at every seed -- the test bites
    WEAKENED  it passes at some seeds and not others
    VACUOUS   it passes at every seed -- the test measures the ansatz

This exists because Phase 1 hit the failure mode twice and nothing reported it:
hard no-slip silently emptied the wall tests, and the stream function silently
emptied the mass-conservation test. The defect is not the disarming, it is the
absence of detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.domain import WALL, Channel
from ..geometry.sampling import evaluation_grid
from ..models.build import build_model
from ..physics import analytic, inlets
from .metrics import cross_channel_flux, evaluate_fields, flux_conservation_error, max_abs, relative_l2

LIVE = "LIVE"
WEAKENED = "WEAKENED"
VACUOUS = "VACUOUS"

#: A check can be LIVE at random initialisation and still be nearly free,
#: because the ansatz makes the correct answer a single trivially reachable
#: point in parameter space. Sampling at the initialisation distribution cannot
#: see that. `degenerate_targets` probes for it directly by zeroing the output
#: layer, i.e. setting the network's contribution identically to zero.
DEGENERATE = "DEGENERATE-TARGET"


@dataclass(frozen=True)
class Check:
    """One assertion, evaluated on a model without training it.

    `passes(config, model, params) -> bool` must mirror the real test's
    assertion, including its tolerance.
    """

    name: str
    passes: Callable
    spec_test: str = ""


def _wall_points(length, n=64):
    y = np.concatenate([np.full(n, WALL), np.full(n, -WALL)])
    x = np.concatenate([np.linspace(0.0, length, n)] * 2)
    return jnp.asarray(np.stack([x, y], axis=-1))


def _grid(config):
    return evaluation_grid(Channel(config.domain.length), 121, 61)


def check_wall_no_slip(config, model, params):
    values = jax.vmap(lambda xy: model.field(params, xy))(
        _wall_points(config.domain.length)
    )
    return max_abs(values[:, 0]) < 1e-12 and max_abs(values[:, 1]) < 1e-12


def check_test_1_u_relative_l2(config, model, params):
    points, _, _ = _grid(config)
    u, _, _ = evaluate_fields(model.field, params, points)
    exact = analytic.u_nd(np.asarray(points[:, 1]))
    return relative_l2(u, exact) <= 1e-5


def check_test_2_v_max_abs(config, model, params):
    points, _, _ = _grid(config)
    _, v, _ = evaluate_fields(model.field, params, points)
    return max_abs(v) <= 1e-5


def check_test_3_mass_conservation(config, model, params):
    length = config.domain.length
    stations = np.linspace(0.05 * length, 0.95 * length, 10)
    fluxes = cross_channel_flux(model.field, params, stations)
    try:
        return flux_conservation_error(fluxes) <= 1e-6
    except ValueError:
        return False


def check_inlet_profile(config, model, params):
    y = jnp.linspace(-1.0, 1.0, 129)
    points = jnp.stack([jnp.zeros_like(y), y], axis=-1)
    u, v, _ = evaluate_fields(model.field, params, points)
    if config.case == "A":
        target = analytic.u_nd(np.asarray(y))
    elif config.inlet_profile == "smoothed":
        target = np.asarray(inlets.profile(y, config.inlet_sharpness))
    else:
        target = np.full(len(y), config.inlet_uniform_value)
    return max_abs(u - target) < 1e-6 and max_abs(v) < 1e-6


def check_pressure_gauge(config, model, params):
    if not model.has_pressure:
        return False  # no pressure in the system: the assertion does not apply
    y = jnp.linspace(-1.0, 1.0, 65)
    points = jnp.stack([jnp.full_like(y, config.domain.length), y], axis=-1)
    _, _, p = evaluate_fields(model.field, params, points)
    return max_abs(p) == 0.0


def check_always_true(config, model, params):
    """T-S5.2: a deliberately vacuous check, to prove the detector detects."""
    return True


def check_always_false(config, model, params):
    """Control in the other direction: must always report LIVE."""
    return False


DEFAULT_CHECKS = (
    Check("wall no-slip", check_wall_no_slip, "structural"),
    Check("test 1  u relative L2", check_test_1_u_relative_l2, "1"),
    Check("test 2  max |v*|", check_test_2_v_max_abs, "2"),
    Check("test 3  mass conservation", check_test_3_mass_conservation, "3"),
    Check("inlet profile", check_inlet_profile, "structural"),
    Check("pressure gauge at outlet", check_pressure_gauge, "structural"),
    Check("CONTROL always-true", check_always_true, "T-S5.2"),
    Check("CONTROL always-false", check_always_false, "T-S5.2"),
)


def classify(config, checks=DEFAULT_CHECKS, seeds=(0, 1, 2, 3, 4)):
    """Classify every check for one configuration, at random initialisation."""
    verdicts = {}
    for check in checks:
        outcomes = []
        for seed in seeds:
            params, model = build_model(config, jax.random.PRNGKey(seed))
            outcomes.append(bool(check.passes(config, model, params)))
        if all(outcomes):
            verdicts[check.name] = VACUOUS
        elif any(outcomes):
            verdicts[check.name] = WEAKENED
        else:
            verdicts[check.name] = LIVE
    return verdicts


def build_matrix(configs, checks=DEFAULT_CHECKS, seeds=(0, 1, 2, 3, 4)):
    """`{check name: {column label: verdict}}` across several configurations."""
    matrix = {check.name: {} for check in checks}
    for label, config in configs.items():
        for name, verdict in classify(config, checks, seeds).items():
            matrix[name][label] = verdict
    return matrix


def format_matrix(matrix, columns) -> str:
    width = max(len(name) for name in matrix) + 2
    header = "| " + "check".ljust(width) + "| " + " | ".join(
        c.ljust(10) for c in columns
    ) + " |"
    rule = "|" + "-" * (width + 1) + "|" + "|".join("-" * 12 for _ in columns) + "|"
    lines = [header, rule]
    for name, row in matrix.items():
        cells = " | ".join(row.get(c, "-").ljust(10) for c in columns)
        lines.append("| " + name.ljust(width) + "| " + cells + " |")
    return "\n".join(lines)


def all_vacuous_rows(matrix):
    """Checks that are vacuous in EVERY column.

    Such a row means the property is untested by this suite and must be flagged
    in the report, never silently accepted.
    """
    return [
        name
        for name, row in matrix.items()
        if row and all(v == VACUOUS for v in row.values())
    ]


def zero_network_contribution(params):
    """Set the output layer to zero, so the network contributes nothing.

    For a psi-omega model under the cubic lift this makes psi exactly the lift,
    whose derivative IS the Case A solution -- so any check comparing against
    that solution passes perfectly. Under the sinusoidal lift it does not.
    """
    import copy

    params = copy.deepcopy(params)
    output = params["output"] if "output" in params else params["layers"][-1]
    weight_key = "W" if "W" in output else "V"
    output[weight_key] = jnp.zeros_like(output[weight_key])
    output["b"] = jnp.zeros_like(output["b"])
    return params


def degenerate_targets(config, checks=DEFAULT_CHECKS, seed=0):
    """Checks whose assertion passes when the network contributes nothing.

    A check listed here is not vacuous -- an untrained network fails it -- but
    the ansatz has placed the correct answer at a single point the optimiser
    can reach exactly, so passing it demonstrates the lift rather than the
    network's expressivity. SPEC v2 S4 flags exactly this for the cubic lift;
    this function decides it by measurement instead of by argument.
    """
    params, model = build_model(config, jax.random.PRNGKey(seed))
    zeroed = zero_network_contribution(params)
    degenerate = []
    for check in checks:
        if check.name.startswith("CONTROL"):
            continue
        try:
            if check.passes(config, model, zeroed):
                degenerate.append(check.name)
        except Exception:  # a check that cannot even run is not degenerate
            continue
    return degenerate
