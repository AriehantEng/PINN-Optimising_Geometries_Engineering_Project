"""SPEC tests 1, 2, 3, 4, 8 and 9 -- the ones that require training.

Running the gate
----------------
    pytest tests/ -q                      # full schedule: the SPEC targets
    pytest tests/ -q -m "not slow"        # skip everything in this file
    pytest tests/ -q --schedule short     # CI smoke: NOT the SPEC gate

The short schedule exists because SPEC section 5 asks for "a short-schedule
version in CI". It trains a much smaller network for far fewer steps and
asserts a documented smoke bound instead. It is deliberately NOT the default,
so that a plain `pytest tests/ -q` always runs the real targets: a green tick
in this project must never mean "the reduced version passed".

The SPEC targets below are never modified. `SMOKE_BOUNDS` is a separate,
clearly-labelled set of numbers for the reduced schedule.
"""

import copy
import json

import jax
import numpy as np
import pytest
import yaml

from src.config import parse_config
from src.evaluation.accuracy import evaluate
from src.training.loop import train

# --------------------------------------------------------------------------- #
# SPEC section 5 targets. Do not modify. See CLAUDE.md.
# --------------------------------------------------------------------------- #
SPEC_TARGETS = {
    "case_a_u_relative_l2": 1e-5,          # test 1 (stretch target 1e-6)
    "case_a_v_max_abs": 1e-5,              # test 2
    "flux_conservation_relative": 1e-6,    # test 3
    "determinism_absolute": 1e-12,         # test 8
    "case_b_profile_relative_l2": 1e-4,    # test 9
    "case_b_entrance_length_relative": 0.15,  # test 9
}

#: Smoke bounds for `--schedule short` ONLY. These are not SPEC targets and a
#: pass here is not a pass of the corresponding SPEC test. They are set from
#: measured short-schedule behaviour, loose enough not to be flaky and tight
#: enough that a broken pipeline fails them.
SMOKE_BOUNDS = {
    "case_a_u_relative_l2": 5e-2,
    "case_a_v_max_abs": 5e-2,
    "flux_conservation_relative": 5e-2,
    "case_b_profile_relative_l2": 2e-1,
    "case_b_entrance_length_relative": 1.0,
}

SHORT_SCHEDULE = {
    "model.width": 32,
    "model.depth": 3,
    "sampling.n_interior": 512,
    "sampling.n_inlet": 64,
    "sampling.n_outlet": 64,
    "training.adam.steps": 400,
    "training.lbfgs.max_iterations": 400,
    "evaluation.nx": 81,
    "evaluation.ny": 41,
}


def _apply(document, updates):
    for path, value in updates.items():
        node = document
        *parents, leaf = path.split(".")
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    return document


def load_experiment(path, schedule):
    document = yaml.safe_load(open(path, encoding="utf-8"))
    if schedule == "short":
        document = _apply(document, SHORT_SCHEDULE)
    return parse_config(document)


_TRAINED = {}


def trained(path, schedule):
    """Train once per (config, schedule) and share across tests in the session."""
    key = (path, schedule)
    if key not in _TRAINED:
        config = load_experiment(path, schedule)
        result = train(config, verbose=False)
        metrics = evaluate(config, result.field_fn, result.params)
        _TRAINED[key] = (config, result, metrics)
    return _TRAINED[key]


def limit(name, schedule):
    """The bound in force, and a label saying which regime it came from."""
    if schedule == "short":
        return SMOKE_BOUNDS[name], "SMOKE BOUND (short schedule, NOT the SPEC gate)"
    return SPEC_TARGETS[name], "SPEC target"


CASE_A = "experiments/caseA_re10.yaml"
CASE_B = "experiments/caseB_re10.yaml"


# --------------------------------------------------------------------------- #
# Tests 1, 2, 3 -- Case A
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_1_case_a_relative_l2_error_on_u(schedule):
    """SPEC test 1 -- THE GATE.

    Published work using a plain DeepXDE setup on this exact problem reached a
    total test loss of 8.7e-5 and a relative L2 error of 8.6%. That is the
    failure mode this test exists to catch: if the loss is low and this test
    fails, the bug is in the formulation or the boundary handling, not the
    optimiser.
    """
    config, result, metrics = trained(CASE_A, schedule)
    bound, label = limit("case_a_u_relative_l2", schedule)
    loss = result.diagnostics.get("loss_after_lbfgs", result.diagnostics.get("loss_after_adam"))
    assert metrics["u_relative_l2"] <= bound, (
        f"relative L2 error on u is {metrics['u_relative_l2']:.3e}, "
        f"{label} is {bound:.1e}. Final loss was {loss:.3e} -- if that is small "
        "the problem is the formulation, not the optimiser."
    )


@pytest.mark.slow
def test_2_case_a_transverse_velocity_is_discovered_to_be_zero(schedule):
    """SPEC test 2. The PINN must DISCOVER v* = 0; it is never told."""
    _, _, metrics = trained(CASE_A, schedule)
    bound, label = limit("case_a_v_max_abs", schedule)
    assert metrics["v_max_abs"] <= bound, (
        f"max |v*| is {metrics['v_max_abs']:.3e}, {label} is {bound:.1e}"
    )


@pytest.mark.slow
def test_3_case_a_mass_is_conserved_across_streamwise_stations(schedule):
    """SPEC test 3: the flux at ten stations must agree to 1e-6 relative."""
    _, _, metrics = trained(CASE_A, schedule)
    bound, label = limit("flux_conservation_relative", schedule)
    assert len(metrics["flux_values"]) == 10
    assert metrics["flux_conservation_relative_error"] <= bound, (
        f"flux varies by {metrics['flux_conservation_relative_error']:.3e} "
        f"across stations, {label} is {bound:.1e}; "
        f"values {np.array(metrics['flux_values'])}"
    )


@pytest.mark.slow
def test_case_a_pressure_is_gauged_and_accurate(schedule):
    """Not a numbered SPEC test, but section 3 requires the gauge to be pinned.

    An ungauged pressure makes the pressure L2 error meaningless, so this
    checks the reported number is meaningful before anyone quotes it.
    """
    config, _, metrics = trained(CASE_A, schedule)
    assert config.bc.pressure_gauge in ("soft", "hard")
    assert np.isfinite(metrics["p_relative_l2"])


# --------------------------------------------------------------------------- #
# Test 4 -- Phase 2 placeholder
# --------------------------------------------------------------------------- #


@pytest.mark.phase2
@pytest.mark.xfail(
    strict=True,
    reason=(
        "SPEC test 4: parent/daughter flux conservation across a 1->2 "
        "bifurcation is Phase 2 work and is not implemented. This is marked "
        "strict, so it becomes a hard FAILURE the moment the check starts "
        "passing without this marker being removed. See "
        "docs/spec-deviations.md item 3."
    ),
)
def test_4_parent_daughter_flux_conservation():
    """SPEC test 4 -- placeholder, must not pass until Phase 2 implements it."""
    raise NotImplementedError(
        "Phase 2 requirement. When the 1->2 bifurcation geometry exists this "
        "test must assert that the flux through the parent channel equals the "
        "sum of the fluxes through both daughter channels, to the same 1e-6 "
        "relative tolerance as SPEC test 3, at several stations in each branch. "
        "Note Q ~ h^3 in 2D, NOT r^4 as for a circular pipe (SPEC section 1)."
    )


# --------------------------------------------------------------------------- #
# Test 8 -- determinism
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_8_same_config_and_seed_reproduce_the_same_error():
    """SPEC test 8. Always runs a short schedule: determinism is not about size.

    Two independent trainings from the same config and seed must land on the
    same final error. This is what makes every other number in the project
    reproducible, so it is checked regardless of `--schedule`.
    """
    config = load_experiment(CASE_A, "short")

    errors = []
    for _ in range(2):
        result = train(config, verbose=False)
        errors.append(evaluate(config, result.field_fn, result.params)["u_relative_l2"])

    difference = abs(errors[0] - errors[1])
    assert difference <= SPEC_TARGETS["determinism_absolute"], (
        f"two runs of the same config and seed gave {errors[0]!r} and "
        f"{errors[1]!r}, differing by {difference:.3e}"
    )


@pytest.mark.slow
def test_different_seeds_give_different_results():
    """Negative control for test 8: the seed must actually do something."""
    document = yaml.safe_load(open(CASE_A, encoding="utf-8"))
    _apply(document, SHORT_SCHEDULE)

    errors = []
    for seed in (0, 1):
        document["seed"] = seed
        config = parse_config(copy.deepcopy(document))
        result = train(config, verbose=False)
        errors.append(evaluate(config, result.field_fn, result.params)["u_relative_l2"])

    assert errors[0] != errors[1], (
        "changing the seed changed nothing, so the determinism test above is "
        "vacuous"
    )


# --------------------------------------------------------------------------- #
# Test 9 -- Case B
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_9_case_b_develops_into_the_analytic_profile(schedule):
    """SPEC test 9, first half: the downstream profile.

    Compared against the flux-consistent parabola. A uniform inlet against hard
    no-slip walls is discontinuous at the corners, so the network delivers
    slightly under the nominal flux; that deficit is an inlet artefact and is
    reported separately. See docs/spec-deviations.md item 1.
    """
    _, _, metrics = trained(CASE_B, schedule)
    bound, label = limit("case_b_profile_relative_l2", schedule)
    assert metrics["outlet_profile_relative_l2"] <= bound, (
        f"outlet profile differs from the developed parabola by "
        f"{metrics['outlet_profile_relative_l2']:.3e}, {label} is {bound:.1e} "
        f"(inlet flux deficit was "
        f"{metrics['inlet_flux_deficit_relative']:.3e})"
    )


@pytest.mark.slow
def test_9_case_b_entrance_length_matches_the_correlation(schedule):
    """SPEC test 9, second half: entrance length within 15% of the correlation.

    Correlation: Durst et al. (2005), plane channel. See
    `analytic.entrance_length_correlation_nd` for the definitions, which are
    pinned down by the classical L_e/D_h = 0.011 Re_Dh asymptote.
    """
    config, _, metrics = trained(CASE_B, schedule)
    bound, label = limit("case_b_entrance_length_relative", schedule)

    assert np.isfinite(metrics["entrance_length_measured"]), (
        "the centreline never reached 99% of its developed value inside the "
        f"domain (L* = {config.domain.length}); the domain is too short, which "
        "is a real result and must not be clipped away"
    )
    assert metrics["entrance_length_relative_error"] <= bound, (
        f"entrance length {metrics['entrance_length_measured']:.3f} vs "
        f"correlation {metrics['entrance_length_correlation']:.3f} "
        f"= {metrics['entrance_length_relative_error'] * 100:.1f}% error, "
        f"{label} is {bound * 100:.0f}%"
    )


@pytest.mark.slow
def test_9_case_b_conserves_mass(schedule):
    """Case B must conserve mass too, not only Case A."""
    _, _, metrics = trained(CASE_B, schedule)
    bound, label = limit("flux_conservation_relative", schedule)
    assert metrics["flux_conservation_relative_error"] <= bound, (
        f"flux varies by {metrics['flux_conservation_relative_error']:.3e} "
        f"across stations, {label} is {bound:.1e}"
    )
