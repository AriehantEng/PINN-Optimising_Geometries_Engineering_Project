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
        metrics = evaluate(config, result.model, result.params)
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
    """SPEC test 2. The PINN must DISCOVER v* = 0; it is never told.

    NOTE (SPEC v2 S5): this assertion is passed by the zero field in every
    formulation -- see docs/falsifiability-matrix.md, "Degenerate targets". It
    cannot distinguish a correct solution from a dead one on its own, so it is
    only meaningful conjoined with test 1, which is asserted here too rather
    than left to a separate test that might be skipped.
    """
    _, _, metrics = trained(CASE_A, schedule)
    u_bound, _ = limit("case_a_u_relative_l2", schedule)
    assert metrics["u_relative_l2"] <= u_bound, (
        "test 2 is vacuous unless test 1 also holds: the zero field satisfies "
        f"max|v*| but has u relative L2 = {metrics['u_relative_l2']:.3e}"
    )
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
        errors.append(evaluate(config, result.model, result.params)["u_relative_l2"])

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
        errors.append(evaluate(config, result.model, result.params)["u_relative_l2"])

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


@pytest.mark.parametrize("run_name", [
    "caseA_re10", "caseA_re10_psiomega_sinusoidal", "caseB_re10",
])
def test_run_summary_prints_for_every_formulation(run_name, capsys):
    """Regression: the summary printer crashed on every psi-omega run.

    It printed `p_relative_l2` unconditionally, which does not exist for a
    formulation with no pressure in its system, so `python -m src.run` exited
    non-zero AFTER writing all its results correctly. Skipped when the run
    directory is absent, since runs/ is git-ignored.
    """
    import json
    from pathlib import Path

    from src.run import _print_summary

    path = Path("runs") / run_name / "metrics.json"
    if not path.is_file():
        pytest.skip(f"{path} not present (runs/ is git-ignored)")
    _print_summary(json.loads(path.read_text()))
    assert capsys.readouterr().out.strip()


def test_a_missed_spec_target_is_a_non_zero_exit():
    """A printed FAIL must not exit zero.

    The `--full` demo printed `mass conservation drift D 1.4263e-06 target <=
    1e-06 FAIL` and returned 0. A green exit code on a printed failure means no
    CI job, queue script or reviewer can catch a regression against the SPEC
    section 5 targets, which are the entire point of the benchmark.
    """
    from src.report import format_gate, gate_results

    failures, vacuous = gate_results({
        "case": "A",
        "u_relative_l2": 1.0628e-06,
        "v_max_abs": 1.4624e-06,
        "flux_conservation_relative_error": 1.4263e-06,
    })
    assert not vacuous
    assert [label for label, _, _ in failures] == ["test 3  flux conservation"]
    assert "GATE FAILED" in format_gate(failures, vacuous)


def test_a_run_meeting_every_target_has_no_gate_failure():
    """Negative control: the extended run's numbers must pass cleanly."""
    from src.report import format_gate, gate_results

    failures, vacuous = gate_results({
        "case": "A",
        "u_relative_l2": 1.8369e-07,
        "v_max_abs": 3.8299e-07,
        "flux_conservation_relative_error": 1.8849e-07,
    })
    assert failures == [] and vacuous == []
    assert format_gate(failures, vacuous) == ""


def test_mass_conservation_at_the_rounding_floor_is_vacuous_not_passed():
    """psi-omega conserves mass by construction, so test 3 proves nothing.

    It is not a failure either -- the run must still exit zero -- but it must
    never be counted as evidence that the formulation conserves mass.
    """
    from src.report import format_gate, gate_results

    failures, vacuous = gate_results({
        "case": "A",
        "u_relative_l2": 1e-06,
        "flux_conservation_relative_error": 9.99e-16,
    })
    assert failures == []
    assert [label for label, _, _ in vacuous] == ["test 3  flux conservation"]
    assert "VACUOUS" in format_gate(failures, vacuous)


def test_case_b_is_not_gated_on_case_a_only_targets():
    """Developing flow has a genuine non-zero v*; gating it would be wrong."""
    from src.report import gate_results

    failures, _ = gate_results({
        "case": "B",
        "v_max_abs": 0.1889,
        "flux_conservation_relative_error": 1e-07,
    })
    assert failures == []


# --------------------------------------------------------------------------- #
# Near-inlet residual band
# --------------------------------------------------------------------------- #


def test_the_inlet_band_metric_is_bounded_by_the_global_one():
    """A max over a subset can never exceed the max over the whole set.

    Trivial, and worth pinning: the band metric exists to decide whether an
    inlet condition is representable, and a band that silently covered the
    whole domain -- or none of it -- would answer that question wrongly while
    looking fine.
    """
    import jax

    from src.config import load_config
    from src.evaluation.accuracy import INLET_BAND_FRACTION, _residual_norms
    from src.geometry.domain import Channel
    from src.geometry.sampling import evaluation_grid
    from src.models.build import build_model

    config = load_config("experiments/caseB_re10_smooth8.yaml")
    params, model = build_model(config, jax.random.PRNGKey(config.seed))
    points, _, _ = evaluation_grid(Channel(config.domain.length), 81, 41)
    out = _residual_norms(config, model, params, points)

    banded = [k for k in out if k.endswith("_max_inlet_band")]
    assert banded, "no inlet-band metric was recorded"
    for key in banded:
        assert out[key] <= out[key.replace("_max_inlet_band", "_max")] * (1 + 1e-12)
    for key in (k for k in out if k.endswith("_rms_inlet_band")):
        assert out[key] > 0.0

    assert out["residual_inlet_band_fraction"] == INLET_BAND_FRACTION
    assert 0 < out["residual_inlet_band_points"] < len(points)


def test_the_inlet_band_actually_selects_the_inlet():
    """The band must contain x* = 0 and exclude the outlet.

    The point of the metric is that a corner singularity is invisible in a
    domain-wide norm. If the mask were wrong the number would still look
    plausible, which is exactly the kind of check that has to be explicit.
    """
    import numpy as np

    from src.config import load_config
    from src.evaluation.accuracy import INLET_BAND_FRACTION
    from src.geometry.domain import Channel
    from src.geometry.sampling import evaluation_grid

    config = load_config("experiments/caseB_re10_smooth8.yaml")
    points, _, _ = evaluation_grid(Channel(config.domain.length), 81, 41)
    x = np.asarray(points)[:, 0]
    band = x < INLET_BAND_FRACTION * config.domain.length

    assert band[x == x.min()].all(), "the band excludes the inlet plane"
    assert not band[x == x.max()].any(), "the band reaches the outlet"
    assert x[band].max() < 0.5, "the band is far wider than intended"
