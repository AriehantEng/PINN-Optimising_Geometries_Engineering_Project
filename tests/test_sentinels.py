"""SPEC v2 S5 - sentinel tests and the falsifiability matrix.

If a randomly initialised, untrained network passes a test, that test cannot
distinguish a trained solution from noise. Every test in the suite is checked
against that standard here, on every formulation, and the result is emitted as
a matrix that must be regenerated on any ansatz change.
"""

from pathlib import Path

import pytest
import yaml

from src.config import parse_config
from src.evaluation.falsifiability import (
    DEFAULT_CHECKS,
    LIVE,
    VACUOUS,
    WEAKENED,
    all_vacuous_rows,
    build_matrix,
    format_matrix,
)

MATRIX_PATH = Path("docs/falsifiability-matrix.md")

COLUMNS = {
    "vp-soft": "experiments/caseA_re10_baseline.yaml",
    "vp-hard": "experiments/caseA_re10.yaml",
    "psi-cubic": "experiments/caseA_re10_psiomega_cubic.yaml",
    "psi-sin": "experiments/caseA_re10_psiomega_sinusoidal.yaml",
}


@pytest.fixture(scope="module")
def matrix():
    configs = {
        label: parse_config(yaml.safe_load(open(path, encoding="utf-8")))
        for label, path in COLUMNS.items()
    }
    return build_matrix(configs), list(configs)


def test_s5_1_matrix_covers_every_formulation(matrix):
    """T-S5.1: the sentinel suite runs on every formulation and emits a matrix."""
    table, columns = matrix
    assert set(columns) == set(COLUMNS)
    for check in DEFAULT_CHECKS:
        assert check.name in table
        assert set(table[check.name]) == set(COLUMNS)
        for verdict in table[check.name].values():
            assert verdict in (LIVE, WEAKENED, VACUOUS)


def test_s5_2_a_deliberately_vacuous_check_is_reported_vacuous(matrix):
    """T-S5.2: proving the detector itself is not vacuous."""
    table, _ = matrix
    assert set(table["CONTROL always-true"].values()) == {VACUOUS}
    assert set(table["CONTROL always-false"].values()) == {LIVE}


def test_hard_no_slip_is_vacuous_everywhere_as_expected(matrix):
    """Wall no-slip is a hard constraint, so its test measures the ansatz.

    This is the expected output, not a defect -- but it must be REPORTED, which
    is the whole point of S5. Phase 1 emptied these tests silently.
    """
    table, _ = matrix
    assert set(table["wall no-slip"].values()) == {VACUOUS}


def test_mass_conservation_is_live_under_velocity_pressure(matrix):
    """Test 3 must still bite somewhere, or the suite cannot catch mass drift.

    It was the velocity-pressure run, where this is LIVE, that caught Case B
    creating 6% of mass from nothing.
    """
    table, _ = matrix
    row = table["test 3  mass conservation"]
    assert row["vp-soft"] == LIVE
    assert row["vp-hard"] == LIVE


def test_mass_conservation_is_vacuous_under_psi_omega(matrix):
    """The phi^2 ansatz pins psi on both walls, so the flux cannot vary.

    Recorded as an expected VACUOUS so that anyone quoting a psi-omega test 3
    pass sees immediately that it measures the ansatz.
    """
    table, _ = matrix
    row = table["test 3  mass conservation"]
    assert row["psi-cubic"] == VACUOUS
    assert row["psi-sin"] == VACUOUS


def test_no_check_is_vacuous_in_every_column_except_known_ones(matrix):
    """An all-vacuous row means the property is untested by this suite."""
    table, _ = matrix
    expected = {"wall no-slip", "CONTROL always-true"}
    unexpected = set(all_vacuous_rows(table)) - expected
    assert not unexpected, (
        f"these properties are untested by the whole suite: {sorted(unexpected)}. "
        "An all-vacuous row is a defect report, not something to accept."
    )


def test_matrix_file_is_current(matrix):
    """The committed matrix must match what the code produces now.

    S5 requires regeneration on every ansatz change; this is what makes
    forgetting it a test failure rather than a stale document.
    """
    table, columns = matrix
    assert MATRIX_PATH.is_file(), (
        f"{MATRIX_PATH} is missing. Regenerate with "
        "`python -m src.report --falsifiability`."
    )
    current = format_matrix(table, columns)
    committed = MATRIX_PATH.read_text(encoding="utf-8")
    assert current in committed, (
        "the committed falsifiability matrix is stale. The ansatz changed "
        "without the matrix being regenerated, which is exactly the silent "
        "disarming S5 exists to prevent. Regenerate with "
        "`python -m src.report --falsifiability`."
    )


# --------------------------------------------------------------------------- #
# Degenerate targets: LIVE is not the same as "not free"
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def degenerate():
    from src.evaluation.falsifiability import degenerate_targets

    return {
        label: set(
            degenerate_targets(
                parse_config(yaml.safe_load(open(path, encoding="utf-8")))
            )
        )
        for label, path in COLUMNS.items()
    }


def test_cubic_lift_makes_the_case_a_suite_degenerate(degenerate):
    """SPEC v2 S4 says to check this; measured rather than argued.

    With N_psi identically zero the cubic lift gives relative L2 = 0.000e+00 on
    u -- exactly the analytic solution -- so test 1, test 2, test 3 and the
    inlet condition all pass with the network removed entirely. The sentinel
    verdict is LIVE only because Glorot initialisation is not near zero.
    """
    cubic = degenerate["psi-cubic"]
    assert {"test 1  u relative L2", "test 2  max |v*|",
            "test 3  mass conservation", "inlet profile"} <= cubic


def test_sinusoidal_lift_still_requires_the_network(degenerate):
    """The expressivity check must not be degenerate on test 1 or the inlet."""
    sine = degenerate["psi-sin"]
    assert "test 1  u relative L2" not in sine
    assert "inlet profile" not in sine


def test_transverse_velocity_check_is_degenerate_everywhere(degenerate):
    """test 2 is passed by the zero field in every formulation.

    Not a defect to fix -- v* = 0 really is the answer for Case A -- but it
    means test 2 alone cannot distinguish a correct solution from a dead one,
    and must never be quoted without test 1 beside it.
    """
    for label, names in degenerate.items():
        assert "test 2  max |v*|" in names, label
