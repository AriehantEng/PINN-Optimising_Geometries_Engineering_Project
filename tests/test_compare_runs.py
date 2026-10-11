"""Side-by-side table of parametric runs (bifurcation/compare_runs.py)."""

import copy

import pytest
import yaml

from bifurcation import compare_runs as C


def fake(label, scale):
    metrics = {"splits": [{"split": s, "relative_l2": {"u": scale * s, "v": 2 * scale, "p": scale},
                           "drop_error": scale * (s - 0.5), "pinn_offset": 0.5,
                           "reference_offset": 0.5 - scale} for s in (0.35, 0.5, 0.65)],
               "optimum": {"argmin_error": -0.0003, "regret": 4e-6, "min_drop_error": 7e-4},
               "training": {"selected": "lm:62", "selected_heldout_loss": 0.0148,
                            "total_seconds": 600.0, "parameter_count": 5487}}
    history = [{"phase": "adam"}] + [{"phase": "lm"}] * 3
    return C.summary(label, metrics, history)


def test_the_summary_and_table():
    rows = [fake("a", 0.01), fake("b", 0.002)]
    assert rows[0]["lm_steps"] == 3 and rows[0]["minutes"] == 10.0
    assert rows[1]["start_heldout"] is None
    text = C.markdown(rows, "test")
    assert "| a | 5.00e-03 | 6.50e-03 |" in text
    assert text.count("\n| ") == 6           # per table: a header row and two runs


def test_the_config_is_checked():
    document = yaml.safe_load(open(C.ROOT / "bifurcation/compare_re10_options.yaml"))
    C.parse_config(document)
    bad = copy.deepcopy(document)
    bad["runs"][0]["seed"] = 1
    with pytest.raises(C.ConfigError, match="runs\\[0\\]"):
        C.parse_config(bad)
