"""The junction-against-Reynolds analysis (bifurcation/junction_re.py)."""

import numpy as np
import pytest
import yaml

from bifurcation import junction_re as J


def fake_row(angles, share, best, murray, steady=True, shift=0.0):
    """A stored optimum whose coarse J is a parabola about `best` (J''/J = 16
    there), and a fine mesh whose relative error is linear in s with slope
    -16 shift, which moves the minimum by `shift` to first order (S1.9)."""
    grid = np.linspace(0.1, 0.9, 17)

    def coarse(s):
        return 5.0 + 40.0 * (s - best) ** 2

    def fine(s):
        return coarse(s) * (1.0 - 16.0 * shift * (s - best))

    k = int(np.argmin([coarse(s) for s in grid]))
    left, right = grid[k - 1], grid[k + 1]
    return {"angles": angles, "share": share, "feasible": True, "best_split": best,
            "on_bound": False, "total_pressure_drop": coarse(best),
            "total_pressure_drop_fine": fine(best), "steady_bracket": steady,
            "friction_law_split": murray, "friction_law_drop": coarse(murray),
            "grid": [{"split": float(s), "total_pressure_drop": coarse(s)} for s in grid],
            "fine_bracket": [{"split": float(s), "total_pressure_drop": fine(s)}
                             for s in (left, best, right)],
            "unsolved": [] if steady else [{"split": float(right), "mesh": "search"}]}


def study(rows):
    return {"config": {"search": {"xatol": 0.001}, "physics": {"reynolds": 50.0},
                       "geometry": {"closed_length": 8.0}}, "rows": rows}


def test_a_resolved_departure_toward_the_turned_daughter():
    rec, = J.records(study([fake_row([0.0, 120.0], 0.3, 0.390, 0.3956)]))
    assert rec["trusted"] and rec["resolved"] and rec["widens_turned"]
    assert abs(rec["departure"] - (0.390 - 0.3956)) < 1e-15
    assert abs(rec["bound"] - 0.001) < 1e-9          # no mesh shift: the search tolerance
    assert rec["murray_regret"] > 0.0


def test_the_mesh_shift_widens_the_bound():
    rec, = J.records(study([fake_row([60.0, 60.0], 0.5, 0.5, 0.5, shift=0.002)]))
    assert rec["bound"] == pytest.approx(0.003, rel=1e-6)
    assert not rec["resolved"] and rec["widens_turned"] is None


def test_an_optimum_next_to_an_unsteady_split_is_not_trusted():
    rec, = J.records(study([fake_row([30.0, 150.0], 0.05, 0.2, 0.1866, steady=False)]))
    assert not rec["trusted"]


def test_the_config_is_checked():
    document = yaml.safe_load(open(J.ROOT / "bifurcation/junction_re.yaml"))
    J.parse_config(document)
    del document["maps"]
    with pytest.raises(J.ConfigError):
        J.parse_config(document)
