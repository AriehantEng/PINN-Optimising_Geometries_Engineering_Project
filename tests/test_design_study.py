"""The flow-share design study (bifurcation/design_study.py): its pieces
that need no finite-element solve. The study itself is the evidence it writes."""

import copy

import pytest
import yaml

from bifurcation import design_study as d
from developing_flow import pinn


def base():
    with open("bifurcation/study_re10_shares.yaml") as handle:
        return yaml.safe_load(handle)


def test_the_friction_law_split():
    """w_i ~ Q_i^(1/2) minimises sum Q_i^2 / w_i^3 under w1 + w2 = w0."""
    assert d.friction_split(0.5, 0.2) == 0.5
    q = 0.3
    s = d.friction_split(q, 0.2)
    cost = lambda x: q**2 / x**3 + (1 - q) ** 2 / (1 - x) ** 3      # noqa: E731
    h = 1e-6
    assert abs((cost(s + h) - cost(s - h)) / (2 * h)) < 1e-6 * cost(s)
    assert d.friction_split(0.001, 0.2) == 0.1                        # clipped to the bound
    assert d.friction_split(0.999, 0.2) == 0.9


def test_the_study_config_is_checked():
    d.parse_config(base())
    for change, match in (({"shares": [0.0, 0.5]}, "share"), ({"angles": [[60.0]]}, "pairs"),
                          ({"search": {"grid": 2, "xatol": 1e-3}}, "grid")):
        doc = copy.deepcopy(base())
        doc.update(change)
        with pytest.raises(pinn.ConfigError, match=match):
            d.parse_config(doc)
    doc = base()
    del doc["workers"]
    with pytest.raises(pinn.ConfigError, match="workers"):
        d.parse_config(doc)


def test_merging_studies_refuses_duplicates_and_mismatched_settings(tmp_path):
    import json
    src = "docs/bifurcation_re10-shares.json"
    one = d.merge([src], tmp_path / "one")
    assert one["rows"] == json.loads(open(src).read())["rows"]
    with pytest.raises(pinn.ConfigError, match="twice"):
        d.merge([src, src], tmp_path / "twice")
    other = json.loads(open(src).read())
    other["config"]["mesh"]["max_area"] = 0.01               # a different mesh
    for row in other["rows"]:
        row["angles"] = [row["angles"][0] + 1.0, row["angles"][1]]
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(pinn.ConfigError, match="mesh differs"):
        d.merge([src, str(tmp_path / "other.json")], tmp_path / "mixed")


def test_exactly_one_length_rule_is_given():
    doc = base()
    doc["geometry"]["closed_length"] = 8.0                    # both rules
    with pytest.raises(pinn.ConfigError, match="exactly one"):
        d.parse_config(doc)
    del doc["geometry"]["daughter_length"]                    # the closed rule alone
    d.parse_config(doc)
    del doc["geometry"]["closed_length"]                      # neither
    with pytest.raises(pinn.ConfigError, match="exactly one"):
        d.parse_config(doc)


def test_continuation_is_required_and_increases_below_the_reynolds_number():
    doc = base()
    del doc["physics"]["continuation"]
    with pytest.raises(pinn.ConfigError, match="continuation"):
        d.parse_config(doc)
    for steps in ([25.0], [5.0, 5.0], [8.0, 4.0], [0.0]):
        doc = base()
        doc["physics"]["continuation"] = steps
        with pytest.raises(pinn.ConfigError, match="continuation"):
            d.parse_config(doc)
    doc = base()
    doc["physics"]["continuation"] = [2.5, 5.0]
    d.parse_config(doc)


def test_the_closed_rule_gives_both_daughters_the_closed_length():
    from bifurcation import sensitivity as s
    config = base()
    config["geometry"] = {"parent_length": 3.0, "closed_length": 8.0, "min_width": 0.2}
    for angles in ([0.0, 150.0], [60.0, 60.0], [30.0, 150.0]):
        g = d.make_geometry(config, 0.3, angles)
        assert max(abs(x - 8.0) for x in s.closed_lengths(g)) < 1e-12
