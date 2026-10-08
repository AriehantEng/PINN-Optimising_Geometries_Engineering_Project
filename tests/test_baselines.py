"""Baselines for Network 3 and the length behind s_L (bifurcation/baselines.py)."""

import json

import pytest
import yaml

from bifurcation import baselines as B
from bifurcation import sensitivity as S

STUDY = json.loads((B.ROOT / "docs/bifurcation_re10-shares-all.json").read_text())["config"]
RECORDS = json.loads((B.ROOT / "docs/bifurcation_re10-sensitivity.json").read_text())["records"]


def test_the_config_is_checked():
    document = yaml.safe_load(open(B.ROOT / "bifurcation/baselines_re10.yaml"))
    B.parse_config(document)
    del document["sensitivity"]
    with pytest.raises(B.ConfigError):
        B.parse_config(document)


def test_every_held_out_case_has_a_network_split():
    config = B.parse_config(yaml.safe_load(open(B.ROOT / "bifurcation/baselines_re10.yaml")))
    splits = B.network3_splits(B.ROOT / config["network3"])
    assert len(splits) == 18
    for a1, a2 in config["pairs"]:
        for share in config["shares"]:
            assert 0.1 < splits[(a1, a2, share)] < 0.9


def test_the_friction_law_at_the_murray_split_is_seven():
    """J_M = 6.4 + 0.6 at q = 0.5 with outer-wall lengths (S1.2)."""
    length = STUDY["geometry"]["daughter_length"]
    assert abs(B.friction_drop(STUDY, (length, length), 0.5, 0.5) - 7.0) < 1e-12


def test_moving_the_friction_baseline_back_to_the_centreline_changes_nothing():
    """The shift between baselines is exact: back to the centreline, the
    sensitivity run's coefficients are returned unchanged."""
    for r in RECORDS[::17]:
        k = B._coefficients(STUDY, r, S.centreline_lengths)
        assert max(abs(a - b) for a, b in zip(k, r["K_measured"])) < 1e-12


def test_a_closed_baseline_raises_every_coefficient():
    """Closed lengths never exceed the centreline, so less friction is removed."""
    for r in RECORDS[::11]:
        k = B._coefficients(STUDY, r, S.closed_lengths)
        assert all(x >= y - 1e-12 for x, y in zip(k, r["K_measured"]))


def test_network3_drops_match_its_splits_and_are_close_to_the_optimum():
    """Same 18 cases as the split table; predicted and finite-element drops agree
    to the few per cent Network 3 reports."""
    path = B.ROOT / "docs/network3_re10_all_w16.md"
    drops, splits = B.network3_drops(path), B.network3_splits(path)
    assert drops.keys() == splits.keys()
    for predicted, fine in drops.values():
        assert 3.0 < fine < 13.0 and abs(predicted / fine - 1.0) < 0.05
