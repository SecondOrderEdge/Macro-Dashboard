"""Composite risk: bounds, weighting, bands."""

from __future__ import annotations

import pytest

from src.models.composite import composite_risk, curve_to_risk, lame_to_risk


@pytest.mark.parametrize(
    "ensemble,lame,curve",
    [(0, 2.0, 3.0), (50, 0.0, 0.0), (100, -2.0, -3.0), (35, -0.5, 0.2)],
)
def test_composite_within_bounds(ensemble, lame, curve):
    out = composite_risk(ensemble, lame, curve)
    assert 0 <= out["composite"] <= 100
    assert out["band"] in {"LOW", "ELEVATED", "HIGH", "CRITICAL"}


def test_lame_to_risk_inverts_correctly():
    assert lame_to_risk(-2.0) == 100.0
    assert lame_to_risk(0.0) == 50.0
    assert lame_to_risk(2.0) == 0.0
    assert lame_to_risk(-10.0) == 100.0  # clipped


def test_curve_to_risk_inverts_correctly():
    assert curve_to_risk(-2.5) == 100.0
    assert curve_to_risk(0.0) == 50.0
    assert curve_to_risk(2.5) == 0.0
    assert curve_to_risk(10.0) == 0.0  # clipped


def test_composite_equals_weighted_sum_at_extremes():
    """When inputs are at the 0-risk anchors, composite == 0."""
    out = composite_risk(0.0, 2.0, 2.5)
    assert out["composite"] == 0
    assert out["band"] == "LOW"

    out = composite_risk(100.0, -2.0, -2.5)
    assert out["composite"] == 100
    assert out["band"] == "CRITICAL"


def test_band_boundaries():
    assert composite_risk(20, 1.2, 2.5)["band"] in {"LOW", "ELEVATED"}
    assert composite_risk(40, 0, 0)["band"] in {"ELEVATED", "HIGH"}
    # Mid-range: ensemble=50, lame=0, curve=0 ⇒ composite = 50 → HIGH
    assert composite_risk(50, 0, 0)["band"] == "HIGH"
    assert composite_risk(80, -1.5, -1.5)["band"] == "CRITICAL"


def test_contributions_sum_close_to_composite():
    out = composite_risk(40, -0.5, 0.5)
    total = sum(out["contributions"].values())
    assert abs(total - out["composite"]) < 1.0


def test_handles_missing_inputs():
    out = composite_risk(float("nan"), -0.5, 0.0)
    assert 0 <= out["composite"] <= 100
    assert out["band"] in {"LOW", "ELEVATED", "HIGH", "CRITICAL"}
