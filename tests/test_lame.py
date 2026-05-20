"""LAME: z-score properties, weight normalization, sign convention."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.series_registry import SERIES_REGISTRY, label_for
from src.models.lame import LAME


def test_every_registry_series_has_a_plain_english_name():
    for key, meta in SERIES_REGISTRY.items():
        assert meta.get("name"), f"{key} is missing a plain-English name"


def test_label_for_maps_keys_and_falls_back():
    assert label_for("unrate") == "Unemployment Rate"
    assert label_for("hy_oas") == "High-Yield Credit Spread (OAS)"
    assert label_for("not_a_key") == "not_a_key"  # graceful fallback


def test_lame_indicators_all_have_labels():
    for key in LAME.INDICATORS:
        assert label_for(key) != key, f"{key} has no registry label"


def test_zscores_normalized(synthetic_panel):
    m = LAME()
    m.compute(synthetic_panel)
    z = m._zscores.dropna(how="all")
    # Each column's z-scores should average to near zero with std near 1, by construction.
    for col in z.columns:
        vals = z[col].dropna()
        if len(vals) < 60:
            continue
        # Expanding-window z-scores aren't strictly mean=0 std=1 on the tail,
        # but should sit within reasonable bounds.
        assert abs(vals.mean()) < 1.5
        assert 0.3 < vals.std() < 3.0


def test_weights_sum_to_one(synthetic_panel):
    m = LAME()
    m.compute(synthetic_panel)
    w = m._weights.dropna(how="all")
    totals = w.sum(axis=1)
    # Where any weight exists, totals should be ~1.
    valid = totals[totals > 0]
    assert ((valid - 1.0).abs() < 1e-9).all()


def test_breakdown_weights_sum_to_one(synthetic_panel):
    m = LAME()
    m.compute(synthetic_panel)
    bd = m.current_breakdown()
    assert abs(bd["weight"].sum() - 1.0) < 1e-9


def test_inverted_indicators_have_correct_contribution_sign(synthetic_panel):
    """A rise in UNRATE (sign=-1) should pull the LAME composite down."""
    m = LAME()
    m.compute(synthetic_panel)
    z = m._zscores

    # Get latest row that has a finite UNRATE z-score
    unrate_z = z["unrate"].dropna()
    civpart_z = z["civpart"].dropna()
    assert SERIES_REGISTRY["unrate"]["sign"] == -1
    assert SERIES_REGISTRY["civpart"]["sign"] == 1

    # Build a controlled panel where UNRATE rises sharply at the tail.
    spiked = synthetic_panel.copy()
    spiked.loc[spiked.index[-12]:, "UNRATE"] = spiked["UNRATE"].iloc[-12] + 3.0

    m2 = LAME()
    m2.compute(spiked)
    z2 = m2._zscores["unrate"].dropna()
    # Higher UNRATE values → after sign flip (-1), z2 tail should be < z tail.
    assert z2.iloc[-1] < unrate_z.iloc[-1]
