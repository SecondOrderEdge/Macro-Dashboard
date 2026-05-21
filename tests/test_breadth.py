"""Breadth / diffusion analytics: counts, percentages, bands."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.breadth import (
    below_trend_breadth,
    breadth_snapshot,
    breadth_state,
    cfnai_diffusion_band,
    momentum_breadth,
)


def _frame():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    # 4 indicators; engineer a clear deterioration over time.
    return pd.DataFrame(
        {
            "a": [2.0, 1.0, 0.0, -1.0, -1.5, -2.0],   # falling, ends below trend
            "b": [1.0, 0.5, 0.2, -0.1, -0.3, -0.5],   # falling, ends below trend
            "c": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.1], # rising, ends above trend
            "d": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],      # rising, above trend
        },
        index=idx,
    )


def test_below_trend_breadth_is_pct():
    z = _frame()
    s = below_trend_breadth(z)
    assert (s >= 0).all() and (s <= 100).all()
    # Final row: a,b below 0 (2 of 4) → 50%.
    assert s.iloc[-1] == 50.0


def test_momentum_breadth_counts_fallers():
    z = _frame()
    m = momentum_breadth(z, lookback=3)
    # Final row vs 3 months earlier: a and b fell, c and d rose → 50%.
    assert m.iloc[-1] == 50.0
    assert (m >= 0).all() and (m <= 100).all()


def test_breadth_snapshot_counts():
    snap = breadth_snapshot(_frame(), lookback=3)
    assert snap["total"] == 4
    assert snap["below_trend"] == 2          # a, b
    assert snap["falling"] == 2              # a, b fell over 3m
    assert snap["below_trend_pct"] == 50.0


def test_breadth_snapshot_handles_empty():
    snap = breadth_snapshot(pd.DataFrame())
    assert snap["total"] == 0
    assert not np.isfinite(snap["below_trend_pct"])


def test_cfnai_diffusion_band_directionality():
    assert cfnai_diffusion_band(0.3)[0] == "BROAD GROWTH"
    assert cfnai_diffusion_band(-0.5)[0] == "BROAD WEAKNESS"
    assert cfnai_diffusion_band(float("nan"))[0] == "—"


def test_breadth_state_directionality():
    assert breadth_state(20)[1] == "low"
    assert breadth_state(80)[1] == "critical"
