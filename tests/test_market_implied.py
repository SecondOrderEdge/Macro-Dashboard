"""Market-implied signal summaries: latest, change, percentile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.market_implied import signal_summary


def _daily(values, end="2026-05-01"):
    idx = pd.date_range(end=end, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_signal_summary_latest_and_percentile():
    # Strictly increasing → latest is the max → 100th percentile, positive change.
    s = _daily(np.linspace(1.0, 3.0, 400))
    out = signal_summary(s)
    assert out["latest"] == 3.0
    assert out["percentile"] == 100.0
    assert out["change"] > 0
    assert out["as_of"] is not None


def test_signal_summary_change_is_one_month():
    s = _daily(np.linspace(0.0, 10.0, 200))
    out = signal_summary(s)
    # ~1 month of daily steps of (10/199) each → change over ~30 days is positive.
    assert np.isfinite(out["change"]) and out["change"] > 0


def test_signal_summary_handles_empty():
    out = signal_summary(pd.Series(dtype=float))
    assert not np.isfinite(out["latest"])
    assert out["as_of"] is None


def test_signal_summary_midrange_percentile():
    # A 0→100 ramp with a final value of 50 → ~median of its window.
    vals = list(np.linspace(0, 100, 400)) + [50.0]
    s = _daily(vals)
    out = signal_summary(s)
    assert 30 <= out["percentile"] <= 70
