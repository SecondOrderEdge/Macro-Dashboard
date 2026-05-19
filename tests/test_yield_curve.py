"""Yield curve: spreads, inversion detection, hit rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.nber import load_nber_recessions
from src.models.yield_curve import YieldCurve


def _synthetic_curve(spread_path: pd.Series) -> pd.DataFrame:
    """Build a daily panel with a known 10Y-3M spread path."""
    idx = pd.date_range(spread_path.index.min(), spread_path.index.max(), freq="D")
    spread = spread_path.reindex(idx).ffill().bfill()
    panel = pd.DataFrame(index=idx)
    panel["DGS10"] = 4.0
    panel["DGS3MO"] = panel["DGS10"] - spread
    panel["DGS2"] = panel["DGS10"] - spread * 0.7
    panel["DGS5"] = panel["DGS10"] - spread * 0.4
    return panel


def test_spread_calculation():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    panel = pd.DataFrame({"DGS10": np.full(len(idx), 4.0), "DGS3MO": np.full(len(idx), 1.0)}, index=idx)
    panel["DGS2"] = 3.0
    panel["DGS5"] = 3.5
    yc = YieldCurve(panel)
    spreads = yc.spreads_history()
    assert "spread_10y3m" in spreads.columns
    assert np.allclose(spreads["spread_10y3m"].dropna().values, 3.0)
    assert np.allclose(spreads["spread_10y2y"].dropna().values, 1.0)


def test_inversion_detection_on_synthetic_curve():
    months = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    spread = pd.Series(1.0, index=months)
    spread.loc["2022-06-01":"2024-03-01"] = -0.5  # ~22 months inverted
    panel = _synthetic_curve(spread)

    yc = YieldCurve(panel)
    nber = load_nber_recessions(start="2018-01-01", end="2024-12-31")
    stats = yc.inversion_stats(nber)

    assert isinstance(stats["months_inverted"], int)
    assert stats["hit_rate"][1] >= 1  # at least one inversion episode detected


def test_hit_rate_counts_correctly_given_known_recessions():
    """Two inversion episodes, one followed by a recession peak."""
    months = pd.date_range("1990-01-01", "2010-12-01", freq="MS")
    spread = pd.Series(1.5, index=months)
    # Episode 1: starts 2000-06, lasts 6 months, followed by NBER peak in 2001-03
    spread.loc["2000-06-01":"2000-11-01"] = -0.3
    # Episode 2: starts 2007-01, lasts 8 months, followed by NBER peak 2007-12
    spread.loc["2007-01-01":"2007-08-01"] = -0.2
    panel = _synthetic_curve(spread)

    yc = YieldCurve(panel)
    nber = load_nber_recessions(start="1990-01-01", end="2010-12-31")
    stats = yc.inversion_stats(nber)

    hits, total = stats["hit_rate"]
    assert total == 2
    assert hits == 2  # both inversions precede a real NBER peak


def test_term_structure_returns_expected_columns():
    months = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    spread = pd.Series(1.0, index=months)
    panel = _synthetic_curve(spread)
    yc = YieldCurve(panel)
    ts = yc.term_structure()
    assert {"maturity", "current", "years"}.issubset(ts.columns)
    assert "m3" in ts.columns
    assert "m12" in ts.columns
