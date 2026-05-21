"""Breadth / diffusion analytics.

Breadth answers a question the recession ensemble can't on its own: *is the
weakness broad or narrow?* A handful of indicators rolling over is noise; a
majority rolling over together is how real downturns begin. These are pure
functions over a signed z-score panel (positive = expansionary), so they're
unit-testable without touching FRED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def below_trend_breadth(zscores: pd.DataFrame, min_coverage: float = 0.5) -> pd.Series:
    """Share of indicators below their own trend (signed z < 0), as a 0–100 %.

    High = broad-based weakness. Months with less than ``min_coverage`` of
    indicators reporting are skipped, so the ragged right edge (where only the
    fastest series have printed) doesn't produce a jumpy, low-N reading.
    """
    if zscores is None or zscores.empty:
        return pd.Series(dtype=float)
    valid = zscores.notna()
    n = valid.sum(axis=1)
    keep = n >= max(1, int(np.ceil(min_coverage * zscores.shape[1])))
    below = ((zscores < 0) & valid).sum(axis=1)
    pct = (below / n.replace(0, np.nan) * 100.0)
    return pct[keep].dropna()


def momentum_breadth(zscores: pd.DataFrame, lookback: int = 3, min_coverage: float = 0.5) -> pd.Series:
    """Share of indicators whose signed z-score *fell* over ``lookback`` months.

    High = deterioration is spreading (momentum breadth), independent of level.
    """
    if zscores is None or zscores.empty:
        return pd.Series(dtype=float)
    delta = zscores - zscores.shift(lookback)
    valid = delta.notna()
    n = valid.sum(axis=1)
    keep = n >= max(1, int(np.ceil(min_coverage * zscores.shape[1])))
    falling = ((delta < 0) & valid).sum(axis=1)
    pct = (falling / n.replace(0, np.nan) * 100.0)
    return pct[keep].dropna()


def _last_valid_per_col(df: pd.DataFrame) -> pd.Series:
    """Each column's most recent non-NaN value (ragged-edge aware)."""
    return pd.Series(
        {c: df[c].dropna().iloc[-1] for c in df.columns if df[c].notna().any()},
        dtype=float,
    )


def breadth_snapshot(zscores: pd.DataFrame, lookback: int = 3) -> dict:
    """Latest breadth counts using each indicator's *own* freshest reading.

    Reading a single shared row would undercount at the ragged edge (where only
    a couple of fast series have reported the current month); instead we take
    the last valid value of each indicator.
    """
    empty = {"below_trend": 0, "falling": 0, "total": 0, "falling_total": 0,
             "below_trend_pct": float("nan"), "falling_pct": float("nan")}
    if zscores is None or zscores.empty:
        return empty

    latest = _last_valid_per_col(zscores)
    total = int(latest.size)
    below = int((latest < 0).sum())

    dlatest = _last_valid_per_col(zscores - zscores.shift(lookback))
    falling_total = int(dlatest.size)
    falling = int((dlatest < 0).sum())

    return {
        "below_trend": below,
        "falling": falling,
        "total": total,
        "falling_total": falling_total,
        "below_trend_pct": (below / total * 100.0) if total else float("nan"),
        "falling_pct": (falling / falling_total * 100.0) if falling_total else float("nan"),
    }


def cfnai_diffusion_band(value: float) -> tuple[str, str]:
    """Map FRED's CFNAI Diffusion Index to (label, severity).

    The diffusion index summarises how broadly the ~85 CFNAI components are
    contributing. It runs roughly [-0.8, +0.6]; sustained readings below about
    -0.35 have historically accompanied recessions.
    """
    if value is None or not np.isfinite(value):
        return "—", "low"
    if value >= 0.0:
        return "BROAD GROWTH", "low"
    if value >= -0.20:
        return "MIXED", "elevated"
    if value >= -0.35:
        return "NARROWING", "high"
    return "BROAD WEAKNESS", "critical"


def breadth_state(below_trend_pct: float) -> tuple[str, str]:
    """Map the share-below-trend reading to (label, severity)."""
    if not np.isfinite(below_trend_pct):
        return "—", "low"
    if below_trend_pct < 40:
        return "BROAD STRENGTH", "low"
    if below_trend_pct < 55:
        return "MIXED", "elevated"
    if below_trend_pct < 70:
        return "WEAKENING", "high"
    return "BROAD WEAKNESS", "critical"
