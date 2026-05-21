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


def below_trend_breadth(zscores: pd.DataFrame) -> pd.Series:
    """Share of indicators below their own trend (signed z < 0), as a 0–100 %.

    High = broad-based weakness. Months are skipped where no indicator has data.
    """
    if zscores is None or zscores.empty:
        return pd.Series(dtype=float)
    valid = zscores.notna()
    below = (zscores < 0) & valid
    denom = valid.sum(axis=1).replace(0, np.nan)
    return (below.sum(axis=1) / denom * 100.0).dropna()


def momentum_breadth(zscores: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """Share of indicators whose signed z-score *fell* over ``lookback`` months.

    High = deterioration is spreading (momentum breadth), independent of level.
    """
    if zscores is None or zscores.empty:
        return pd.Series(dtype=float)
    delta = zscores - zscores.shift(lookback)
    valid = delta.notna()
    falling = (delta < 0) & valid
    denom = valid.sum(axis=1).replace(0, np.nan)
    return (falling.sum(axis=1) / denom * 100.0).dropna()


def breadth_snapshot(zscores: pd.DataFrame, lookback: int = 3) -> dict:
    """Latest breadth counts: how many indicators are below trend / falling."""
    if zscores is None or zscores.empty:
        return {"below_trend": 0, "falling": 0, "total": 0,
                "below_trend_pct": float("nan"), "falling_pct": float("nan")}
    latest = zscores.dropna(how="all").iloc[-1]
    valid = latest.notna()
    total = int(valid.sum())
    below = int(((latest < 0) & valid).sum())

    delta = (zscores - zscores.shift(lookback)).dropna(how="all")
    if not delta.empty:
        dlatest = delta.iloc[-1]
        dvalid = dlatest.notna()
        falling = int(((dlatest < 0) & dvalid).sum())
        falling_total = int(dvalid.sum())
    else:
        falling, falling_total = 0, 0

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
