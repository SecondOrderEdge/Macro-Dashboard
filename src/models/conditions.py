"""Financial-conditions and activity indices, sourced live from FRED.

These series are *parallel display indicators* — they are not inputs to the
recession-probability ensemble. Including financial-conditions composites
inside the credit submodel would introduce strong collinearity with the
spread series we already use.

Series definitions:

* ``NFCI`` (weekly) — Chicago Fed National Financial Conditions Index. A
  z-score across 100+ financial market, credit and leverage variables.
  Positive = tighter-than-average; zero is the long-run average.

* ``ANFCI`` (weekly) — Chicago Fed Adjusted NFCI. Same as NFCI but with
  the cyclical macroeconomic component partialled out, so it isolates the
  purely financial component of stress.

* ``STLFSI4`` (weekly) — St Louis Fed Financial Stress Index, current
  release. Composite of 18 financial market measures. Same zero-mean
  convention as NFCI.

* ``CFNAI3MA`` (monthly) — Chicago Fed National Activity Index, 3-month
  moving average. A coincident composite of 85 real-economy indicators.
  The canonical recession-signal threshold is < -0.7; the canonical
  overheating threshold is > +0.7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- fetchers


def nfci(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "NFCI")


def anfci(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "ANFCI")


def stlfsi(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "STLFSI4")


def cfnai_3ma(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "CFNAI3MA")


def wage_tracker(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "FRBATLWGT12MMUMHWGO")


def _series(panel: pd.DataFrame, fred_id: str) -> pd.Series:
    if fred_id not in panel.columns:
        return pd.Series(dtype=float, name=fred_id)
    s = panel[fred_id].dropna()
    s.name = fred_id
    return s


# ---------------------------------------------------------------- bands


def stress_band(value: float) -> tuple[str, str]:
    """Map NFCI/ANFCI/STLFSI4 value to (label, severity).

    Convention: positive = tighter-than-average financial conditions.
    """
    if not np.isfinite(value):
        return "—", "low"
    if value < -0.5:
        return "EASY", "low"
    if value < 0.5:
        return "AVERAGE", "low"
    if value < 1.0:
        return "TIGHTENING", "elevated"
    if value < 2.0:
        return "STRESSED", "high"
    return "CRISIS", "critical"


def cfnai_band(value: float) -> tuple[str, str]:
    """Map CFNAI3MA value to (label, severity).

    Canonical thresholds: > +0.7 overheating; -0.7 to +0.7 normal;
    -0.7 to -1.5 recession warning; < -1.5 deep recession.
    """
    if not np.isfinite(value):
        return "—", "low"
    if value > 0.7:
        return "ABOVE TREND", "low"
    if value > -0.7:
        return "AT TREND", "low"
    if value > -1.5:
        return "BELOW TREND", "high"
    return "RECESSION ZONE", "critical"


def wage_band(value: float) -> tuple[str, str]:
    """Map Atlanta Fed wage growth value (% YoY) to (label, severity).

    Long-run average since 1997 is around 3.5%. Sustained readings above
    4.5% historically coincide with Fed tightening cycles.
    """
    if not np.isfinite(value):
        return "—", "low"
    if value < 2.5:
        return "SOFT", "high"
    if value < 4.0:
        return "NORMAL", "low"
    if value < 5.0:
        return "FIRM", "elevated"
    return "HOT", "critical"
