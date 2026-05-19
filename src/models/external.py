"""External / comparison signals that aren't part of the in-house ensemble.

These series are pulled from FRED for display alongside our own model output:

* ``RECPROUSM156N`` — Smoothed U.S. Recession Probabilities (Chauvet & Piger),
  published by the St Louis Fed. The de facto "NY Fed-style" public number.
* ``SAHMREALTIME`` — Sahm Rule recession indicator. Fires at 0.5; uses
  *real-time* unemployment data with no NBER dependency, so it carries no
  look-ahead and is not revised after a release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ny_fed_probability(panel: pd.DataFrame) -> pd.Series:
    """Latest NY Fed-style probability (%), monthly indexed.

    Returns an empty series if the FRED column isn't present in the panel.
    """
    if "RECPROUSM156N" not in panel.columns:
        return pd.Series(dtype=float, name="ny_fed_prob")
    s = panel["RECPROUSM156N"].dropna()
    # The series is already in percent (0–100).
    s.name = "ny_fed_prob"
    return s


def sahm_rule(panel: pd.DataFrame) -> pd.Series:
    """Sahm Rule indicator: 3-month MA of UNRATE minus 12-month minimum.

    Threshold is 0.5. Returns the FRED-published series if available;
    otherwise reconstructs it from UNRATE.
    """
    if "SAHMREALTIME" in panel.columns:
        s = panel["SAHMREALTIME"].dropna()
        s.name = "sahm"
        return s
    if "UNRATE" not in panel.columns:
        return pd.Series(dtype=float, name="sahm")
    unrate = panel["UNRATE"].dropna().resample("ME").last()
    three_mo = unrate.rolling(3).mean()
    twelve_mo_min = unrate.rolling(12).min()
    sahm = (three_mo - twelve_mo_min).rename("sahm")
    return sahm.dropna()


def sahm_state(value: float) -> tuple[str, str]:
    """Return (label, severity) for a Sahm Rule reading.

    Severity is one of: 'low', 'elevated', 'high', 'critical'.
    """
    if not np.isfinite(value):
        return "—", "low"
    if value < 0.2:
        return "LOW", "low"
    if value < 0.4:
        return "WARMING", "elevated"
    if value < 0.5:
        return "WATCH", "high"
    return "TRIGGERED", "critical"
