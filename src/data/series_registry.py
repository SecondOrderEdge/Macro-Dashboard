"""Single source of truth for the FRED series the dashboard uses.

Each entry carries the FRED ID, native frequency, the transform applied
before z-scoring or regression, and (for labor indicators) the sign
convention so positive values always mean "more expansion".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SERIES_REGISTRY: dict[str, dict] = {
    # Yield curve
    "t10y3m": {"fred_id": "T10Y3M", "freq": "D", "transform": "level"},
    "t10y2y": {"fred_id": "T10Y2Y", "freq": "D", "transform": "level"},
    "dgs1mo": {"fred_id": "DGS1MO", "freq": "D", "transform": "level"},
    "dgs3mo": {"fred_id": "DGS3MO", "freq": "D", "transform": "level"},
    "dgs6mo": {"fred_id": "DGS6MO", "freq": "D", "transform": "level"},
    "dgs1":   {"fred_id": "DGS1",   "freq": "D", "transform": "level"},
    "dgs2":   {"fred_id": "DGS2",   "freq": "D", "transform": "level"},
    "dgs3":   {"fred_id": "DGS3",   "freq": "D", "transform": "level"},
    "dgs5":   {"fred_id": "DGS5",   "freq": "D", "transform": "level"},
    "dgs7":   {"fred_id": "DGS7",   "freq": "D", "transform": "level"},
    "dgs10":  {"fred_id": "DGS10",  "freq": "D", "transform": "level"},
    "dgs20":  {"fred_id": "DGS20",  "freq": "D", "transform": "level"},
    "dgs30":  {"fred_id": "DGS30",  "freq": "D", "transform": "level"},

    # Labor (used by LAME and recession ensemble)
    "unrate":    {"fred_id": "UNRATE",    "freq": "M", "transform": "level",   "sign": -1},
    "icsa":      {"fred_id": "ICSA",      "freq": "W", "transform": "ma4",     "sign": -1},
    "ccsa":      {"fred_id": "CCSA",      "freq": "W", "transform": "level",   "sign": -1},
    "jtsjol":    {"fred_id": "JTSJOL",    "freq": "M", "transform": "level",   "sign": +1},
    "jtsqur":    {"fred_id": "JTSQUR",    "freq": "M", "transform": "level",   "sign": +1},
    "awhaetp":   {"fred_id": "AWHAETP",   "freq": "M", "transform": "level",   "sign": +1},
    "temphelps": {"fred_id": "TEMPHELPS", "freq": "M", "transform": "yoy",     "sign": +1},
    "payems":    {"fred_id": "PAYEMS",    "freq": "M", "transform": "diff_3m", "sign": +1},
    "u6rate":    {"fred_id": "U6RATE",    "freq": "M", "transform": "level",   "sign": -1},
    "civpart":   {"fred_id": "CIVPART",   "freq": "M", "transform": "level",   "sign": +1},

    # Credit
    "baa10y":   {"fred_id": "BAA10Y",       "freq": "D", "transform": "level"},
    "drtscilm": {"fred_id": "DRTSCILM",     "freq": "Q", "transform": "level"},
    "hy_oas":   {"fred_id": "BAMLH0A0HYM2", "freq": "D", "transform": "level"},

    # Housing & real
    "permit": {"fred_id": "PERMIT", "freq": "M", "transform": "yoy"},
    "houst":  {"fred_id": "HOUST",  "freq": "M", "transform": "level"},
    "pcec96": {"fred_id": "PCEC96", "freq": "M", "transform": "yoy"},

    # Sentiment & market
    "vixcls":  {"fred_id": "VIXCLS",  "freq": "D", "transform": "ma_3m"},
    "usslind": {"fred_id": "USSLIND", "freq": "M", "transform": "level"},
    "sp500":   {"fred_id": "SP500",   "freq": "D", "transform": "ret_6m"},
}


def fred_ids() -> list[str]:
    """Convenience: list every FRED ID we know about."""
    return [meta["fred_id"] for meta in SERIES_REGISTRY.values()]


def fred_id_for(name: str) -> str:
    return SERIES_REGISTRY[name]["fred_id"]


_VALID_TRANSFORMS = {"level", "yoy", "diff_3m", "ma4", "ma_3m", "ret_6m"}


def transform_series(series: pd.Series, transform: str) -> pd.Series:
    """Apply one of the supported transforms.

    - level   : no change
    - yoy     : 12-period percent change (×100)
    - diff_3m : 3-period difference (raw units)
    - ma4     : trailing 4-period mean
    - ma_3m   : trailing 3-month mean (resamples daily to month-end first)
    - ret_6m  : 6-month percent change (×100); resamples daily to month-end
    """
    if transform not in _VALID_TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; expected one of {_VALID_TRANSFORMS}.")

    s = series.dropna().astype(float)
    if s.empty:
        return s

    if transform == "level":
        out = s
    elif transform == "yoy":
        out = s.pct_change(12) * 100.0
    elif transform == "diff_3m":
        out = s.diff(3)
    elif transform == "ma4":
        out = s.rolling(window=4, min_periods=1).mean()
    elif transform == "ma_3m":
        monthly = s.resample("ME").mean() if _is_subdaily(s) else s.rolling(3, min_periods=1).mean()
        out = monthly
    elif transform == "ret_6m":
        monthly = s.resample("ME").last() if _is_subdaily(s) else s
        out = monthly.pct_change(6) * 100.0
    else:  # pragma: no cover - guarded above
        out = s

    return out.replace([np.inf, -np.inf], np.nan)


def _is_subdaily(series: pd.Series) -> bool:
    """Heuristic: treat any series with median spacing ≤ 7 days as daily/weekly."""
    if len(series) < 3:
        return False
    diffs = series.index.to_series().diff().dropna()
    if diffs.empty:
        return False
    return diffs.median() <= pd.Timedelta(days=7)
