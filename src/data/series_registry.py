"""Single source of truth for the FRED series the dashboard uses.

Each entry carries the FRED ID, native frequency, the transform applied
before z-scoring or regression, and (for labor indicators) the sign
convention so positive values always mean "more expansion".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SERIES_REGISTRY: dict[str, dict] = {
    # Yield curve — money-market / front-end
    "sofr": {"fred_id": "SOFR", "freq": "D", "transform": "level", "name": "Secured Overnight Financing Rate"},
    "effr": {"fred_id": "DFF",  "freq": "D", "transform": "level", "name": "Effective Federal Funds Rate"},
    "iorb": {"fred_id": "IORB", "freq": "D", "transform": "level", "name": "Interest on Reserve Balances"},

    # Yield curve — Treasury constant-maturity
    "t10y3m": {"fred_id": "T10Y3M", "freq": "D", "transform": "level", "name": "10-Year minus 3-Month Treasury spread"},
    "t10y2y": {"fred_id": "T10Y2Y", "freq": "D", "transform": "level", "name": "10-Year minus 2-Year Treasury spread"},
    "dgs1mo": {"fred_id": "DGS1MO", "freq": "D", "transform": "level", "name": "1-Month Treasury yield"},
    "dgs3mo": {"fred_id": "DGS3MO", "freq": "D", "transform": "level", "name": "3-Month Treasury yield"},
    "dgs6mo": {"fred_id": "DGS6MO", "freq": "D", "transform": "level", "name": "6-Month Treasury yield"},
    "dgs1":   {"fred_id": "DGS1",   "freq": "D", "transform": "level", "name": "1-Year Treasury yield"},
    "dgs2":   {"fred_id": "DGS2",   "freq": "D", "transform": "level", "name": "2-Year Treasury yield"},
    "dgs3":   {"fred_id": "DGS3",   "freq": "D", "transform": "level", "name": "3-Year Treasury yield"},
    "dgs5":   {"fred_id": "DGS5",   "freq": "D", "transform": "level", "name": "5-Year Treasury yield"},
    "dgs7":   {"fred_id": "DGS7",   "freq": "D", "transform": "level", "name": "7-Year Treasury yield"},
    "dgs10":  {"fred_id": "DGS10",  "freq": "D", "transform": "level", "name": "10-Year Treasury yield"},
    "dgs20":  {"fred_id": "DGS20",  "freq": "D", "transform": "level", "name": "20-Year Treasury yield"},
    "dgs30":  {"fred_id": "DGS30",  "freq": "D", "transform": "level", "name": "30-Year Treasury yield"},

    # Labor (used by LAME and recession ensemble)
    "unrate":    {"fred_id": "UNRATE",    "freq": "M", "transform": "level",   "sign": -1, "name": "Unemployment Rate"},
    "icsa":      {"fred_id": "ICSA",      "freq": "W", "transform": "ma4",     "sign": -1, "name": "Initial Jobless Claims"},
    "ccsa":      {"fred_id": "CCSA",      "freq": "W", "transform": "level",   "sign": -1, "name": "Continued Jobless Claims"},
    "jtsjol":    {"fred_id": "JTSJOL",    "freq": "M", "transform": "level",   "sign": +1, "name": "Job Openings (JOLTS)"},
    "jtsqur":    {"fred_id": "JTSQUR",    "freq": "M", "transform": "level",   "sign": +1, "name": "Quits Rate (JOLTS)"},
    "awhaetp":   {"fred_id": "AWHAETP",   "freq": "M", "transform": "level",   "sign": +1, "name": "Avg Weekly Hours, Private"},
    "temphelps": {"fred_id": "TEMPHELPS", "freq": "M", "transform": "yoy",     "sign": +1, "name": "Temporary-Help Payrolls"},
    "payems":    {"fred_id": "PAYEMS",    "freq": "M", "transform": "diff_3m", "sign": +1, "name": "Total Nonfarm Payrolls"},
    "u6rate":    {"fred_id": "U6RATE",    "freq": "M", "transform": "level",   "sign": -1, "name": "U-6 Underemployment Rate"},
    "civpart":   {"fred_id": "CIVPART",   "freq": "M", "transform": "level",   "sign": +1, "name": "Labor-Force Participation Rate"},

    # Credit
    "baa10y":   {"fred_id": "BAA10Y",       "freq": "D", "transform": "level", "name": "Baa Corporate minus 10-Year Treasury spread"},
    "drtscilm": {"fred_id": "DRTSCILM",     "freq": "Q", "transform": "level", "name": "Banks Tightening C&I Loan Standards"},
    "hy_oas":   {"fred_id": "BAMLH0A0HYM2", "freq": "D", "transform": "level", "name": "High-Yield Credit Spread (OAS)"},

    # Housing & real
    "permit": {"fred_id": "PERMIT", "freq": "M", "transform": "yoy", "name": "Building Permits"},
    "houst":  {"fred_id": "HOUST",  "freq": "M", "transform": "level", "name": "Housing Starts"},
    "pcec96": {"fred_id": "PCEC96", "freq": "M", "transform": "yoy", "name": "Real Personal Consumption"},

    # Sentiment & market
    "vixcls":  {"fred_id": "VIXCLS",  "freq": "D", "transform": "ma_3m", "name": "VIX Volatility Index"},
    "usslind": {"fred_id": "USSLIND", "freq": "M", "transform": "level", "name": "Leading Index for the U.S."},
    "sp500":   {"fred_id": "SP500",   "freq": "D", "transform": "ret_6m", "name": "S&P 500"},

    # External / comparison probabilities (not used as inputs; pulled for display).
    "ny_fed_prob": {"fred_id": "RECPROUSM156N", "freq": "M", "transform": "level", "name": "Smoothed Recession Probability (Chauvet–Piger)"},

    # Real-time recession nowcast — Sahm Rule (no NBER look-ahead).
    "sahm": {"fred_id": "SAHMREALTIME", "freq": "M", "transform": "level", "name": "Sahm Rule Recession Indicator"},

    # Financial-conditions indices (all weekly, pulled for display only).
    "nfci":    {"fred_id": "NFCI",    "freq": "W", "transform": "level", "name": "National Financial Conditions Index"},
    "anfci":   {"fred_id": "ANFCI",   "freq": "W", "transform": "level", "name": "Adjusted National Financial Conditions Index"},
    "stlfsi":  {"fred_id": "STLFSI4", "freq": "W", "transform": "level", "name": "St. Louis Fed Financial Stress Index"},

    # Chicago Fed National Activity Index — coincident, monthly, with a
    # canonical recession-signal threshold of CFNAIMA3 < -0.7.
    "cfnai":     {"fred_id": "CFNAI",    "freq": "M", "transform": "level", "name": "Chicago Fed National Activity Index"},
    "cfnai_3ma": {"fred_id": "CFNAIMA3", "freq": "M", "transform": "level", "name": "Chicago Fed Activity Index (3-month avg)"},

    # Atlanta Fed Wage Growth Tracker — 12-month MA of median wage growth.
    "wage_tracker": {"fred_id": "FRBATLWGT12MMAWMHWGO", "freq": "M", "transform": "level", "name": "Atlanta Fed Wage Growth Tracker"},
}


def fred_ids() -> list[str]:
    """Convenience: list every FRED ID we know about."""
    return [meta["fred_id"] for meta in SERIES_REGISTRY.values()]


def fred_id_for(name: str) -> str:
    return SERIES_REGISTRY[name]["fred_id"]


def label_for(name: str) -> str:
    """Plain-English name for a registry key (e.g. ``unrate`` → "Unemployment
    Rate"). Falls back to the FRED ID, then the key itself, if unmapped."""
    meta = SERIES_REGISTRY.get(name)
    if meta:
        return meta.get("name") or meta.get("fred_id", name)
    return name


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
