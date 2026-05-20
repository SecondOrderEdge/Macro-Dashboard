"""CAPE ratio (Shiller cyclically-adjusted P/E) from Yale's published dataset.

CAPE is intentionally NOT an input to the recession ensemble — empirically
it doesn't help short-horizon recession prediction (it ran hot for most of
2014-2024 without a recession arriving). It's exposed here as a
*valuation* context indicator: at extreme readings, the market downside
conditional on a recession arriving is materially larger than at average
readings.

Data is fetched live from Robert Shiller's published spreadsheet at
econ.yale.edu/~shiller/data/. The fetcher is fault-tolerant: any failure
returns an empty Series so the dashboard degrades gracefully rather than
crashing.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


YALE_URLS = (
    "https://shillerdata.com/wp-content/uploads/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xlsx",
)


def _cache_data(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        return st.cache_data(*args, **kwargs)
    except Exception:
        def _passthrough(fn):
            return fn

        return _passthrough


@_cache_data(ttl=604800, show_spinner=False)  # 7 days — CAPE updates monthly
def fetch_cape_history() -> pd.Series:
    """Return a monthly-indexed Series of CAPE values, or empty on failure."""
    try:
        import requests
    except ImportError:
        return pd.Series(dtype=float, name="cape")

    for url in YALE_URLS:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception:
            continue
        for engine in ("openpyxl", "xlrd"):
            try:
                df = pd.read_excel(
                    io.BytesIO(r.content),
                    sheet_name="Data",
                    skiprows=7,
                    engine=engine,
                )
                parsed = _parse_shiller(df)
                if not parsed.empty:
                    return parsed
            except Exception:
                continue
    return pd.Series(dtype=float, name="cape")


def _parse_shiller(df: pd.DataFrame) -> pd.Series:
    """Pull the CAPE column out of Shiller's spreadsheet.

    Yale's first column is a date in float form: ``2024.04`` means April 2024,
    ``2024.10`` means October 2024 (the digits after the decimal are the
    month). The CAPE column is usually labelled ``CAPE`` but Shiller has
    renamed it in some editions to ``Cyclically Adjusted P/E Ratio P/E10``
    or similar — match by substring.
    """
    if df.empty:
        return pd.Series(dtype=float, name="cape")
    date_col = df.columns[0]
    cape_col = None
    for c in df.columns:
        cu = str(c).upper().replace(" ", "")
        if "CAPE" in cu or "P/E10" in cu or "PE10" in cu or "CYCLICALLYADJUSTED" in cu:
            cape_col = c
            break
    if cape_col is None:
        return pd.Series(dtype=float, name="cape")

    df = df[[date_col, cape_col]].copy()
    df = df.dropna(subset=[date_col, cape_col])
    df["date"] = df[date_col].apply(_yale_date_to_timestamp)
    df = df.dropna(subset=["date"])
    # CAPE column may contain header text rows; coerce and drop non-numeric.
    df[cape_col] = pd.to_numeric(df[cape_col], errors="coerce")
    df = df.dropna(subset=[cape_col])
    s = pd.Series(df[cape_col].astype(float).values, index=df["date"], name="cape")
    return s.sort_index()


def _yale_date_to_timestamp(v) -> pd.Timestamp | None:
    """Yale's date column: float like ``2024.04`` (Apr 2024) or ``2024.1``
    (Oct 2024 — the digit after the decimal is *positional*, not numeric)."""
    if pd.isna(v):
        return None
    try:
        s = f"{float(v):.2f}"  # forces two decimal digits
    except Exception:
        return None
    if "." not in s:
        return None
    year_str, frac = s.split(".")
    try:
        year = int(year_str)
        # '04' means April, '10' means October.
        month = int(frac.ljust(2, "0")[:2])
    except Exception:
        return None
    if month < 1 or month > 12:
        return None
    if year < 1871 or year > 2100:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def cape_summary(cape: pd.Series, modern_start: str = "1950-01-01") -> dict:
    """Return current value, percentile rank, median, and recent trend.

    The percentile rank is computed against the *modern* history (default
    post-1950) because the pre-WWII sample contains structural breaks
    (gold standard, very different reporting cadence) that make full-history
    percentiles misleading.
    """
    if cape is None or cape.empty:
        return {}
    cape = cape.dropna().sort_index()
    today = float(cape.iloc[-1])
    as_of = cape.index[-1]

    modern = cape.loc[cape.index >= pd.Timestamp(modern_start)]
    if modern.empty:
        modern = cape
    pct = float((modern <= today).mean() * 100.0)
    median = float(modern.median())

    one_yr_ago_idx = as_of - pd.DateOffset(months=12)
    s_to_year_ago = cape.loc[cape.index <= one_yr_ago_idx]
    yr_ago = float(s_to_year_ago.iloc[-1]) if not s_to_year_ago.empty else float("nan")

    # Sample peaks
    peaks = {
        "1929 peak": float(cape.loc["1929-01-01":"1929-12-31"].max()) if "1929-09-01" in cape.index or any(cape.loc["1929":"1929"].notna()) else float("nan"),
        "2000 dot-com peak": float(cape.loc["1999-01-01":"2000-12-31"].max()) if not cape.loc["1999-01-01":"2000-12-31"].empty else float("nan"),
        "2007 peak": float(cape.loc["2007-01-01":"2007-12-31"].max()) if not cape.loc["2007-01-01":"2007-12-31"].empty else float("nan"),
    }

    return {
        "as_of": as_of,
        "today": today,
        "modern_percentile": pct,
        "modern_median": median,
        "one_year_ago": yr_ago,
        "modern_start": pd.Timestamp(modern_start),
        "peaks": peaks,
    }


def cape_band(percentile: float) -> tuple[str, str]:
    """Map a percentile rank to a label + severity bucket."""
    if percentile < 25:
        return "CHEAP", "low"
    if percentile < 60:
        return "AVERAGE", "elevated"
    if percentile < 85:
        return "EXPENSIVE", "high"
    return "EXTREME", "critical"
