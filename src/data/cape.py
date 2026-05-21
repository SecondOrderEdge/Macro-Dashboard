"""CAPE ratio (Shiller cyclically-adjusted P/E) from Shiller's published dataset.

CAPE is intentionally NOT an input to the recession ensemble — empirically
it doesn't help short-horizon recession prediction (it ran hot for most of
2014-2024 without a recession arriving). It's exposed here as a *valuation*
context indicator: at extreme readings, the market downside conditional on
a recession arriving is materially larger than at average readings.

Data path:

1. **Primary**: GitHub-hosted CSV mirror at
   ``raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv``.
   This is the ``datahub.io/core/s-and-p-500`` repo, which curates Shiller's
   historical series as a flat CSV. The ``PE10`` column is the CAPE ratio.
   We use it as the primary source because GitHub Raw is reliable and
   doesn't require any binary spreadsheet parser.

2. **Fallback**: Robert Shiller's original Excel at Yale. Requires
   ``openpyxl`` (for .xlsx) or ``xlrd<2.0`` (for .xls); served at
   ``econ.yale.edu/~shiller/data/`` and at ``shillerdata.com``.

Either source returns the same monthly CAPE series. Any failure returns an
empty Series so the dashboard degrades gracefully rather than crashing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

_BUNDLED_PATH = Path(__file__).resolve().parents[2] / "data" / "cape.csv"


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

_GITHUB_CSV = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv"
)

_YALE_URLS = (
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


@_cache_data(ttl=604800, show_spinner=False)  # 7 days
def fetch_cape_history() -> pd.Series:
    """Return a monthly-indexed CAPE Series, or an empty one on total failure.

    Sources are overlaid in increasing order of freshness so the most recent
    available value wins for each month:

    1. **datahub GitHub CSV** — long history, but the mirror is unmaintained
       (it has lagged by years), so it only fills old gaps.
    2. **Bundled ``data/cape.csv``** — refreshed monthly by the refresh-cape
       GitHub Action; this is the reliable, current base in deployment.
    3. **Shiller's Excel (live)** — freshest if reachable, but the source blocks
       bots in many environments, so it's best-effort on top.

    Errors during fetching/parsing are recorded on the returned Series via
    ``attrs['fetch_log']`` so the UI can surface them.
    """
    log: list[str] = []
    github_series = _try_fetch_github(log)
    bundled_series = _load_bundled(log)
    yale_series = _try_fetch_yale(log)

    merged = _merge([github_series, bundled_series, yale_series])
    if merged.empty:
        return _empty(log)
    merged.attrs["fetch_log"] = log
    return merged


def _merge(series_list: list[pd.Series]) -> pd.Series:
    """Overlay CAPE series in order; later (fresher) sources win per month."""
    out = pd.Series(dtype=float, name="cape")
    sources: list[str] = []
    for s in series_list:
        if s is None or s.empty:
            continue
        for ts, val in s.items():
            out.loc[ts] = val
        sources.append(s.attrs.get("source", "?"))
    out = out.sort_index()
    out.attrs["source"] = " + ".join(sources) if sources else "none"
    return out


def _load_bundled(log: list[str]) -> pd.Series:
    """Read the committed ``data/cape.csv`` (date,cape), refreshed by the Action."""
    try:
        if not _BUNDLED_PATH.exists():
            return pd.Series(dtype=float, name="cape")
        df = pd.read_csv(_BUNDLED_PATH)
        date_col = "date" if "date" in df.columns else df.columns[0]
        val_col = "cape" if "cape" in df.columns else df.columns[-1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col, val_col])
        s = pd.Series(df[val_col].astype(float).values, index=df[date_col], name="cape").sort_index()
        s.attrs["source"] = "bundled:data/cape.csv"
        return s
    except Exception as exc:  # noqa: BLE001
        log.append(f"bundled cape.csv: {type(exc).__name__}: {exc}")
        return pd.Series(dtype=float, name="cape")


def _try_fetch_yale(log: list[str]) -> pd.Series:
    try:
        import requests
    except ImportError:
        log.append("requests not installed")
        return pd.Series(dtype=float, name="cape")

    for url in _YALE_URLS:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30)
        except Exception as exc:
            log.append(f"yale {url}: {type(exc).__name__}: {exc}")
            continue
        if r.status_code != 200:
            log.append(f"yale {url}: HTTP {r.status_code}")
            continue
        for engine in ("openpyxl", "xlrd"):
            try:
                df = pd.read_excel(
                    io.BytesIO(r.content),
                    sheet_name="Data",
                    skiprows=7,
                    engine=engine,
                )
            except Exception as exc:
                log.append(f"yale {url} via {engine}: {type(exc).__name__}: {exc}")
                continue
            s = _parse_shiller_excel(df)
            if not s.empty:
                s.attrs["source"] = f"yale-excel ({engine})"
                return s
            log.append(f"yale {url} via {engine}: parsed but empty")
    return pd.Series(dtype=float, name="cape")


def _try_fetch_github(log: list[str]) -> pd.Series:
    try:
        return _fetch_github_csv(log)
    except Exception as exc:
        log.append(f"github csv: {type(exc).__name__}: {exc}")
        return pd.Series(dtype=float, name="cape")


def _empty(log: list[str]) -> pd.Series:
    s = pd.Series(dtype=float, name="cape")
    s.attrs["fetch_log"] = log
    return s


# ---------------------------------------------------------------- GitHub CSV


def _fetch_github_csv(log: list[str]) -> pd.Series:
    import requests

    r = requests.get(_GITHUB_CSV, headers=_HEADERS, timeout=30)
    if r.status_code != 200:
        log.append(f"{_GITHUB_CSV}: HTTP {r.status_code}")
        return pd.Series(dtype=float, name="cape")

    df = pd.read_csv(io.BytesIO(r.content))
    if "Date" not in df.columns or "PE10" not in df.columns:
        log.append(f"github csv: missing columns; got {list(df.columns)[:6]}")
        return pd.Series(dtype=float, name="cape")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "PE10"])
    df["PE10"] = pd.to_numeric(df["PE10"], errors="coerce")
    df = df.dropna(subset=["PE10"])
    # PE10 == 0 is the missing-data placeholder used for the first 10 years
    # of the sample (before there's enough history to compute it) and
    # sometimes for the most-recent months that haven't accumulated yet.
    df = df[df["PE10"] > 0]
    s = pd.Series(df["PE10"].astype(float).values, index=df["Date"], name="cape")
    s = s.sort_index()
    s.attrs["source"] = "github:datasets/s-and-p-500"
    return s


# ---------------------------------------------------------------- Yale Excel


def _parse_shiller_excel(df: pd.DataFrame) -> pd.Series:
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
    df[cape_col] = pd.to_numeric(df[cape_col], errors="coerce")
    df = df.dropna(subset=[cape_col])
    s = pd.Series(df[cape_col].astype(float).values, index=df["date"], name="cape")
    return s.sort_index()


def _yale_date_to_timestamp(v) -> pd.Timestamp | None:
    """Yale's date column: float like ``2024.04`` (Apr 2024) or ``2024.1``
    (Oct 2024 — digits after the decimal are positional, not numeric)."""
    if pd.isna(v):
        return None
    try:
        s = f"{float(v):.2f}"
    except Exception:
        return None
    if "." not in s:
        return None
    year_str, frac = s.split(".")
    try:
        year = int(year_str)
        month = int(frac.ljust(2, "0")[:2])
    except Exception:
        return None
    if month < 1 or month > 12 or year < 1871 or year > 2100:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


# ---------------------------------------------------------------- summary


def cape_summary(cape: pd.Series, modern_start: str = "1950-01-01") -> dict:
    """Current value, percentile rank, median, recent trend, classic peaks."""
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

    def _peak(start: str, end: str) -> float:
        window = cape.loc[start:end]
        return float(window.max()) if not window.empty else float("nan")

    peaks = {
        "1929 peak": _peak("1929-01-01", "1929-12-31"),
        "2000 dot-com peak": _peak("1999-01-01", "2000-12-31"),
        "2007 peak": _peak("2007-01-01", "2007-12-31"),
    }

    return {
        "as_of": as_of,
        "today": today,
        "modern_percentile": pct,
        "modern_median": median,
        "one_year_ago": yr_ago,
        "modern_start": pd.Timestamp(modern_start),
        "peaks": peaks,
        "source": cape.attrs.get("source", "unknown"),
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
