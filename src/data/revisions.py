"""Revision awareness via ALFRED first-release vs final-revised data.

Official statistics are revised — sometimes enough to rewrite the narrative
(e.g. payroll benchmark revisions flipping reported growth into contraction).
Backtesting on final-revised data is look-ahead bias. This module surfaces the
*size* of those revisions by comparing each series' first public print to its
latest value, so the dashboard can show how much initial estimates move.

The full point-in-time backtest (re-fitting on as-of-date vintages) is a batch
pipeline, out of scope for the live app; this is the contained, visual slice.
The revision math is pure and unit-tested; the ALFRED fetch is isolated and
degrades gracefully when the network/key is unavailable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# High-revision series worth showing (FRED IDs + plain-English labels + unit).
REVISION_SERIES: list[tuple[str, str, str]] = [
    ("PAYEMS", "Nonfarm Payrolls", "thousands"),
    ("GDPC1", "Real GDP", "bn chained $"),
    ("RSAFS", "Retail Sales", "$mn"),
]


def revision_summary(first: pd.Series, latest: pd.Series) -> dict:
    """Compare a first-release series to its final-revised counterpart.

    Returns revision stats plus the aligned frame and revision series for
    charting. ``revision = latest - first`` per observation date.
    """
    empty = {
        "n": 0, "mean_abs_revision": float("nan"), "median_revision": float("nan"),
        "share_revised_down": float("nan"), "last_first": float("nan"),
        "last_latest": float("nan"), "revision": pd.Series(dtype=float),
        "aligned": pd.DataFrame(),
    }
    if first is None or latest is None:
        return empty
    df = pd.concat([first.rename("first"), latest.rename("latest")], axis=1, sort=True).dropna()
    if df.empty:
        return empty
    rev = (df["latest"] - df["first"]).rename("revision")
    return {
        "n": int(len(df)),
        "mean_abs_revision": float(rev.abs().mean()),
        "median_revision": float(rev.median()),
        "share_revised_down": float((rev < 0).mean() * 100.0),
        "last_first": float(df["first"].iloc[-1]),
        "last_latest": float(df["latest"].iloc[-1]),
        "revision": rev,
        "aligned": df,
    }


def _cache_data(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        return st.cache_data(*args, **kwargs)
    except Exception:
        def _passthrough(fn):
            return fn

        return _passthrough


@_cache_data(ttl=86400, show_spinner=False)  # 1 day
def fetch_revision_pair(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    """First-release vs latest-revised values for ``series_id`` via ALFRED.

    Returns a DataFrame indexed by observation date with ``first`` and
    ``latest`` columns, or an empty frame on any failure.
    """
    try:
        from src.data.fred_client import _get_client

        fred = _get_client()
        first = fred.get_series_first_release(series_id)
        latest = fred.get_series_latest_release(series_id)
        first = pd.Series(first).dropna()
        latest = pd.Series(latest).dropna()
        first.index = pd.DatetimeIndex(first.index)
        latest.index = pd.DatetimeIndex(latest.index)
        df = pd.concat(
            [first.rename("first"), latest.rename("latest")], axis=1, sort=True
        )
        return df.loc[df.index >= pd.Timestamp(start)]
    except Exception:  # noqa: BLE001 - no key / no network / API change → graceful
        return pd.DataFrame(columns=["first", "latest"])
