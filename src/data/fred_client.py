"""Thin caching wrapper around fredapi.Fred.

The Streamlit `st.cache_data` decorator persists results in-process for the
TTL window, which makes the dashboard cheap to reload while staying within
FRED's rate limits. Outside Streamlit (e.g. notebooks, tests) the decorator
degrades gracefully to a plain function call.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _cache_data(*args, **kwargs):
    try:
        import streamlit as st

        return st.cache_data(*args, **kwargs)
    except Exception:
        def _passthrough(fn):
            return fn

        return _passthrough


def _get_client():
    """Build a FRED client, reading the API key from env or Streamlit secrets."""
    from fredapi import Fred

    key = os.getenv("FRED_API_KEY")
    if not key:
        try:
            import streamlit as st

            key = st.secrets.get("FRED_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not set. Add it to .env or .streamlit/secrets.toml."
        )
    return Fred(api_key=key)


@_cache_data(ttl=21600, show_spinner=False)
def fetch_series(series_id: str, start: str = "1959-01-01") -> pd.Series:
    """Fetch a single FRED series, cached for 6 hours."""
    fred = _get_client()
    try:
        s = fred.get_series(series_id, observation_start=start)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch FRED series {series_id!r}: {exc}") from exc
    if s is None or len(s) == 0:
        raise RuntimeError(f"FRED series {series_id!r} returned no observations.")
    s = pd.Series(s).copy()
    s.index = pd.DatetimeIndex(s.index)
    s.name = series_id
    return s


@_cache_data(ttl=21600, show_spinner=False)
def fetch_panel(series_ids: Iterable[str], start: str = "1959-01-01") -> pd.DataFrame:
    """Fetch many FRED series and align them into a single DataFrame.

    Series with no observations raise rather than silently being dropped — we
    want loud failures during development.
    """
    frames = []
    for sid in series_ids:
        frames.append(fetch_series(sid, start))
    df = pd.concat(frames, axis=1)
    df = df.sort_index()
    return df


def forward_fill_limited(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Forward-fill sparse monthly/weekly data, but only across short gaps.

    Useful when aligning mixed-frequency panels — we don't want a stale
    quarterly print to be carried six months forward.
    """
    return df.ffill(limit=limit)
