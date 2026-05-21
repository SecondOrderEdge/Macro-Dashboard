"""Market-implied signal summaries.

The "market-implied economy" — forward expectations priced into traded assets,
available continuously and (unlike official statistics) never revised. These are
pure functions over a price/yield series, so they're unit-testable offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Curated market-implied signals, all FRED daily/weekly series. Each entry:
# (fred_id, plain-English label, unit, value format).
MARKET_IMPLIED_SIGNALS: list[tuple[str, str, str, str]] = [
    ("T5YIFR",       "5y5y forward inflation",     "%",  "{:.2f}"),
    ("T10YIE",       "10y breakeven inflation",    "%",  "{:.2f}"),
    ("DFII10",       "10y real yield (TIPS)",      "%",  "{:+.2f}"),
    ("T10Y3M",       "10y–3m term spread",         "pp", "{:+.2f}"),
    ("BAMLH0A0HYM2", "High-yield credit spread",   "pp", "{:.2f}"),
    ("STLFSI4",      "Financial stress index",     "",   "{:+.2f}"),
]


def signal_summary(series: pd.Series, *, change_months: int = 1, pct_window_years: int = 5) -> dict:
    """Latest value, change over the last ``change_months``, and percentile over
    a trailing ``pct_window_years`` window — all frequency-agnostic (uses the
    series' own dates, so daily and weekly inputs are handled the same way)."""
    empty = {"latest": float("nan"), "change": float("nan"),
             "percentile": float("nan"), "as_of": None}
    if series is None:
        return empty
    s = series.dropna().sort_index()
    if s.empty:
        return empty

    as_of = s.index[-1]
    latest = float(s.iloc[-1])

    prior = s.loc[s.index <= as_of - pd.DateOffset(months=change_months)]
    change = latest - float(prior.iloc[-1]) if not prior.empty else float("nan")

    window = s.loc[s.index >= as_of - pd.DateOffset(years=pct_window_years)]
    percentile = float((window <= latest).mean() * 100.0) if not window.empty else float("nan")

    return {"latest": latest, "change": change, "percentile": percentile, "as_of": as_of}
