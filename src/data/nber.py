"""NBER recession dates loader.

`data/nber_recessions.csv` holds the official NBER cycle peaks and troughs.
This module exposes a monthly boolean series and a forward-looking dependent
variable used by the probit ensemble.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "nber_recessions.csv"


def load_nber_recessions(
    path: Path | str | None = None,
    start: str = "1950-01-01",
    end: str | None = None,
) -> pd.Series:
    """Return a monthly boolean series, True during NBER recessions.

    A month is flagged True if it falls on or after a peak and on or before
    the trough (inclusive at both ends).
    """
    path = Path(path) if path else _DEFAULT_PATH
    cycles = pd.read_csv(path, parse_dates=["peak", "trough"])

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    idx = pd.date_range(start=start, end=end_ts, freq="MS")
    flag = pd.Series(False, index=idx, name="nber_recession")

    for _, row in cycles.iterrows():
        mask = (idx >= row["peak"]) & (idx <= row["trough"])
        flag.loc[mask] = True

    return flag


def load_recession_flags(start: str = "1950-01-01") -> pd.Series:
    """Monthly recession-flag series, preferring FRED ``USREC`` over the CSV.

    ``USREC`` is the NBER-based recession indicator published live on FRED, so
    it auto-updates when the NBER dates a new cycle. If the fetch fails (no key,
    no network) we fall back to the bundled ``data/nber_recessions.csv`` so the
    dashboard still renders. The returned series matches
    :func:`load_nber_recessions`: a month-start boolean indexed from ``start``.
    """
    try:
        from src.data.fred_client import fetch_series

        usrec = fetch_series("USREC", start)
        s = pd.Series(usrec).dropna()
        if s.empty:
            raise RuntimeError("USREC returned no observations.")
        s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
        flag = (s > 0)
        flag = flag[~flag.index.duplicated(keep="last")].sort_index()
        flag.name = "nber_recession"
        return flag
    except Exception:
        return load_nber_recessions(start=start)


def recession_in_next_12m(nber: pd.Series) -> pd.Series:
    """Forward-looking dependent variable: True if a recession occurs in [t+1, t+12].

    This is the standard probit target used by the NY Fed model and Estrella–Mishkin.
    """
    out = pd.Series(False, index=nber.index, name="recession_in_next_12m")
    arr = nber.values
    n = len(arr)
    for i in range(n):
        upper = min(i + 13, n)
        if arr[i + 1 : upper].any():
            out.iloc[i] = True
    return out
