"""Yield curve analytics: term structure, spreads, inversion statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Maturity:
    label: str
    years: float
    series: str


_MATURITY_TABLE: list[_Maturity] = [
    _Maturity("1M", 1 / 12, "dgs1mo"),
    _Maturity("3M", 0.25, "dgs3mo"),
    _Maturity("6M", 0.5, "dgs6mo"),
    _Maturity("1Y", 1.0, "dgs1"),
    _Maturity("2Y", 2.0, "dgs2"),
    _Maturity("3Y", 3.0, "dgs3"),
    _Maturity("5Y", 5.0, "dgs5"),
    _Maturity("7Y", 7.0, "dgs7"),
    _Maturity("10Y", 10.0, "dgs10"),
    _Maturity("20Y", 20.0, "dgs20"),
    _Maturity("30Y", 30.0, "dgs30"),
]


class YieldCurve:
    """Operates on a daily panel of Treasury yields.

    The panel must contain columns matching the FRED series IDs used in the
    registry (e.g. `DGS10`, `DGS3MO`). Missing maturities are tolerated — the
    term structure simply omits them.
    """

    MATURITIES = [m.label for m in _MATURITY_TABLE]
    SERIES_MAP = {m.label: m.series for m in _MATURITY_TABLE}

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel.copy()
        self.panel.index = pd.DatetimeIndex(self.panel.index)
        self.panel = self.panel.sort_index()

    def _col_for(self, fred_id: str) -> str | None:
        if fred_id in self.panel.columns:
            return fred_id
        upper = fred_id.upper()
        if upper in self.panel.columns:
            return upper
        return None

    def _series(self, name: str) -> pd.Series | None:
        """Return the yield series for a registry key (e.g. 'dgs10')."""
        fred_id = {m.series: m.label for m in _MATURITY_TABLE}  # not used
        # Map registry key -> FRED id by uppercasing.
        col = self._col_for(name.upper())
        if col is None:
            return None
        return self.panel[col].dropna()

    def term_structure(
        self,
        as_of: pd.Timestamp | None = None,
        comparison_offsets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Snapshot the curve today vs. earlier dates.

        Returns a tidy DataFrame with columns ``maturity, years, current,
        m3, m12`` (one column per comparison offset).
        """
        comparison_offsets = comparison_offsets or ["3M", "12M"]
        idx = self.panel.dropna(how="all").index
        if len(idx) == 0:
            return pd.DataFrame(columns=["maturity", "years", "current"])

        as_of = pd.Timestamp(as_of) if as_of is not None else idx.max()
        offset_map = {"3M": pd.DateOffset(months=3), "12M": pd.DateOffset(months=12)}

        rows = []
        for mat in _MATURITY_TABLE:
            series = self._series(mat.series)
            if series is None or series.empty:
                continue
            row = {"maturity": mat.label, "years": mat.years}
            row["current"] = _asof(series, as_of)
            for off_label in comparison_offsets:
                delta = offset_map.get(off_label, pd.DateOffset(months=int(off_label.rstrip("M"))))
                key = f"m{off_label.rstrip('M')}"
                row[key] = _asof(series, as_of - delta)
            rows.append(row)

        return pd.DataFrame(rows)

    def spreads_history(self) -> pd.DataFrame:
        """Common spread history: 10Y-3M, 10Y-2Y, 5Y-2Y."""
        dgs10 = self._col_for("DGS10")
        dgs2 = self._col_for("DGS2")
        dgs5 = self._col_for("DGS5")
        dgs3mo = self._col_for("DGS3MO")

        out = pd.DataFrame(index=self.panel.index)
        if dgs10 and dgs3mo:
            out["spread_10y3m"] = self.panel[dgs10] - self.panel[dgs3mo]
        if dgs10 and dgs2:
            out["spread_10y2y"] = self.panel[dgs10] - self.panel[dgs2]
        if dgs5 and dgs2:
            out["spread_5y2y"] = self.panel[dgs5] - self.panel[dgs2]
        return out.dropna(how="all")

    def inversion_stats(self, nber: pd.Series) -> dict:
        """Inversion-driven recession statistics, computed off the 10Y-3M spread.

        - months_inverted: current consecutive months below zero (monthly avg)
        - max_depth_current: most-negative monthly average in the current run
        - avg_lead_to_recession: average months from inversion *start* to NBER peak,
          for inversions lasting >3 months
        - hit_rate: (recessions following inversion, total inversions)
        """
        spreads = self.spreads_history()
        if "spread_10y3m" not in spreads.columns:
            return {
                "months_inverted": 0,
                "max_depth_current": float("nan"),
                "avg_lead_to_recession": float("nan"),
                "hit_rate": (0, 0),
            }

        monthly = spreads["spread_10y3m"].resample("ME").mean().dropna()
        inverted = monthly < 0

        # Current consecutive run of inverted months at the tail.
        months_inverted = 0
        max_depth_current = float("nan")
        if not inverted.empty and bool(inverted.iloc[-1]):
            run = 0
            depths = []
            for ts in reversed(monthly.index):
                if monthly.loc[ts] < 0:
                    run += 1
                    depths.append(monthly.loc[ts])
                else:
                    break
            months_inverted = run
            max_depth_current = float(min(depths)) if depths else float("nan")

        # Episodes lasting >3 months.
        episodes = _runs(inverted)
        episodes = [ep for ep in episodes if ep[1] - ep[0] + 1 > 3]

        nber_monthly = nber.copy()
        nber_monthly.index = pd.DatetimeIndex(nber_monthly.index).to_period("M").to_timestamp()
        peaks = _nber_peaks(nber_monthly)

        hits = 0
        leads = []
        for start_idx, _end_idx in episodes:
            start_date = monthly.index[start_idx]
            # Look for the next NBER peak within 36 months.
            future_peaks = [p for p in peaks if 0 <= (p - start_date).days / 30.5 <= 36]
            if future_peaks:
                hits += 1
                lead = (future_peaks[0] - start_date).days / 30.5
                leads.append(lead)

        avg_lead = float(np.mean(leads)) if leads else float("nan")
        return {
            "months_inverted": int(months_inverted),
            "max_depth_current": max_depth_current,
            "avg_lead_to_recession": avg_lead,
            "hit_rate": (hits, len(episodes)),
        }


def _asof(series: pd.Series, ts: pd.Timestamp) -> float:
    """Most recent observation on or before ``ts``; NaN if none."""
    try:
        s = series.loc[:ts]
    except KeyError:
        return float("nan")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])


def _runs(flags: pd.Series) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx), ...] for True runs in a boolean series."""
    out: list[tuple[int, int]] = []
    arr = flags.values
    i = 0
    n = len(arr)
    while i < n:
        if arr[i]:
            j = i
            while j + 1 < n and arr[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _nber_peaks(nber_monthly: pd.Series) -> list[pd.Timestamp]:
    """Identify NBER peak months (transitions from False to True)."""
    peaks = []
    prev = False
    for ts, val in nber_monthly.items():
        if val and not prev:
            peaks.append(ts)
        prev = bool(val)
    return peaks
