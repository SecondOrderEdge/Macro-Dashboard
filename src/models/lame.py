"""LAME — Labor Aggregate Market Engine.

Z-score composite of 10 labor indicators, inverse-volatility weighted.
Positive composite = expansionary; negative = contractionary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.series_registry import SERIES_REGISTRY, transform_series


class LAME:
    INDICATORS: list[str] = [
        "unrate",
        "icsa",
        "ccsa",
        "jtsjol",
        "jtsqur",
        "awhaetp",
        "temphelps",
        "payems",
        "u6rate",
        "civpart",
    ]

    MIN_WINDOW = 60  # months of history required before z-scoring begins
    VOL_WINDOW = 60  # 5-year rolling volatility window

    def __init__(self):
        self._composite: pd.DataFrame | None = None
        self._weights: pd.DataFrame | None = None
        self._zscores: pd.DataFrame | None = None
        self._values: pd.DataFrame | None = None

    # ----------------------------------------------------------------- compute

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Build the LAME composite from a raw FRED panel.

        ``panel`` is expected to be indexed by date with columns matching
        FRED series IDs (e.g. ``UNRATE``). Missing indicators are tolerated
        but they reduce coverage in months where they are missing.
        """
        monthly_values = self._prepare_monthly_values(panel)
        signed_z = self._signed_zscores(monthly_values)
        weights = self._inverse_vol_weights(signed_z)
        composite = (signed_z * weights).sum(axis=1, min_count=1)

        out = pd.DataFrame({"composite": composite})
        for col in signed_z.columns:
            out[f"{col}_z"] = signed_z[col]

        self._values = monthly_values
        self._zscores = signed_z
        self._weights = weights
        self._composite = out
        return out

    # -------------------------------------------------------------- accessors

    def history(self) -> pd.Series:
        """Composite series, restricted to rows with adequate indicator coverage."""
        if self._composite is None:
            raise RuntimeError("Call compute(panel) before history().")
        composite = self._composite["composite"].dropna()
        if self._zscores is None or self._zscores.empty:
            return composite
        coverage = self._zscores.notna().sum(axis=1)
        threshold = max(int(coverage.max() * 0.7), 1)
        full_idx = coverage[coverage >= threshold].index
        return composite.loc[composite.index.intersection(full_idx)]

    def reference_date(self) -> pd.Timestamp | None:
        """Latest month-end where indicator coverage is at least 70% of peak.

        Slow-release monthlies (UNRATE, PAYEMS, JOLTS) trail weekly claims data
        by 2–6 weeks; snapshotting the *very* latest row often shows only ICSA
        and CCSA. The reference date is the latest jointly-observed month.
        """
        if self._zscores is None or self._zscores.empty:
            return None
        coverage = self._zscores.notna().sum(axis=1)
        if coverage.max() == 0:
            return None
        threshold = max(int(coverage.max() * 0.7), 1)
        adequate = coverage[coverage >= threshold]
        return adequate.index[-1] if not adequate.empty else None

    def current_breakdown(self) -> pd.DataFrame:
        """Snapshot at the reference date: value, z, weight, contribution by indicator."""
        if self._composite is None:
            raise RuntimeError("Call compute(panel) before current_breakdown().")
        z = self._zscores
        w = self._weights
        v = self._values
        if z is None or w is None or v is None:
            raise RuntimeError("Internal state missing — recompute the model.")

        ref = self.reference_date()
        if ref is None:
            return pd.DataFrame(columns=["name", "current_value", "z_score", "weight", "contribution"])

        z_row = z.loc[ref] if ref in z.index else pd.Series(dtype=float)
        w_row = w.loc[ref] if ref in w.index else pd.Series(dtype=float)
        v_row = v.loc[ref] if ref in v.index else pd.Series(dtype=float)

        rows = []
        for name in self.INDICATORS:
            if name not in z.columns:
                continue
            rows.append(
                {
                    "name": name,
                    "current_value": float(v_row.get(name, np.nan)),
                    "z_score": float(z_row.get(name, np.nan)),
                    "weight": float(w_row.get(name, np.nan)),
                    "contribution": float(
                        z_row.get(name, np.nan) * w_row.get(name, np.nan)
                    ),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- internals

    def _prepare_monthly_values(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Resample each indicator to month-end, then apply its registry transform."""
        cols: dict[str, pd.Series] = {}
        for name in self.INDICATORS:
            meta = SERIES_REGISTRY[name]
            fred_id = meta["fred_id"]
            if fred_id not in panel.columns:
                continue
            raw = panel[fred_id].dropna()
            if raw.empty:
                continue
            # Resample to month-end first so weekly/daily transforms are stable.
            monthly = raw.resample("ME").last()
            transformed = transform_series(monthly, meta["transform"])
            cols[name] = transformed
        if not cols:
            return pd.DataFrame()
        df = pd.concat(cols.values(), axis=1, keys=cols.keys())
        df = df.sort_index()
        return df

    def _signed_zscores(self, values: pd.DataFrame) -> pd.DataFrame:
        if values.empty:
            return values
        z = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
        for col in values.columns:
            sign = SERIES_REGISTRY[col].get("sign", 1)
            s = values[col].astype(float)
            mean = s.expanding(min_periods=self.MIN_WINDOW).mean()
            std = s.expanding(min_periods=self.MIN_WINDOW).std()
            z[col] = sign * (s - mean) / std
        return z.replace([np.inf, -np.inf], np.nan)

    def _inverse_vol_weights(self, signed_z: pd.DataFrame) -> pd.DataFrame:
        if signed_z.empty:
            return signed_z
        vol = signed_z.rolling(window=self.VOL_WINDOW, min_periods=24).std()
        inv = 1.0 / vol.replace(0.0, np.nan)
        # Mask weights where the z-score itself is missing.
        inv = inv.where(signed_z.notna())
        totals = inv.sum(axis=1, min_count=1)
        weights = inv.div(totals, axis=0)
        return weights


def _last_valid_row(df: pd.DataFrame) -> pd.Series:
    """Find the most recent row with at least one non-NaN value."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    valid = df.dropna(how="all")
    if valid.empty:
        return pd.Series(dtype=float)
    return valid.iloc[-1]
