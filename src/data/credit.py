"""Credit & funding-stress data layer (with a CLO-supply gauge).

A leading-credit-conditions lens for the recession framework. Credit tightens
before the real economy turns, so a standardized stress composite — high-yield
and investment-grade spreads, the Chicago/St. Louis conditions indices, and the
SLOOS bank-tightening survey — is a natural upstream signal.

Scope is deliberately honest: this is *systemic credit & funding stress*, not
tranche-level CLO analytics. The CLO piece is the Fed's quarterly Z.1 supply
estimate (is the CLO machine expanding or contracting) — coarse and lagged ~10
weeks, never timely demand/default data, which lives behind paid vendors.

Every fetch degrades gracefully: a missing key, dead network, or renamed FRED
ID is logged and skipped, never raised.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# (FRED id, label, sign): sign = +1 means a higher value is MORE stress, so the
# z-scored series can be averaged directly into a "higher = more stress" index.
STRESS_SERIES: list[tuple[str, str, int]] = [
    ("BAMLH0A0HYM2", "High-yield OAS", +1),
    ("BAMLC0A0CM", "Investment-grade OAS", +1),
    ("BAA10Y", "Baa–10y spread", +1),
    ("NFCI", "Financial conditions (NFCI)", +1),
    ("STLFSI4", "Financial stress (STLFSI)", +1),
    ("DRTSCILM", "Banks tightening C&I", +1),
]

# Funding / liquidity context (shown alongside, not folded into the composite —
# their "stress" sign is regime-dependent, e.g. balance-sheet contraction).
LIQUIDITY_SERIES: list[tuple[str, str]] = [
    ("SOFR", "SOFR (overnight)"),
    ("WALCL", "Fed balance sheet"),
    ("WRESBAL", "Bank reserves"),
    ("RRPONTSYD", "Reverse repo (ON RRP)"),
    ("M2SL", "M2 money stock"),
]

# CLO supply, Fed Z.1 Financial Accounts (quarterly, ~10-week lag). The RoW
# market-value liability is where most US CLOs sit (Cayman-domiciled). The
# asset-side "loans held by CLOs" id is unconfirmed, so it's best-effort.
CLO_SERIES: list[tuple[str, str]] = [
    ("BOGZ1LM263163063Q", "CLO liabilities outstanding"),
    ("BOGZ1FL673069503Q", "Leveraged loans held by CLOs"),  # best-effort; skipped if FRED rejects
]

def _cache_data(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        return st.cache_data(*args, **kwargs)
    except Exception:
        def _passthrough(fn):
            return fn

        return _passthrough


def _fetch_group(ids: list[str], start: str, log: list[str]) -> dict[str, pd.Series]:
    """Fetch each id; skip and log any that fail (no key / network / bad id)."""
    from src.data.fred_client import fetch_series

    out: dict[str, pd.Series] = {}
    for sid in ids:
        try:
            s = pd.Series(fetch_series(sid, start)).dropna()
        except Exception as exc:  # noqa: BLE001
            log.append(f"{sid}: {type(exc).__name__}")
            continue
        if not s.empty:
            s.index = pd.DatetimeIndex(s.index)
            out[sid] = s.sort_index()
    return out


@_cache_data(ttl=21600, show_spinner=False)  # 6 hours
def credit_stress(start: str = "1997-01-01") -> dict:
    """Standardized credit-stress composite (higher = more stress).

    Each input is resampled to month-start, oriented so higher = more stress,
    z-scored over the common sample, and the available z-scores are averaged.
    Returns ``{"composite", "components", "log"}`` (empty composite on failure).
    """
    log: list[str] = []
    raw = _fetch_group([s for s, _, _ in STRESS_SERIES], start, log)
    monthly: dict[str, pd.Series] = {}
    for sid, label, sign in STRESS_SERIES:
        s = raw.get(sid)
        if s is None:
            continue
        monthly[label] = s.resample("MS").last() * sign

    if not monthly:
        return {"composite": pd.Series(dtype=float), "components": pd.DataFrame(), "log": log}

    df = pd.concat(monthly, axis=1).sort_index().ffill(limit=2)  # bridge quarterly SLOOS
    std = df.std(ddof=0).replace(0, float("nan"))
    z = (df - df.mean()) / std
    composite = z.mean(axis=1, skipna=True).rename("credit_stress")
    return {"composite": composite, "components": z, "log": log}


@_cache_data(ttl=21600, show_spinner=False)
def fetch_liquidity(start: str = "2000-01-01") -> dict[str, pd.Series]:
    """Funding/liquidity context series (id → series); failures skipped."""
    return _fetch_group([s for s, _ in LIQUIDITY_SERIES], start, [])


@_cache_data(ttl=21600, show_spinner=False)
def fetch_clo(start: str = "2000-01-01") -> dict[str, pd.Series]:
    """CLO supply series from the Z.1 Financial Accounts (id → series)."""
    return _fetch_group([s for s, _ in CLO_SERIES], start, [])


def latest(series: pd.Series | None) -> tuple[float, pd.Timestamp | None]:
    """Last finite value and its date, or (nan, None)."""
    if series is None or series.empty:
        return float("nan"), None
    s = series.dropna()
    if s.empty:
        return float("nan"), None
    return float(s.iloc[-1]), s.index[-1]


def yoy(series: pd.Series | None) -> pd.Series:
    """Year-over-year percent change (handles monthly or quarterly indexes)."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = series.dropna()
    if isinstance(s.index, pd.DatetimeIndex):
        inferred = pd.infer_freq(s.index)
        periods = 4 if (inferred or "").startswith("Q") else 12
    else:
        periods = 12
    return (s.pct_change(periods) * 100.0).dropna()


def align_corr(a: pd.Series, b: pd.Series) -> tuple[pd.DataFrame, float]:
    """Align two series on their common dates; return the frame + Pearson r."""
    if a is None or b is None or a.empty or b.empty:
        return pd.DataFrame(columns=["a", "b"]), float("nan")
    frame = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    r = float(frame["a"].corr(frame["b"])) if len(frame) >= 3 else float("nan")
    return frame, r
