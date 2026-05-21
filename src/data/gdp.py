"""Real-time GDP / growth monitoring data layer.

The recession-probit engine answers "how likely is a downturn in the next
year." This module answers the *complementary* question — "how fast is the
economy growing right now" — by surfacing authoritative FRED series rather than
rebuilding an institutional nowcaster:

* the official BEA real-GDP growth print and **Atlanta Fed GDPNow** nowcast,
* real **GDI** (often catches turns GDP misses; the divergence is itself a tell),
* the BEA **contributions to percent change** (C / I / G / net exports /
  inventories) — GDPNow's signature "what's driving it" decomposition,
* the Dallas Fed **Weekly Economic Index** for a high-frequency read, and
* a light, transparent **coincident growth factor** (a z-scored composite of
  monthly hard-data series) — the one original model, kept deliberately simple.

Every fetch degrades gracefully: a missing key, dead network, or renamed FRED
ID is logged and skipped, never raised, so the tab still renders.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# (FRED id, label) groupings. Contribution/GDI mnemonics follow BEA's NIPA
# naming; any that FRED rejects are skipped, so the panel self-heals.
HEADLINE_SERIES: list[tuple[str, str]] = [
    ("A191RL1Q225SBEA", "Real GDP"),
    ("GDPNOW", "GDPNow (Atlanta Fed)"),
    ("A261RL1Q225SBEA", "Real GDI"),
]

CONTRIBUTION_SERIES: list[tuple[str, str]] = [
    ("DPCERY2Q224SBEA", "Consumption"),
    ("A006RY2Q224SBEA", "Investment"),
    ("A822RY2Q224SBEA", "Government"),
    ("A019RY2Q224SBEA", "Net exports"),
    ("A014RY2Q224SBEA", "Inventories"),
]

HIGHFREQ_SERIES: list[tuple[str, str]] = [
    ("WEI", "Weekly Economic Index"),
]

# Monthly hard-data inputs to the coincident factor. All YoY-transformed,
# z-scored, then averaged. Kept to four broad, low-overlap activity series.
COINCIDENT_INPUTS: list[tuple[str, str]] = [
    ("PAYEMS", "Payrolls"),
    ("INDPRO", "Industrial production"),
    ("RSAFS", "Retail sales"),
    ("PCEC96", "Real consumption"),
]

LABELS: dict[str, str] = {
    sid: label
    for sid, label in HEADLINE_SERIES + CONTRIBUTION_SERIES + HIGHFREQ_SERIES + COINCIDENT_INPUTS
}


def _cache_data(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        return st.cache_data(*args, **kwargs)
    except Exception:
        def _passthrough(fn):
            return fn

        return _passthrough


def _fetch_group(ids: list[tuple[str, str]], start: str, log: list[str]) -> dict[str, pd.Series]:
    """Fetch each (id, label); skip and log any that fail."""
    from src.data.fred_client import fetch_series

    out: dict[str, pd.Series] = {}
    for sid, _label in ids:
        try:
            s = pd.Series(fetch_series(sid, start)).dropna()
        except Exception as exc:  # noqa: BLE001 - no key / network / bad id → skip
            log.append(f"{sid}: {type(exc).__name__}")
            continue
        if not s.empty:
            s.index = pd.DatetimeIndex(s.index)
            out[sid] = s.sort_index()
    return out


@_cache_data(ttl=21600, show_spinner=False)  # 6 hours
def fetch_gdp_bundle(start: str = "1990-01-01") -> dict:
    """Headline growth, nowcast, GDI, component contributions, and the WEI.

    Returns ``{"headline", "contributions", "highfreq", "log"}`` where each
    group maps FRED id → Series. Empty groups mean the source was unreachable
    (e.g. no ``FRED_API_KEY``); callers should render a fallback, not crash.
    """
    log: list[str] = []
    return {
        "headline": _fetch_group(HEADLINE_SERIES, start, log),
        "contributions": _fetch_group(CONTRIBUTION_SERIES, start, log),
        "highfreq": _fetch_group(HIGHFREQ_SERIES, start, log),
        "log": log,
    }


def latest(series: pd.Series | None) -> tuple[float, pd.Timestamp | None]:
    """Last finite value and its date, or (nan, None)."""
    if series is None or series.empty:
        return float("nan"), None
    s = series.dropna()
    if s.empty:
        return float("nan"), None
    return float(s.iloc[-1]), s.index[-1]


@_cache_data(ttl=21600, show_spinner=False)
def coincident_factor(start: str = "1960-01-01") -> dict:
    """A standardized coincident growth factor from monthly hard data.

    Each input is taken year-over-year, z-scored over the common sample, and
    the available z-scores are averaged per month. The result is a unitless
    "growth momentum" series (0 = at trend, positive = above trend), not a GDP
    percentage — deliberately simple and fully interpretable.

    Returns ``{"composite", "components", "log"}``; ``composite`` is empty when
    no inputs could be fetched.
    """
    log: list[str] = []
    raw = _fetch_group(COINCIDENT_INPUTS, start, log)
    zframes: dict[str, pd.Series] = {}
    for sid, s in raw.items():
        monthly = s.resample("MS").last()
        yoy = monthly.pct_change(12) * 100.0
        yoy = yoy.dropna()
        if yoy.std(ddof=0) == 0 or yoy.empty:
            continue
        zframes[LABELS.get(sid, sid)] = (yoy - yoy.mean()) / yoy.std(ddof=0)

    if not zframes:
        return {"composite": pd.Series(dtype=float), "components": pd.DataFrame(), "log": log}

    components = pd.concat(zframes, axis=1).sort_index()
    composite = components.mean(axis=1, skipna=True).rename("coincident")
    return {"composite": composite, "components": components, "log": log}


# ----------------------------------------------------------- analysis helpers


def _normalize_quarter_start(s: pd.Series) -> pd.Series:
    """Snap a quarterly series' index to quarter-start timestamps for alignment."""
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index).to_period("Q").to_timestamp(how="start")
    return out


def factor_gdp_frame(composite: pd.Series, gdp: pd.Series) -> pd.DataFrame:
    """Align the (monthly) coincident factor with quarterly GDP growth.

    The factor is averaged within each quarter; both are snapped to quarter
    starts. Returns columns ``factor`` and ``gdp`` (empty if either is missing).
    """
    if composite is None or composite.empty or gdp is None or gdp.empty:
        return pd.DataFrame(columns=["factor", "gdp"])
    q = composite.resample("QS").mean().rename("factor")
    g = _normalize_quarter_start(gdp.dropna()).rename("gdp")
    return pd.concat([q, g], axis=1).dropna()


def factor_prob_frame(composite: pd.Series, prob: pd.Series) -> pd.DataFrame:
    """Align the coincident factor with the recession-probability history.

    Both are monthly. Returns columns ``factor`` and ``prob`` (0–1), empty if
    either is missing.
    """
    if composite is None or composite.empty or prob is None or prob.empty:
        return pd.DataFrame(columns=["factor", "prob"])
    return pd.concat([composite.rename("factor"), prob.rename("prob")], axis=1).dropna()


def pearson(frame: pd.DataFrame, a: str, b: str) -> float:
    """Pearson correlation between two columns, or NaN if too few points."""
    if frame is None or len(frame) < 3 or a not in frame or b not in frame:
        return float("nan")
    return float(frame[a].corr(frame[b]))

