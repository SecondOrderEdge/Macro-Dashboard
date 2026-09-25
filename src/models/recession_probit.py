"""Probit recession-start ensemble (next 12 months) + "in recession now" nowcast panel.

Ported from the standalone Recession_Probability_Model repo that drives the
weekly investment-committee email. The headline ensemble is the equal-weighted
mean of four *methodologically distinct* specifications over a shared 37-series
FRED universe. The target is start-dated: y_t = 1 if an NBER peak falls in
t+1..t+12 (a new recession starts within the next 12 months); months already in
recession (peak month through trough) are dropped from training and scoring.

1. **NY Fed**         — probit on the 10y-3m term spread alone (Estrella-Mishkin 1998).
2. **Wright**         — probit on spread + fed funds rate (Wright 2006).
3. **BIC-selected**   — term spread + <=3 stationary, sign-restricted indicators (forward-stepwise BIC, cap 4).
4. **Estrella-Mishkin** — closed form with frozen 2006 parameters.

**Chauvet-Piger** (FRED's smoothed Markov-switching series RECPROUSM156N) and
the real-time **Sahm rule** (SAHMREALTIME) form a separate, descriptive
"in recession now" nowcast panel (:func:`nowcast_panel`). They answer a
different question from the start-dated ensemble and are never averaged into it.

On top of the ensemble it produces the analytics the email reports: a bootstrap
90% CI on the BIC model, per-indicator watchlist trigger levels (the exact value
that pushes probability to 30% / 50%), a 24-month trend attribution decomposed
into per-indicator partial effects, indicator percentiles, and a consensus read.

The modelling functions are pure (they take a raw monthly DataFrame) so they can
be unit-tested without hitting FRED; :func:`compute_probit_report` wires in the
cached FRED fetch for the live dashboard.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:  # statsmodels is optional at import time (tests import the helpers)
    import statsmodels.api as sm
except Exception:  # pragma: no cover - exercised only when dep is missing
    sm = None  # type: ignore[assignment]

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None  # type: ignore[assignment]


# --------------------------------------------------------------- configuration

OBS_START = "1967-01-01"
MIN_WINDOW = 120
MAX_FEATURES_BIC = 4          # SPREAD + at most 3 pool features (fixed a priori)
THRESHOLD_WARNING = 30
THRESHOLD_ELEVATED = 50
# "start"  = an NBER peak falls in t+1..t+12 (a new recession starts within 12
#            months); peak month..trough months are excluded (NaN) — the default.
# "window" = any recession month in t+1..t+12; "point" = recession at t+12.
TARGET_DEFINITION = "start"
TARGET_HORIZON = 12
# "In recession now" nowcast panel thresholds (descriptive; fixed a priori).
CP_SIGNAL_THRESHOLD = 50.0     # Chauvet-Piger smoothed probability, percent
SAHM_SIGNAL_THRESHOLD = 0.50   # Sahm rule (real-time), percentage points
BOOTSTRAP_ITERS = 300        # email job uses 500-1000; trimmed for dashboard latency

# FRED universe: 35 candidate features across eight macro categories. ``freq``
# drives resampling to month-start; ``transform`` is applied in feature engineering.
SERIES_CONFIG: dict[str, dict] = {
    "CFNAI":    {"name": "Chicago Fed National Activity Index",    "category": "National Activity", "transform": "level"},
    "CFNAIMA3": {"name": "CFNAI 3-Month Moving Average",           "category": "National Activity", "transform": "level"},
    "GDPC1":    {"name": "Real GDP",                               "category": "National Activity", "transform": "yoy", "freq": "Q"},
    "USSLIND":  {"name": "Leading Index for the US",               "category": "National Activity", "transform": "level"},
    "INDPRO":   {"name": "Industrial Production Index",            "category": "Industrial", "transform": "yoy"},
    "BSCICP02USM460S": {"name": "OECD Manufacturing Confidence",   "category": "Industrial", "transform": "level"},
    "TCU":      {"name": "Capacity Utilization",                   "category": "Industrial", "transform": "level"},
    "DGORDER":  {"name": "Durable Goods Orders",                   "category": "Industrial", "transform": "yoy"},
    "IPMAN":    {"name": "Industrial Production: Manufacturing",   "category": "Industrial", "transform": "yoy"},
    "UMCSENT":  {"name": "U. Michigan Consumer Sentiment",         "category": "Consumer", "transform": "level"},
    "PCECC96":  {"name": "Real Personal Consumption Expenditures", "category": "Consumer", "transform": "yoy"},
    "DSPIC96":  {"name": "Real Disposable Personal Income",        "category": "Consumer", "transform": "yoy"},
    "RSAFS":    {"name": "Advance Retail Sales",                   "category": "Consumer", "transform": "yoy"},
    "UNRATE":   {"name": "Unemployment Rate",                      "category": "Labor", "transform": "level"},
    "ICSA":     {"name": "Initial Unemployment Claims",            "category": "Labor", "transform": "yoy", "freq": "W"},
    "PAYEMS":   {"name": "Total Nonfarm Payrolls",                 "category": "Labor", "transform": "yoy"},
    "JTSJOL":   {"name": "Job Openings (JOLTS)",                   "category": "Labor", "transform": "yoy"},
    "CPIAUCSL": {"name": "CPI All Urban Consumers",               "category": "Inflation", "transform": "yoy"},
    "PCEPILFE": {"name": "Core PCE Price Index",                   "category": "Inflation", "transform": "yoy"},
    "PCEPI":    {"name": "PCE Chain-Type Price Index",             "category": "Inflation", "transform": "yoy"},
    "CPILFESL": {"name": "Core CPI",                               "category": "Inflation", "transform": "yoy"},
    "PPIACO":   {"name": "PPI All Commodities",                    "category": "Inflation", "transform": "yoy"},
    "HOUST":    {"name": "Housing Starts",                         "category": "Housing", "transform": "yoy"},
    "PERMIT":   {"name": "Building Permits",                       "category": "Housing", "transform": "yoy"},
    "HSN1F":    {"name": "New One-Family Houses Sold",             "category": "Housing", "transform": "yoy"},
    "CSUSHPISA":{"name": "Case-Shiller National Home Price Index", "category": "Housing", "transform": "yoy"},
    "BAA10YM":  {"name": "Baa Corp Bond - 10Y Treasury Spread",   "category": "Banking", "transform": "level"},
    "BUSLOANS": {"name": "Commercial & Industrial Loans",          "category": "Banking", "transform": "yoy"},
    "DRALACBS": {"name": "Delinquency Rate, All Loans",            "category": "Banking", "transform": "level", "freq": "Q"},
    "DRTSCILM": {"name": "Tightening Standards C&I Loans",         "category": "Banking", "transform": "level", "freq": "Q"},
    "T10Y3M":   {"name": "10Y-3M Treasury Spread",               "category": "Yields", "transform": "level", "freq": "D"},
    "T10Y2Y":   {"name": "10Y-2Y Treasury Spread",               "category": "Yields", "transform": "level", "freq": "D"},
    "GS10":     {"name": "10-Year Treasury Yield",                "category": "Yields", "transform": "level"},
    "TB3MS":    {"name": "3-Month Treasury Bill Rate",            "category": "Yields", "transform": "level"},
    "FEDFUNDS": {"name": "Federal Funds Rate",                    "category": "Yields", "transform": "level"},
}

TARGET_SERIES: dict[str, dict] = {
    "USREC": {"name": "NBER Recession Indicator", "category": "Target"},
    "RECPROUSM156N": {"name": "Chauvet-Piger Recession Prob", "category": "Nowcast"},
    "SAHMREALTIME": {"name": "Sahm Rule (real-time)", "category": "Nowcast"},
}

# Approximate publication lag (months) between a series' FRED reference date and
# the month in which the value is first public. Used by the walk-forward backtest
# so a prediction dated month ``t`` only sees data published by the end of ``t``.
# FRED dates quarterly series at the first month of the quarter, so the advance
# GDP estimate (~4 weeks after quarter end) is ~4 months after its reference date.
# Market series (Treasury yields, fed funds, Baa spread) are monthly averages of
# daily data known by month-end, so they lag 0. Anything not listed defaults to 0.
PUBLICATION_LAG_MONTHS: dict[str, int] = {
    # National activity / industrial
    "CFNAI": 1, "CFNAIMA3": 1, "USSLIND": 1, "GDPC1": 4,
    "INDPRO": 1, "BSCICP02USM460S": 1, "TCU": 1, "DGORDER": 1, "IPMAN": 1,
    # Consumer
    "UMCSENT": 1, "PCECC96": 4, "DSPIC96": 1, "RSAFS": 1,
    # Labor
    "UNRATE": 1, "ICSA": 1, "PAYEMS": 1, "JTSJOL": 2,
    # Inflation
    "CPIAUCSL": 1, "PCEPILFE": 1, "PCEPI": 1, "CPILFESL": 1, "PPIACO": 1,
    # Housing
    "HOUST": 1, "PERMIT": 1, "HSN1F": 1, "CSUSHPISA": 2,
    # Banking (quarterly delinquency ~7 weeks after quarter end; SLOOS in-quarter)
    "BUSLOANS": 1, "DRALACBS": 5, "DRTSCILM": 1,
}


# Expected coefficient signs for economic validity (enforced during selection).
SIGN_CONSTRAINTS = {
    "SPREAD": "negative",       # lower spread = higher recession risk
    "UNRATE_CHG3": "positive",  # rising unemployment = higher recession risk
    "UMCSENT": "negative",      # lower sentiment = higher recession risk
    "BUSLOANS_YOY": "negative", # credit contraction = higher recession risk
}

# BIC member: stationary candidate pool, each with an a-priori sign that is
# enforced on every selected feature. SPREAD is always forced in first. Rate
# levels (FEDFUNDS, GS10, TB3MS), UNRATE/TCU/DRALACBS levels, inflation rates,
# near-duplicate spreads (T10Y2Y, T10Y3M), USSLIND (discontinued) and PCECC96
# (quarterly, mostly NaN after resampling) are deliberately not candidates.
BIC_FORCED_FEATURE = "SPREAD"
BIC_POOL_SIGNS: dict[str, str] = {
    **{f: "negative" for f in [
        "GDPC1_YOY", "INDPRO_YOY", "IPMAN_YOY", "DGORDER_YOY", "DSPIC96_YOY", "RSAFS_YOY",
        "PAYEMS_YOY", "JTSJOL_YOY", "HOUST_YOY", "PERMIT_YOY", "HSN1F_YOY", "CSUSHPISA_YOY",
        "BUSLOANS_YOY", "CFNAI", "CFNAIMA3", "BSCICP02USM460S", "UMCSENT"]},
    **{f: "positive" for f in ["UNRATE_CHG3", "ICSA_YOY", "BAA10YM", "DRTSCILM"]},
}
BIC_SIGNS: dict[str, str] = {BIC_FORCED_FEATURE: "negative", **BIC_POOL_SIGNS}
BIC_MIN_COVERAGE = 0.80

# Frozen Estrella-Mishkin (Estrella & Trubin 2006) closed-form parameters.
_EM_CONST = -0.6045
_EM_SPREAD = -0.7374


def all_series_ids() -> list[str]:
    """Every FRED ID this engine pulls (features + targets/benchmark)."""
    return list(SERIES_CONFIG.keys()) + list(TARGET_SERIES.keys())


# Plain-English labels for the two engineered features that aren't raw FRED IDs.
_DERIVED_LABELS = {
    "SPREAD": "10Y–3M Treasury spread",
    "UNRATE_CHG3": "Unemployment rate · 3-month change",
}


def feature_label(feat: str) -> str:
    """Human-readable name for a model feature code (e.g. ``CPILFESL_YOY`` →
    "Core CPI (YoY)"). Falls back to the raw code for anything unmapped."""
    if feat in _DERIVED_LABELS:
        return _DERIVED_LABELS[feat]
    base = feat[:-4] if feat.endswith("_YOY") else feat
    info = SERIES_CONFIG.get(base)
    if info:
        return f"{info['name']} (YoY)" if feat.endswith("_YOY") else info["name"]
    return feat


# ----------------------------------------------------------------- data access


def fetch_probit_panel(start: str = OBS_START) -> pd.DataFrame:
    """Pull the probit universe from FRED, resampled to month-start.

    Reuses :func:`src.data.fred_client.fetch_series` (cached per-series), so
    IDs already loaded for the main dashboard panel are served from cache.
    Per-series failures are skipped rather than aborting the whole fetch.
    """
    from src.data.fred_client import fetch_series

    raw = pd.DataFrame()
    for sid, info in {**SERIES_CONFIG, **TARGET_SERIES}.items():
        try:
            s = fetch_series(sid, start)
        except Exception:  # noqa: BLE001 - any FRED-side failure → skip the series
            continue
        s = pd.Series(s).dropna()
        if s.empty:
            continue
        freq = info.get("freq", "M")
        if freq == "W":
            s = s.resample("MS").mean()
        elif freq == "D":
            s = s.resample("MS").last()
        elif freq == "Q":
            s = s.resample("MS").ffill()
        else:
            s = s.resample("MS").last()
        raw[sid] = s

    if not raw.empty:
        raw.index = pd.to_datetime(raw.index)
        raw = raw.resample("MS").last()
    return raw


# ------------------------------------------------------------ feature engineering


def apply_publication_lags(raw: pd.DataFrame, lags: dict[str, int] | None = None) -> pd.DataFrame:
    """Shift each raw series forward by its publication lag (months).

    After shifting, row ``t`` holds only values that were public by the end of
    month ``t``. The target (USREC) and the nowcast series are untouched.
    """
    lags = PUBLICATION_LAG_MONTHS if lags is None else lags
    out = raw.copy()
    for sid, lag in lags.items():
        if lag and sid in out.columns:
            out[sid] = out[sid].shift(int(lag))
    return out


def engineer_features(
    raw: pd.DataFrame, *, publication_lags: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Apply transforms and build the feature matrix + target column.

    With ``publication_lags=True`` each raw series is first shifted by
    :data:`PUBLICATION_LAG_MONTHS` (derived features such as ``UNRATE_CHG3``
    inherit the lag). The target is always built from the unshifted USREC.
    """
    data = apply_publication_lags(raw) if publication_lags else raw.copy()

    # Derived: long-history spread (GS10-TB3MS reaches back to 1959 monthly) and
    # a Sahm-style 3-month unemployment momentum term.
    if "GS10" in data.columns and "TB3MS" in data.columns:
        data["SPREAD"] = data["GS10"] - data["TB3MS"]
    if "UNRATE" in data.columns:
        ma3 = data["UNRATE"].rolling(3).mean()
        data["UNRATE_CHG3"] = ma3 - ma3.shift(12)

    feat_to_cat: dict[str, str] = {}
    feature_cols: list[str] = []
    for sid, info in SERIES_CONFIG.items():
        if sid not in data.columns:
            continue
        if info["transform"] == "yoy":
            col = f"{sid}_YOY"
            data[col] = data[sid].pct_change(12) * 100
            feature_cols.append(col)
            feat_to_cat[col] = info["category"]
        else:
            feature_cols.append(sid)
            feat_to_cat[sid] = info["category"]

    if "SPREAD" in data.columns:
        feature_cols.append("SPREAD")
        feat_to_cat["SPREAD"] = "Yields (derived)"
    if "UNRATE_CHG3" in data.columns:
        feature_cols.append("UNRATE_CHG3")
        feat_to_cat["UNRATE_CHG3"] = "Labor (derived)"

    feature_cols = sorted(set(feature_cols))

    if "USREC" in data.columns:
        data["TARGET"] = build_target(data["USREC"], TARGET_DEFINITION, TARGET_HORIZON)

    return data, feature_cols, feat_to_cat


def nber_turning_points(usrec: pd.Series) -> list[tuple[pd.Timestamp | None, pd.Timestamp]]:
    """(peak, trough) pairs implied by a monthly 0/1 USREC series.

    NBER convention: USREC = 1 from the month after the peak through the trough.
    For every maximal run of 1s covering months a..b the peak is ``a - 1 month``
    and the trough is ``b``. A run that starts in the first observed month has
    no peak inside the sample (``None``). A run still open at the last
    observation has that month as its provisional trough.
    """
    u = pd.Series(usrec).dropna().astype(float).sort_index()
    if u.empty:
        return []
    pts: list[tuple[pd.Timestamp | None, pd.Timestamp]] = []
    in_rec, start, prev = False, None, None
    for ts, v in u.items():
        if v >= 0.5 and not in_rec:
            in_rec, start = True, ts
        elif v < 0.5 and in_rec:
            in_rec = False
            peak = None if start == u.index[0] else start - pd.DateOffset(months=1)
            pts.append((peak, prev))
        prev = ts
    if in_rec:
        peak = None if start == u.index[0] else start - pd.DateOffset(months=1)
        pts.append((peak, u.index[-1]))
    return pts


def start_target(usrec: pd.Series, horizon: int = TARGET_HORIZON) -> pd.Series:
    """Start-dated label: 1 if an NBER peak P satisfies t+1 <= P <= t+horizon.

    * Months P..T (the peak month and every recession month) are excluded
      (NaN): the model is only trained and scored on months not already in a
      recession. The peak month is dropped rather than labelled 0 because the
      recession starts in P+1, so "no new recession within 12 months" would be
      the wrong label for it.
    * A label is defined only once its whole window is observed, i.e. for
      t + horizon <= last USREC observation; later months are NaN.
    """
    u = pd.Series(usrec).dropna().astype(float).sort_index()
    idx = pd.DatetimeIndex(pd.Series(usrec).index)
    y = pd.Series(0.0, index=idx, name="TARGET")
    if u.empty:
        return y * np.nan
    excl = pd.Series(False, index=idx)
    for peak, trough in nber_turning_points(u):
        lo = peak if peak is not None else idx.min()
        excl |= (idx >= lo) & (idx <= trough)
        if peak is not None:
            y[(idx >= peak - pd.DateOffset(months=horizon)) & (idx <= peak - pd.DateOffset(months=1))] = 1.0
    y[excl.values] = np.nan
    y[idx > u.index[-1] - pd.DateOffset(months=horizon)] = np.nan
    y[idx < u.index[0]] = np.nan
    return y


def build_target(usrec: pd.Series, definition: str = TARGET_DEFINITION, horizon: int = TARGET_HORIZON) -> pd.Series:
    """Training target for ``definition`` ("start", "window" or "point")."""
    if definition == "start":
        return start_target(usrec, horizon)
    if definition == "point":
        return usrec.shift(-horizon)
    return usrec.rolling(window=horizon).max().shift(-horizon)


def recession_state(usrec_latest: float | None, cp: float | None, sahm: float | None) -> str:
    """Which headline regime applies (rule fixed before any result was seen).

    ``nber_recession``: the latest USREC observation is 1 — the start-dated
    model has no training data for this state, so the headline is withheld.
    ``nowcast_flag``: USREC is 0 but Chauvet-Piger >= 50% or the Sahm rule
    >= 0.50 — NBER dates peaks months late, so a recession may already be
    under way; the headline is shown with a caveat. Otherwise ``expansion``.
    """
    if usrec_latest is not None and np.isfinite(usrec_latest) and usrec_latest >= 0.5:
        return "nber_recession"
    cp_on = cp is not None and np.isfinite(cp) and cp >= CP_SIGNAL_THRESHOLD
    sahm_on = sahm is not None and np.isfinite(sahm) and sahm >= SAHM_SIGNAL_THRESHOLD
    return "nowcast_flag" if (cp_on or sahm_on) else "expansion"


def nowcast_panel(raw: pd.DataFrame) -> dict:
    """Descriptive "in recession now?" panel: Chauvet-Piger + Sahm rule.

    Read from the unshifted raw series (as published on FRED). Not scored, not
    averaged, and never part of the ensemble. Chauvet-Piger is FRED's smoothed
    Markov-switching probability, re-estimated with each vintage (not real-time).
    """
    def latest(col):
        if col not in raw.columns:
            return None, None
        s = raw[col].dropna()
        if s.empty:
            return None, None
        return float(s.iloc[-1]), s.index[-1].strftime("%Y-%m")

    cp, cp_dt = latest("RECPROUSM156N")
    sahm, sahm_dt = latest("SAHMREALTIME")
    usrec, usrec_dt = latest("USREC")
    indicators = {
        "Chauvet-Piger": {
            "value": None if cp is None else round(cp, 2), "as_of": cp_dt, "unit": "%",
            "threshold": CP_SIGNAL_THRESHOLD,
            "signal": bool(cp is not None and cp >= CP_SIGNAL_THRESHOLD),
            "source": "FRED RECPROUSM156N (smoothed, revised each vintage)",
        },
        "Sahm rule": {
            "value": None if sahm is None else round(sahm, 2), "as_of": sahm_dt, "unit": "pp",
            "threshold": SAHM_SIGNAL_THRESHOLD,
            "signal": bool(sahm is not None and sahm >= SAHM_SIGNAL_THRESHOLD),
            "source": "FRED SAHMREALTIME (real-time vintages)",
        },
    }
    state = recession_state(usrec, cp, sahm)
    return {
        "indicators": indicators,
        "usrec_latest": usrec,
        "usrec_as_of": usrec_dt,
        "state": state,
        "headline_applicable": state != "nber_recession",
    }


def filter_by_coverage(data: pd.DataFrame, feature_cols: list[str], min_coverage: float = 0.80) -> list[str]:
    """Drop features that don't cover at least ``min_coverage`` of the target window."""
    if "TARGET" not in data.columns:
        return []
    date_range = data["TARGET"].dropna().index
    available = []
    for c in feature_cols:
        if c not in data.columns:
            continue
        if data.loc[date_range, c].notna().mean() >= min_coverage:
            available.append(c)
    return available


# --------------------------------------------------------------- probit helpers


def has_separation(res) -> bool:
    """Detect quasi-complete separation (degenerate probit fit)."""
    if res.prsquared > 0.99:
        return True
    if np.any(np.abs(res.params) > 100):
        return True
    if np.any(np.isnan(res.bse)):
        return True
    return False


def check_sign_constraints(res, selected_feats: Iterable[str], signs: dict[str, str] | None = None) -> bool:
    """True if every constrained feature has the economically correct sign."""
    signs = SIGN_CONSTRAINTS if signs is None else signs
    for feat in selected_feats:
        if feat in signs and feat in res.params.index:
            coef = res.params[feat]
            expected = signs[feat]
            if expected == "negative" and coef > 0:
                return False
            if expected == "positive" and coef < 0:
                return False
    return True


def _fit_probit(y: pd.Series, X: pd.DataFrame, maxiter: int = 300):
    return sm.Probit(y, sm.add_constant(X.astype(float))).fit(
        disp=False, method="bfgs", maxiter=maxiter
    )


def complete_rows(data: pd.DataFrame, feats: list[str], cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Training rows complete for *this model's* features and the target.

    Each model is fit on its own complete-case sample rather than the
    intersection of every candidate feature, so short-history series used by
    other models don't truncate (e.g.) the spread-only NY Fed model.
    """
    cols = list(dict.fromkeys(list(feats) + ["TARGET"]))
    rows = data[cols].dropna()
    if cutoff is not None:
        rows = rows.loc[rows.index <= cutoff]
    return rows


def forward_stepwise_bic(
    y: pd.Series, X_all: pd.DataFrame, feature_names: list[str],
    max_features: int = MAX_FEATURES_BIC, seed: list[str] | None = None,
    signs: dict[str, str] | None = None,
) -> list[str]:
    """Forward-stepwise BIC selection with separation + sign-constraint guards.

    ``max_features`` caps the total (seed included). ``signs`` overrides the
    default :data:`SIGN_CONSTRAINTS` map of required coefficient signs.
    """
    if sm is None:
        raise RuntimeError("statsmodels is required for probit fitting.")
    selected = list(seed) if seed else []
    remaining = [f for f in feature_names if f not in selected]

    if selected:
        best_bic = _fit_probit(y, X_all[selected]).bic
    else:
        best_bic = sm.Probit(y, sm.add_constant(pd.DataFrame(index=X_all.index))).fit(
            disp=False, method="bfgs", maxiter=300
        ).bic

    while remaining and len(selected) < max_features:
        candidates = []
        for feat in remaining:
            try:
                res = _fit_probit(y, X_all[selected + [feat]])
                if not has_separation(res) and check_sign_constraints(res, selected + [feat], signs):
                    candidates.append((feat, res.bic))
            except Exception:  # noqa: BLE001 - singular/non-converged fits are skipped
                pass
        if not candidates:
            break
        best_feat, best_candidate_bic = min(candidates, key=lambda x: x[1])
        if best_candidate_bic >= best_bic:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_bic = best_candidate_bic

    return selected


def select_bic_features(data: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> list[str]:
    """Pre-registered BIC-member selection on the labelled rows up to ``cutoff``.

    SPREAD is forced first; up to ``MAX_FEATURES_BIC - 1`` features are added
    from the stationary, sign-restricted :data:`BIC_POOL_SIGNS` pool by
    forward-stepwise BIC (signs enforced on every selected feature, separation
    guard). A pool feature is eligible if it is observed on at least 80% of the
    labelled training rows; candidates are compared on the rows complete for
    SPREAD + every eligible feature. With fewer than ``MIN_WINDOW`` such rows,
    or if nothing lowers BIC, the member is SPREAD alone. Used on the full
    sample for the live model and inside every walk-forward fold.
    """
    forced = BIC_FORCED_FEATURE
    if forced not in data.columns or "TARGET" not in data.columns:
        return [forced] if forced in data.columns else []
    labelled = data.loc[data["TARGET"].notna()]
    if cutoff is not None:
        labelled = labelled.loc[labelled.index <= cutoff]
    if labelled.empty:
        return [forced]
    eligible = [
        f for f in BIC_POOL_SIGNS
        if f in data.columns and labelled[f].notna().mean() >= BIC_MIN_COVERAGE
    ]
    sample = labelled[[forced] + eligible + ["TARGET"]].dropna()
    if len(sample) < MIN_WINDOW or sample["TARGET"].nunique() < 2:
        return [forced]
    try:
        return forward_stepwise_bic(
            sample["TARGET"].astype(float), sample[[forced] + eligible], [forced] + eligible,
            MAX_FEATURES_BIC, seed=[forced], signs=BIC_SIGNS,
        )
    except Exception:  # noqa: BLE001 - a failed selection falls back to the spread alone
        return [forced]


def _prob(params: np.ndarray, x: np.ndarray) -> float:
    """Probit probability (%) for design row ``x`` (no leading constant)."""
    xc = np.concatenate([[1.0], np.asarray(x, dtype=float)])
    return float(_stats.norm.cdf(xc @ params) * 100.0)


def _latest_values(data: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, pd.Timestamp] | tuple[None, None]:
    """Most recent row in which *every* feature in ``feats`` is observed.

    Using the per-model latest complete row (rather than a panel-wide dropna)
    keeps each model's reading as current as its own inputs allow — a globally
    aligned dropna would stale the headline whenever any one peripheral series
    lags.
    """
    sub = data[feats].dropna()
    if sub.empty:
        return None, None
    return sub.iloc[-1].astype(float).values, sub.index[-1]


# ------------------------------------------------------------------- main report


def _prepare(raw: pd.DataFrame, *, publication_lags: bool = False) -> dict:
    """Engineer features, filter coverage, and run full-sample BIC selection.

    Shared by :func:`build_report` (current estimate) and :func:`walk_forward`
    (out-of-sample). ``model_df`` (rows complete for *every* available feature)
    is only the common sample BIC selection compares candidates on; the models
    themselves are fit on :func:`complete_rows` for their own features.
    """
    data, feature_cols, feat_to_cat = engineer_features(raw, publication_lags=publication_lags)
    if "TARGET" not in data.columns:
        raise RuntimeError("USREC target unavailable — cannot fit the probit ensemble.")

    available = filter_by_coverage(data, feature_cols)
    if not available:
        raise RuntimeError("No features with sufficient coverage to fit the ensemble.")

    model_df = data[available + ["TARGET", "USREC"]].dropna()
    predict_df = data[available].dropna()
    if len(model_df) < MIN_WINDOW:
        raise RuntimeError(f"Only {len(model_df)} training rows; need >= {MIN_WINDOW}.")

    y = model_df["TARGET"].astype(float)
    spread_feat = "SPREAD" if "SPREAD" in available else available[0]

    # Pre-registered BIC-member rule (stationary pool, signs, cap 4, SPREAD forced).
    bic_selected = select_bic_features(data) or [spread_feat]

    return {
        "data": data,
        "feat_to_cat": feat_to_cat,
        "available": available,
        "model_df": model_df,
        "predict_df": predict_df,
        "y": y,
        "spread_feat": spread_feat,
        "bic_selected": bic_selected,
    }


def build_report(raw: pd.DataFrame, *, bootstrap: int = BOOTSTRAP_ITERS, rng_seed: int = 42) -> dict:
    """Fit the four forward models (+ Chauvet-Piger benchmark) and assemble the report.

    ``raw`` is a month-start indexed DataFrame of raw FRED levels (see
    :func:`fetch_probit_panel`). Returns a dict mirroring the email's
    ``daily_summary.json`` plus the time series the dashboard charts need.
    """
    if sm is None or _stats is None:
        raise RuntimeError("statsmodels and scipy are required for the probit ensemble.")

    prep = _prepare(raw)
    data, feat_to_cat = prep["data"], prep["feat_to_cat"]
    available, bic_selected = prep["available"], prep["bic_selected"]
    predict_df = prep["predict_df"]
    spread_feat = prep["spread_feat"]

    # --- fit the re-estimated probit models -----------------------------------
    # Each model is fit on rows complete for its own features (not the
    # intersection of all candidates), so e.g. NY Fed keeps the pre-1976 cycles.
    models: dict[str, dict] = {}
    ny_rows = complete_rows(data, [spread_feat])
    res_ny = _fit_probit(ny_rows["TARGET"].astype(float), ny_rows[[spread_feat]], maxiter=500)
    models["NY Fed"] = {"res": res_ny, "features": [spread_feat]}

    wright_feats = [f for f in [spread_feat, "FEDFUNDS"] if f in available]
    wr_rows = complete_rows(data, wright_feats)
    res_wr = _fit_probit(wr_rows["TARGET"].astype(float), wr_rows[wright_feats], maxiter=500)
    models["Wright"] = {"res": res_wr, "features": wright_feats}

    bic_df = complete_rows(data, bic_selected)
    y_bic = bic_df["TARGET"].astype(float)
    res_bic = _fit_probit(y_bic, bic_df[bic_selected], maxiter=500)
    models["BIC-selected"] = {"res": res_bic, "features": bic_selected}

    # --- current probabilities ------------------------------------------------
    # Each model is scored on the latest row where ITS features are all present,
    # so a lagging peripheral series can't stale the whole panel.
    latest_vals, _ = _latest_values(data, bic_selected)
    if latest_vals is None:
        raise RuntimeError("No complete recent observation for the BIC features.")

    # Forward (recession-start) models — these form the ensemble.
    model_probs: dict[str, float] = {}
    for name, m in models.items():
        x, _ = _latest_values(data, m["features"])
        if x is None:
            continue
        model_probs[name] = _prob(m["res"].params.values, x)
    if "SPREAD" in data.columns:
        spread_val = float(data["SPREAD"].dropna().iloc[-1])
        model_probs["Estrella-Mishkin"] = float(_stats.norm.cdf(_EM_CONST + _EM_SPREAD * spread_val) * 100)

    # "In recession now" nowcast panel — NOT part of the ensemble. Chauvet-Piger
    # and the Sahm rule answer a different question from the start-dated
    # forward models, so they are shown separately and never averaged in.
    nowcast = nowcast_panel(raw)
    benchmarks: dict[str, float] = {}
    cp_now = nowcast["indicators"]["Chauvet-Piger"]["value"]
    if cp_now is not None:
        benchmarks["Chauvet-Piger"] = float(cp_now)

    ensemble_prob = float(np.mean(list(model_probs.values())))
    bic_prob = _prob(res_bic.params.values, latest_vals)

    # --- data currency --------------------------------------------------------
    bic_last_dates: dict[str, pd.Timestamp] = {}
    for feat in bic_selected:
        raw_sid = feat[:-4] if feat.endswith("_YOY") else feat
        src = raw[raw_sid] if raw_sid in raw.columns else (data[feat] if feat in data.columns else None)
        if src is not None and src.last_valid_index() is not None:
            bic_last_dates[feat] = src.last_valid_index()
    if bic_last_dates:
        most_lagged = min(bic_last_dates, key=bic_last_dates.get)
        data_through = bic_last_dates[most_lagged].strftime("%Y-%m")
    else:
        data_through = predict_df.index[-1].strftime("%Y-%m")
    run_dt = pd.Timestamp.today().normalize()
    lagged_series = [
        f"{feat} (last: {dt.strftime('%Y-%m')})"
        for feat, dt in bic_last_dates.items()
        if (run_dt - dt).days > 30
    ]

    # --- bootstrap 90% CI on the BIC model ------------------------------------
    rng = np.random.default_rng(rng_seed)
    boot = []
    n = len(bic_df)
    Xb = bic_df[bic_selected]
    for _ in range(max(0, bootstrap)):
        idx = rng.integers(0, n, size=n)
        try:
            rb = _fit_probit(y_bic.iloc[idx], Xb.iloc[idx], maxiter=200)
            boot.append(_prob(rb.params.values, latest_vals))
        except Exception:  # noqa: BLE001
            pass
    if boot:
        ci_lower, ci_upper = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    else:
        ci_lower = ci_upper = float("nan")

    # --- watchlist trigger levels + ±1SD sensitivity --------------------------
    sensitivity = _watchlist(bic_df, bic_selected, latest_vals, res_bic, feat_to_cat)

    # --- adverse scenario (shock every feature 1SD in its risk direction) -----
    x_adv = latest_vals.copy()
    for j, feat in enumerate(bic_selected):
        coef = res_bic.params.iloc[j + 1]
        sd = bic_df[feat].std()
        x_adv[j] += sd if coef > 0 else -sd
    adverse_prob = _prob(res_bic.params.values, x_adv)

    # --- historical ensemble + BIC fitted series ------------------------------
    ensemble_history, bic_history = _history(predict_df, data, models, res_bic, bic_selected)

    # --- 24-month trend attribution -------------------------------------------
    bic_panel = data[bic_selected].dropna()
    trend_attribution = _trend_attribution(bic_panel, res_bic, bic_selected, latest_vals, bic_prob)

    # --- consensus / signal ---------------------------------------------------
    pv = list(model_probs.values())
    prob_range = max(pv) - min(pv)
    consensus = "STRONG" if prob_range < 15 else "MODERATE" if prob_range < 30 else "WEAK"
    signal = "HIGH" if ensemble_prob > THRESHOLD_ELEVATED else "ELEVATED" if ensemble_prob > THRESHOLD_WARNING else "LOW"

    indicator_readings = {
        feat: {
            "value": round(float(latest_vals[j]), 4),
            "category": feat_to_cat.get(feat, ""),
            "percentile": round(float((bic_df[feat] < latest_vals[j]).mean() * 100), 0),
        }
        for j, feat in enumerate(bic_selected)
    }

    # Trailing 24-month series for sparklines (BIC features).
    indicator_series = {
        feat: data[feat].dropna().tail(24) for feat in bic_selected if feat in data.columns
    }

    return {
        "run_date": run_dt.strftime("%Y-%m-%d"),
        "data_through": data_through,
        "lagged_series": lagged_series,
        "ensemble_probability": round(ensemble_prob, 2),
        "bic_probability": round(bic_prob, 2),
        "ci_lower": round(ci_lower, 2) if np.isfinite(ci_lower) else None,
        "ci_upper": round(ci_upper, 2) if np.isfinite(ci_upper) else None,
        "signal": signal,
        "consensus": consensus,
        "prob_range": round(prob_range, 2),
        "model_probabilities": {k: round(v, 2) for k, v in model_probs.items()},
        "benchmark_probabilities": {k: round(v, 2) for k, v in benchmarks.items()},
        "nowcast": nowcast,
        "recession_state": nowcast["state"],
        "headline_applicable": nowcast["headline_applicable"],
        "bic_selected_features": bic_selected,
        "bic_const": float(res_bic.params.iloc[0]),
        "bic_coefficients": {feat: float(res_bic.params.iloc[j + 1]) for j, feat in enumerate(bic_selected)},
        "indicator_readings": indicator_readings,
        "sensitivity": sensitivity,
        "adverse_scenario_probability": round(adverse_prob, 2),
        "trend_attribution": trend_attribution,
        "model_metadata": {
            "training_observations": len(bic_df),
            "training_start": bic_df.index.min().strftime("%Y-%m"),
            "training_end": bic_df.index.max().strftime("%Y-%m"),
            "pseudo_r2": round(float(res_bic.prsquared), 4),
            "target_definition": TARGET_DEFINITION,
            "target_horizon": TARGET_HORIZON,
            "feature_count": len(available),
        },
        # pandas objects for charts (not JSON-serialisable, dashboard-only)
        "ensemble_history": ensemble_history,
        "bic_history": bic_history,
        "usrec": data["USREC"].dropna() if "USREC" in data.columns else pd.Series(dtype=float),
        "indicator_series": indicator_series,
        "feat_to_cat": feat_to_cat,
    }


def scenario_probability(report: dict, overrides: dict[str, float] | None = None) -> float:
    """BIC-model recession probability (%) under user-set indicator values.

    Recomputes Φ(const + Σ βⱼ xⱼ) for the BIC-selected multivariate model,
    starting from the current reading and applying any ``overrides`` (feature →
    value). Holds every non-overridden feature at its current value — i.e. a
    pure *ceteris paribus* perturbation, which is what the watchlist triggers
    also assume. Cheap (no refit), so it's safe to call on every slider move.
    """
    if _stats is None:
        raise RuntimeError("scipy is required for scenario_probability.")
    coefs = report.get("bic_coefficients") or {}
    const = report.get("bic_const", 0.0)
    readings = report.get("indicator_readings") or {}
    values = {f: float(readings.get(f, {}).get("value", 0.0)) for f in coefs}
    if overrides:
        values.update({f: float(v) for f, v in overrides.items() if f in coefs})
    z = const + sum(coefs[f] * values[f] for f in coefs)
    return float(_stats.norm.cdf(z) * 100.0)


def _watchlist(model_df, bic_selected, latest_vals, res_bic, feat_to_cat) -> list[dict]:
    """For each BIC feature: ±1SD impact and the value that triggers 30%/50%."""
    out = []
    params = res_bic.params.values
    for j, feat in enumerate(bic_selected):
        sd = float(model_df[feat].std())
        current = float(latest_vals[j])
        coef = res_bic.params.iloc[j + 1]

        x_up, x_down = latest_vals.copy(), latest_vals.copy()
        x_up[j] += sd
        x_down[j] -= sd
        prob_up, prob_down = _prob(params, x_up), _prob(params, x_down)

        triggers: dict[str, float | None] = {}
        for threshold in (THRESHOLD_WARNING, THRESHOLD_ELEVATED):
            lo, hi = current - 6 * sd, current + 6 * sd
            mid = current
            for _ in range(60):
                mid = (lo + hi) / 2
                x_test = latest_vals.copy()
                x_test[j] = mid
                p_test = _prob(params, x_test)
                if p_test < threshold:
                    lo, hi = (mid, hi) if coef > 0 else (lo, mid)
                else:
                    lo, hi = (lo, mid) if coef > 0 else (mid, hi)
            x_check = latest_vals.copy()
            x_check[j] = mid
            if abs(_prob(params, x_check) - threshold) < 1.0:
                triggers[f"trigger_{threshold}"] = round(float(mid), 2)
                triggers[f"distance_{threshold}"] = round(float(mid - current), 2)
            else:
                triggers[f"trigger_{threshold}"] = None
                triggers[f"distance_{threshold}"] = None

        out.append({
            "feature": feat,
            "category": feat_to_cat.get(feat, ""),
            "current_value": round(current, 2),
            "std_dev": round(sd, 4),
            "coef": float(coef),
            "hist_min": round(float(model_df[feat].min()), 2),
            "hist_max": round(float(model_df[feat].max()), 2),
            "prob_minus_1sd": round(prob_down, 2),
            "prob_plus_1sd": round(prob_up, 2),
            "impact_pp": round(prob_up - prob_down, 2),
            **triggers,
        })
    return out


def _history(predict_df, data, models, res_bic, bic_selected):
    """Per-date ensemble + BIC fitted probability series (0-100)."""
    bic_history = (res_bic.predict(sm.add_constant(predict_df[bic_selected].astype(float))) * 100).rename("bic")

    em_series = None
    if "SPREAD" in data.columns:
        em_series = pd.Series(
            _stats.norm.cdf(_EM_CONST + _EM_SPREAD * data["SPREAD"].values) * 100,
            index=data.index,
        )

    # The nowcast panel (Chauvet-Piger, Sahm) is intentionally excluded from
    # the ensemble average (different question).
    rows = []
    for dt in predict_df.index:
        vals = []
        for m in models.values():
            feats = m["features"]
            if all(f in predict_df.columns for f in feats):
                x = predict_df.loc[dt, feats].astype(float).values
                vals.append(_prob(m["res"].params.values, x))
        if em_series is not None and dt in em_series.index and np.isfinite(em_series.loc[dt]):
            vals.append(float(em_series.loc[dt]))
        rows.append(np.mean(vals) if vals else np.nan)
    ensemble_history = pd.Series(rows, index=predict_df.index, name="ensemble").dropna()
    return ensemble_history, bic_history.dropna()


def _trend_attribution(predict_df, res_bic, bic_selected, latest_vals, bic_prob) -> dict:
    """Decompose the 24-month BIC probability change into per-feature partial effects."""
    try:
        idx_24m = max(0, len(predict_df) - 25)
        vals_24m = predict_df[bic_selected].iloc[idx_24m].astype(float).values
        prob_24m = _prob(res_bic.params.values, vals_24m)
        prob_change = bic_prob - prob_24m

        linear_index = np.concatenate([[1.0], latest_vals]) @ res_bic.params.values
        phi = float(_stats.norm.pdf(linear_index))
        effects = {
            feat: float(res_bic.params.iloc[j + 1] * (latest_vals[j] - vals_24m[j]) * phi * 100)
            for j, feat in enumerate(bic_selected)
        }
        raw_sum = sum(effects.values())
        if abs(raw_sum) > 1e-6 and abs(prob_change) > 1e-4:
            scale = prob_change / raw_sum
            effects = {k: v * scale for k, v in effects.items()}
        ordered = sorted(effects.items(), key=lambda x: x[1])
        return {
            "prob_24m_ago": round(prob_24m, 2),
            "prob_current": round(bic_prob, 2),
            "prob_change_pp": round(prob_change, 2),
            "partial_effects": {k: round(v, 4) for k, v in effects.items()},
            "top_improvement": {"feature": ordered[0][0], "effect_pp": round(ordered[0][1], 4)},
            "top_risk": {"feature": ordered[-1][0], "effect_pp": round(ordered[-1][1], 4)},
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ----------------------------------------------------------- walk-forward / calibration


def target_series(raw: pd.DataFrame) -> pd.Series:
    """The training/scoring target as a 0/1 series (NaN rows dropped).

    With the default start-dated target this is 1 if an NBER peak falls in
    t+1 … t+12; months already in recession (peak..trough) and months whose
    12-month window is not yet observed are absent.
    """
    data, _, _ = engineer_features(raw)
    return data["TARGET"].dropna().astype(float)


def walk_forward(
    raw: pd.DataFrame, *, oos_start: str = "1985-01-01", refit_every_months: int = 12,
) -> pd.Series:
    """True out-of-sample ensemble probability (%), refit on expanding windows.

    The ensemble here is the four forward models — NY Fed, Wright, BIC
    (re-estimated) and Estrella-Mishkin (closed form). The nowcast panel is
    excluded. At each refit date the re-estimated models are fit using only
    observations whose label was already known by that date (t <= refit_ts - 12
    months; months already in recession carry no label and are dropped), then
    used to predict every month until the next refit. Each model is fit on rows complete for its own
    features and joins the ensemble once it has ``MIN_WINDOW`` such rows.
    Features are shifted by :data:`PUBLICATION_LAG_MONTHS`, so the prediction
    dated ``t`` uses only data published by the end of month ``t``.

    The BIC member's features are **reselected inside every refit** by
    :func:`select_bic_features` on that fold's training rows only, so neither
    selection nor coefficients see future labels. The per-refit selections are
    stored in ``result.attrs["bic_selections"]``.
    """
    if sm is None or _stats is None:
        raise RuntimeError("statsmodels and scipy are required for walk-forward.")

    prep = _prepare(raw, publication_lags=True)
    data, available = prep["data"], prep["available"]
    spread_feat = prep["spread_feat"]

    wright_feats = [f for f in [spread_feat, "FEDFUNDS"] if f in available]
    base_specs = {"NY Fed": [spread_feat], "Wright": wright_feats}
    selections: dict[str, list[str]] = {}

    em = (
        pd.Series(_stats.norm.cdf(_EM_CONST + _EM_SPREAD * data["SPREAD"].values) * 100, index=data.index)
        if "SPREAD" in data.columns else None
    )

    start_ts = pd.Timestamp(oos_start)
    end_ts = data["TARGET"].last_valid_index()
    refit_dates = pd.date_range(start=start_ts, end=end_ts, freq=f"{refit_every_months}MS")
    if len(refit_dates) == 0:
        return pd.Series(dtype=float, name="ensemble_oos")

    monthly: dict[pd.Timestamp, float] = {}
    for i, refit_ts in enumerate(refit_dates):
        next_ts = refit_dates[i + 1] if i + 1 < len(refit_dates) else end_ts + pd.DateOffset(months=1)
        # The label for observation t (an NBER peak in t+1..t+12) is not
        # observable until t+12. To avoid look-ahead, train only on rows whose label was known by
        # the refit date: t <= refit_ts - 12 months.
        label_cutoff = refit_ts - pd.DateOffset(months=12)

        # BIC member reselected on this fold's training rows only.
        fold_bic = select_bic_features(data, cutoff=label_cutoff)
        selections[refit_ts.strftime("%Y-%m")] = list(fold_bic)
        specs = {**base_specs, "BIC-selected": fold_bic}

        fitted: dict[str, np.ndarray] = {}
        for name, feats in specs.items():
            if not feats:
                continue
            train = complete_rows(data, feats, cutoff=label_cutoff)
            if len(train) < MIN_WINDOW:
                continue
            try:
                fitted[name] = _fit_probit(train["TARGET"].astype(float), train[feats], maxiter=300).params.values
            except Exception:  # noqa: BLE001
                pass
        if not fitted:
            continue

        window = data.loc[(data.index >= refit_ts) & (data.index < next_ts)]
        for ts in window.index:
            vals = []
            for name, feats in specs.items():
                if name not in fitted:
                    continue
                row = data.loc[ts, feats]
                if row.isna().any():
                    continue
                vals.append(_prob(fitted[name], row.astype(float).values))
            if em is not None and ts in em.index and np.isfinite(em.loc[ts]):
                vals.append(float(em.loc[ts]))
            if vals:
                monthly[ts] = float(np.mean(vals))

    out = pd.Series(monthly, name="ensemble_oos").sort_index()
    out.attrs["bic_selections"] = selections
    return out


def expanding_base_rate(target: pd.Series, index: pd.Index, label_lag_months: int = 12) -> pd.Series:
    """Base rate a forecaster could know at each date in ``index``.

    For month ``t`` it is the mean of every label observed by then, i.e. labels
    for months ``<= t - label_lag_months`` (the window label for ``s`` needs
    USREC through ``s + 12``). Uses the whole ``target`` history, including
    months before the evaluation window. NaN where no label is known yet.
    """
    y = target.astype(float).dropna().sort_index()
    known = y.expanding().mean()
    known.index = known.index + pd.DateOffset(months=label_lag_months)
    idx = pd.DatetimeIndex(index)
    return known.reindex(known.index.union(idx)).ffill().reindex(idx)


def calibration_stats(pred_pct: pd.Series, target: pd.Series, *, label_lag_months: int = 12) -> dict:
    """Brier / AUC / reliability + base-rate skill for a probability series.

    Two no-skill benchmarks are reported: the *full-sample* base rate (mean of
    the outcome over the evaluation window, only knowable in hindsight) and the
    *expanding* base rate (:func:`expanding_base_rate`, known at each date).
    """
    pred = (pred_pct / 100.0).rename("p")
    y = target.astype(float).rename("y")
    df = pd.concat([pred, y], axis=1, sort=True).dropna()
    if df.empty:
        return {"brier": float("nan"), "auc": float("nan"), "reliability_curve": pd.DataFrame(),
                "baseline_brier": float("nan"), "skill_score": float("nan"),
                "baseline_brier_expanding": float("nan"), "skill_score_expanding": float("nan"),
                "n_obs": 0, "n_obs_expanding": 0, "start": None, "end": None}

    brier = float(((df["p"] - df["y"]) ** 2).mean())
    base_rate = float(df["y"].mean())
    baseline = float(((base_rate - df["y"]) ** 2).mean())
    skill = (1 - brier / baseline) * 100 if baseline > 0 else float("nan")

    # Expanding (known-at-the-time) base rate, scored on the rows where one exists.
    br = expanding_base_rate(target, df.index, label_lag_months)
    ex = df.assign(br=br.values).dropna(subset=["br"])
    if ex.empty:
        baseline_exp = skill_exp = float("nan")
    else:
        brier_ex = float(((ex["p"] - ex["y"]) ** 2).mean())
        baseline_exp = float(((ex["br"] - ex["y"]) ** 2).mean())
        skill_exp = (1 - brier_ex / baseline_exp) * 100 if baseline_exp > 0 else float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(df["y"].values, df["p"].values)) if df["y"].nunique() > 1 else float("nan")
    except Exception:  # noqa: BLE001
        auc = float("nan")

    bins = np.linspace(0, 1, 11)
    df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    rel = (
        df.groupby("bin", observed=True)
        .agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
        .dropna()
    )
    return {
        "brier": brier, "auc": auc, "reliability_curve": rel,
        "baseline_brier": baseline, "skill_score": skill, "n_obs": int(len(df)),
        "baseline_brier_expanding": baseline_exp, "skill_score_expanding": skill_exp,
        "n_obs_expanding": int(len(ex)),
        "start": df.index.min(), "end": df.index.max(),
    }


def compute_probit_report(*, bootstrap: int = BOOTSTRAP_ITERS) -> dict:
    """Fetch the FRED universe and build the full report + calibration backtest.

    Adds in-sample and walk-forward calibration to the base report so the
    methodology page can show the same diagnostics the thematic ensemble had.
    """
    raw = fetch_probit_panel()
    if raw.empty or "USREC" not in raw.columns:
        raise RuntimeError("FRED returned no usable data for the probit ensemble.")

    report = build_report(raw, bootstrap=bootstrap)
    target = target_series(raw)

    report["in_sample_calibration"] = calibration_stats(report["ensemble_history"], target)
    try:
        oos = walk_forward(raw)
        report["oos_history"] = oos
        report["oos_calibration"] = calibration_stats(oos, target)
    except Exception as exc:  # noqa: BLE001
        report["oos_history"] = pd.Series(dtype=float, name="ensemble_oos")
        report["oos_calibration"] = {"error": str(exc)}
    return report
