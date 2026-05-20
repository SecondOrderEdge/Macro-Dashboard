"""Five-model probit recession ensemble (12-month-ahead).

Ported from the standalone Recession_Probability_Model repo that drives the
weekly investment-committee email. This engine runs five *methodologically
distinct* academic specifications over a shared 37-series FRED universe and
averages them:

1. **NY Fed**         — probit on the 10y-3m term spread alone (Estrella-Mishkin 1998).
2. **Wright**         — probit on spread + fed funds rate (Wright 2006).
3. **BIC-selected**   — forward-stepwise probit, sign-constrained, <=9 features.
4. **Estrella-Mishkin** — closed form with frozen 2006 parameters.
5. **Chauvet-Piger**  — FRED's smoothed Markov-switching series (RECPROUSM156N).

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
MAX_FEATURES_BIC = 9
THRESHOLD_WARNING = 30
THRESHOLD_ELEVATED = 50
TARGET_DEFINITION = "point"  # "point" = recession at t+12; "window" = any in t+1..t+12
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
    "RECPROUSM156N": {"name": "Chauvet-Piger Recession Prob", "category": "Benchmark"},
}

# Expected coefficient signs for economic validity (enforced during selection).
SIGN_CONSTRAINTS = {
    "SPREAD": "negative",       # lower spread = higher recession risk
    "UNRATE_CHG3": "positive",  # rising unemployment = higher recession risk
    "UMCSENT": "negative",      # lower sentiment = higher recession risk
    "BUSLOANS_YOY": "negative", # credit contraction = higher recession risk
}

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


def engineer_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Apply transforms and build the feature matrix + target column."""
    data = raw.copy()

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
        if TARGET_DEFINITION == "point":
            data["TARGET"] = data["USREC"].shift(-12)
        else:
            data["TARGET"] = data["USREC"].rolling(window=12).max().shift(-12)

    return data, feature_cols, feat_to_cat


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


def check_sign_constraints(res, selected_feats: Iterable[str]) -> bool:
    """True if every constrained feature has the economically correct sign."""
    for feat in selected_feats:
        if feat in SIGN_CONSTRAINTS and feat in res.params.index:
            coef = res.params[feat]
            expected = SIGN_CONSTRAINTS[feat]
            if expected == "negative" and coef > 0:
                return False
            if expected == "positive" and coef < 0:
                return False
    return True


def _fit_probit(y: pd.Series, X: pd.DataFrame, maxiter: int = 300):
    return sm.Probit(y, sm.add_constant(X.astype(float))).fit(
        disp=False, method="bfgs", maxiter=maxiter
    )


def forward_stepwise_bic(
    y: pd.Series, X_all: pd.DataFrame, feature_names: list[str],
    max_features: int = MAX_FEATURES_BIC, seed: list[str] | None = None,
) -> list[str]:
    """Forward-stepwise BIC selection with separation + sign-constraint guards."""
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
                if not has_separation(res) and check_sign_constraints(res, selected + [feat]):
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


def _prepare(raw: pd.DataFrame) -> dict:
    """Engineer features, filter coverage, and run full-sample BIC selection.

    Shared by :func:`build_report` (point-in-time) and :func:`walk_forward`
    (out-of-sample) so both see an identical feature universe and selection.
    """
    data, feature_cols, feat_to_cat = engineer_features(raw)
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
    seed = ["SPREAD"] if "SPREAD" in available else []

    bic_selected = forward_stepwise_bic(y, model_df[available], available, MAX_FEATURES_BIC, seed)
    if len(bic_selected) <= len(seed):
        # No valid multivariate combination — fall back to the Wright pair.
        bic_selected = [f for f in [spread_feat, "FEDFUNDS"] if f in available]

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
    """Fit all five models and assemble the full analytics report.

    ``raw`` is a month-start indexed DataFrame of raw FRED levels (see
    :func:`fetch_probit_panel`). Returns a dict mirroring the email's
    ``daily_summary.json`` plus the time series the dashboard charts need.
    """
    if sm is None or _stats is None:
        raise RuntimeError("statsmodels and scipy are required for the probit ensemble.")

    prep = _prepare(raw)
    data, feat_to_cat = prep["data"], prep["feat_to_cat"]
    available, bic_selected = prep["available"], prep["bic_selected"]
    model_df, predict_df = prep["model_df"], prep["predict_df"]
    y, spread_feat = prep["y"], prep["spread_feat"]

    # --- fit the re-estimated probit models -----------------------------------
    models: dict[str, dict] = {}
    res_ny = _fit_probit(y, model_df[[spread_feat]], maxiter=500)
    models["NY Fed"] = {"res": res_ny, "features": [spread_feat]}

    wright_feats = [f for f in [spread_feat, "FEDFUNDS"] if f in available]
    res_wr = _fit_probit(y, model_df[wright_feats], maxiter=500)
    models["Wright"] = {"res": res_wr, "features": wright_feats}

    res_bic = _fit_probit(y, model_df[bic_selected], maxiter=500)
    models["BIC-selected"] = {"res": res_bic, "features": bic_selected}

    # --- point-in-time probabilities ------------------------------------------
    # Each model is scored on the latest row where ITS features are all present,
    # so a lagging peripheral series can't stale the whole panel.
    latest_vals, _ = _latest_values(data, bic_selected)
    if latest_vals is None:
        raise RuntimeError("No complete recent observation for the BIC features.")
    model_probs: dict[str, float] = {}
    for name, m in models.items():
        x, _ = _latest_values(data, m["features"])
        if x is None:
            continue
        model_probs[name] = _prob(m["res"].params.values, x)

    if "SPREAD" in data.columns:
        spread_val = float(data["SPREAD"].dropna().iloc[-1])
        model_probs["Estrella-Mishkin"] = float(_stats.norm.cdf(_EM_CONST + _EM_SPREAD * spread_val) * 100)
    if "RECPROUSM156N" in data.columns and not data["RECPROUSM156N"].dropna().empty:
        model_probs["Chauvet-Piger"] = float(data["RECPROUSM156N"].dropna().iloc[-1])

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
    n = len(model_df)
    Xb = model_df[bic_selected]
    for _ in range(max(0, bootstrap)):
        idx = rng.integers(0, n, size=n)
        try:
            rb = _fit_probit(y.iloc[idx], Xb.iloc[idx], maxiter=200)
            boot.append(_prob(rb.params.values, latest_vals))
        except Exception:  # noqa: BLE001
            pass
    if boot:
        ci_lower, ci_upper = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    else:
        ci_lower = ci_upper = float("nan")

    # --- watchlist trigger levels + ±1SD sensitivity --------------------------
    sensitivity = _watchlist(model_df, bic_selected, latest_vals, res_bic, feat_to_cat)

    # --- adverse scenario (shock every feature 1SD in its risk direction) -----
    x_adv = latest_vals.copy()
    for j, feat in enumerate(bic_selected):
        coef = res_bic.params.iloc[j + 1]
        sd = model_df[feat].std()
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
            "percentile": round(float((model_df[feat] < latest_vals[j]).mean() * 100), 0),
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
        "bic_selected_features": bic_selected,
        "bic_const": float(res_bic.params.iloc[0]),
        "bic_coefficients": {feat: float(res_bic.params.iloc[j + 1]) for j, feat in enumerate(bic_selected)},
        "indicator_readings": indicator_readings,
        "sensitivity": sensitivity,
        "adverse_scenario_probability": round(adverse_prob, 2),
        "trend_attribution": trend_attribution,
        "model_metadata": {
            "training_observations": len(model_df),
            "training_start": model_df.index.min().strftime("%Y-%m"),
            "training_end": model_df.index.max().strftime("%Y-%m"),
            "pseudo_r2": round(float(res_bic.prsquared), 4),
            "target_definition": TARGET_DEFINITION,
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
    cp_series = data["RECPROUSM156N"] if "RECPROUSM156N" in data.columns else None

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
        if cp_series is not None and dt in cp_series.index and np.isfinite(cp_series.loc[dt]):
            vals.append(float(cp_series.loc[dt]))
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
    """The point-in-time training target (recession at t+12), as a 0/1 series."""
    data, _, _ = engineer_features(raw)
    return data["TARGET"].dropna().astype(float)


def walk_forward(
    raw: pd.DataFrame, *, oos_start: str = "1985-01-01", refit_every_months: int = 12,
) -> pd.Series:
    """True out-of-sample ensemble probability (%), refit on expanding windows.

    At each refit date the re-estimated models (NY Fed, Wright, BIC) are fit
    using only observations whose 12-month-ahead label was already known by that
    date (t <= refit_ts - 12 months), then used to predict every month until the
    next refit. Estrella-Mishkin (closed form) and Chauvet-Piger (a published
    series) are inherently out-of-sample. No future data — features or labels —
    enters any prediction.

    The BIC *feature set* is selected once on the full sample (parameters are
    re-estimated out-of-sample, selection is not) — the same in-sample-selection
    caveat the methodology page documents; reselecting features at every refit
    would multiply runtime without changing the headline conclusion.
    """
    if sm is None or _stats is None:
        raise RuntimeError("statsmodels and scipy are required for walk-forward.")

    prep = _prepare(raw)
    data, available, bic_selected = prep["data"], prep["available"], prep["bic_selected"]
    spread_feat = prep["spread_feat"]
    model_df = prep["model_df"]

    wright_feats = [f for f in [spread_feat, "FEDFUNDS"] if f in available]
    specs = {"NY Fed": [spread_feat], "Wright": wright_feats, "BIC-selected": bic_selected}

    em = (
        pd.Series(_stats.norm.cdf(_EM_CONST + _EM_SPREAD * data["SPREAD"].values) * 100, index=data.index)
        if "SPREAD" in data.columns else None
    )
    cp = data["RECPROUSM156N"] if "RECPROUSM156N" in data.columns else None

    start_ts = pd.Timestamp(oos_start)
    end_ts = model_df.index.max()
    refit_dates = pd.date_range(start=start_ts, end=end_ts, freq=f"{refit_every_months}MS")
    if len(refit_dates) == 0:
        return pd.Series(dtype=float, name="ensemble_oos")

    monthly: dict[pd.Timestamp, float] = {}
    for i, refit_ts in enumerate(refit_dates):
        next_ts = refit_dates[i + 1] if i + 1 < len(refit_dates) else end_ts + pd.DateOffset(months=1)
        # The label for observation t (recession at t+12) is not observable until
        # t+12. To avoid look-ahead, train only on rows whose label was known by
        # the refit date: t <= refit_ts - 12 months.
        label_cutoff = refit_ts - pd.DateOffset(months=12)
        train = model_df.loc[model_df.index <= label_cutoff]
        if len(train) < MIN_WINDOW:
            continue
        y_train = train["TARGET"].astype(float)

        fitted: dict[str, np.ndarray] = {}
        for name, feats in specs.items():
            if not feats:
                continue
            try:
                fitted[name] = _fit_probit(y_train, train[feats], maxiter=300).params.values
            except Exception:  # noqa: BLE001
                pass

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
            if cp is not None and ts in cp.index and np.isfinite(cp.loc[ts]):
                vals.append(float(cp.loc[ts]))
            if vals:
                monthly[ts] = float(np.mean(vals))

    return pd.Series(monthly, name="ensemble_oos").sort_index()


def calibration_stats(pred_pct: pd.Series, target: pd.Series) -> dict:
    """Brier / AUC / reliability + base-rate skill for a probability series."""
    pred = (pred_pct / 100.0).rename("p")
    y = target.astype(float).rename("y")
    df = pd.concat([pred, y], axis=1, sort=True).dropna()
    if df.empty:
        return {"brier": float("nan"), "auc": float("nan"), "reliability_curve": pd.DataFrame(),
                "baseline_brier": float("nan"), "skill_score": float("nan"), "n_obs": 0}

    brier = float(((df["p"] - df["y"]) ** 2).mean())
    base_rate = float(df["y"].mean())
    baseline = float(((base_rate - df["y"]) ** 2).mean())
    skill = (1 - brier / baseline) * 100 if baseline > 0 else float("nan")

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
