"""Methodology view: data sources, formulas, validation — designed to be scrutiny-proof.

Renders a long, sectioned page documenting every input series, every transform,
the probit specification, the sign-constrained BIC selection algorithm, the
LAME construction, the composite mapping, calibration, and known limitations.

Anything stated here is also reflected in the code; the data sources table is
generated directly from ``SERIES_REGISTRY`` so it cannot drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.series_registry import SERIES_REGISTRY
from src.models.recession_ensemble import RecessionEnsemble, SUBMODELS
from src.ui.components import apply_template, reliability_diagram
from src.ui.theme import PALETTE


# Human-readable descriptions for each FRED series we use. Keep these tightly
# aligned with FRED's own definitions; the FRED ID is the source of truth.
SERIES_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "t10y3m": {"name": "10Y minus 3M Treasury spread", "unit": "percentage points"},
    "t10y2y": {"name": "10Y minus 2Y Treasury spread", "unit": "percentage points"},
    "dgs1mo": {"name": "1-Month Treasury constant-maturity yield", "unit": "percent"},
    "dgs3mo": {"name": "3-Month Treasury constant-maturity yield", "unit": "percent"},
    "dgs6mo": {"name": "6-Month Treasury constant-maturity yield", "unit": "percent"},
    "dgs1":   {"name": "1-Year Treasury constant-maturity yield",  "unit": "percent"},
    "dgs2":   {"name": "2-Year Treasury constant-maturity yield",  "unit": "percent"},
    "dgs3":   {"name": "3-Year Treasury constant-maturity yield",  "unit": "percent"},
    "dgs5":   {"name": "5-Year Treasury constant-maturity yield",  "unit": "percent"},
    "dgs7":   {"name": "7-Year Treasury constant-maturity yield",  "unit": "percent"},
    "dgs10":  {"name": "10-Year Treasury constant-maturity yield", "unit": "percent"},
    "dgs20":  {"name": "20-Year Treasury constant-maturity yield", "unit": "percent"},
    "dgs30":  {"name": "30-Year Treasury constant-maturity yield", "unit": "percent"},
    "unrate": {"name": "Civilian unemployment rate", "unit": "percent"},
    "icsa":   {"name": "Initial jobless claims (SA)", "unit": "persons"},
    "ccsa":   {"name": "Continued unemployment claims (SA)", "unit": "persons"},
    "jtsjol": {"name": "JOLTS — Job openings", "unit": "thousands"},
    "jtsqur": {"name": "JOLTS — Quits rate", "unit": "percent"},
    "awhaetp":{"name": "Average weekly hours, total private", "unit": "hours"},
    "temphelps": {"name": "Temporary help services employment", "unit": "thousands"},
    "payems": {"name": "Total nonfarm payrolls", "unit": "thousands"},
    "u6rate": {"name": "U-6 broad unemployment rate", "unit": "percent"},
    "civpart":{"name": "Labor force participation rate", "unit": "percent"},
    "baa10y": {"name": "Moody's BAA corporate − 10Y Treasury (credit spread)", "unit": "percentage points"},
    "drtscilm": {"name": "Senior Loan Officer Survey: tighter C&I lending standards (net %)", "unit": "percent"},
    "hy_oas": {"name": "ICE BofA U.S. High-Yield option-adjusted spread", "unit": "percentage points"},
    "permit": {"name": "New private housing units authorised by building permits", "unit": "thousands"},
    "houst":  {"name": "Housing starts (total, SAAR)", "unit": "thousands"},
    "pcec96": {"name": "Real personal consumption expenditures", "unit": "billions of chained 2017 USD"},
    "vixcls": {"name": "CBOE VIX (S&P 500 implied volatility)", "unit": "index"},
    "usslind":{"name": "Philadelphia Fed State Coincident Leading Index", "unit": "index"},
    "sp500":  {"name": "S&P 500 index", "unit": "index"},
}


TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "level":   "Raw value, no transform.",
    "yoy":     "12-period percent change × 100.",
    "diff_3m": "3-period first difference (raw units).",
    "ma4":     "Trailing 4-period mean (used to smooth weekly claims).",
    "ma_3m":   "Trailing 3-month mean (sub-daily inputs are resampled to month-end first).",
    "ret_6m":  "6-month percent change × 100; sub-daily inputs use month-end last-of-period.",
}


def render(
    ensemble: RecessionEnsemble | None = None,
    oos_history: pd.DataFrame | None = None,
    oos_stats: dict | None = None,
    exclude_pandemic: bool = True,
) -> None:
    _heading()
    _model_controls(exclude_pandemic)
    _philosophy()
    _data_sources()
    _transforms()
    _yield_curve_section()
    _labor_section()
    _recession_section(ensemble)
    _valuation_section()
    _composite_section()
    _calibration_section(ensemble)
    _walk_forward_section(ensemble, oos_history, oos_stats)
    _nber_section()
    _limitations()
    _reproducibility()


def _model_controls(exclude_pandemic: bool) -> None:
    """Interactive toggle for the pandemic-exclusion training option."""
    import streamlit as st

    st.markdown(
        '<div class="label-small" style="margin-top:8px;">Model fit options</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.6;color:{PALETTE["text_primary"]};">'
            "The 2020 pandemic recession was triggered by an exogenous shock, not a "
            "business-cycle peak. When it's included in training, it pulls labor and "
            "credit coefficients in directions that don't represent typical cyclical dynamics. "
            "We exclude <code>2020-02</code> through <code>2021-06</code> by default; toggle "
            "it back in to see how much the coefficients shift."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        new_value = st.toggle(
            "Exclude 2020-2021 from training",
            value=exclude_pandemic,
            key="exclude_pandemic_toggle",
        )
        if new_value != exclude_pandemic:
            st.session_state.exclude_pandemic = new_value
            st.cache_resource.clear()
            st.rerun()


# ---------------------------------------------------------------- top


def _heading() -> None:
    st.markdown(
        '<div class="label-small">Methodology & Sources</div>'
        '<div style="font-family:Fraunces,serif;font-size:22px;color:#d4d4d0;margin-bottom:4px;">'
        "How the dashboard is built</div>"
        f'<div style="color:{PALETTE["text_muted"]};font-size:12px;letter-spacing:0.05em;">'
        "Every chart on this dashboard is built from FRED data with the formulas below. "
        "Open the source code at <a href=\"https://github.com/SecondOrderEdge/Macro-Dashboard\" "
        f'style="color:{PALETTE["accent"]};">github.com/SecondOrderEdge/Macro-Dashboard</a>.'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


# ---------------------------------------------------------------- philosophy


def _philosophy() -> None:
    _section_header("1. Philosophy")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:14px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};font-family:Fraunces,serif;">'
        "<p>Recession-risk dashboards usually do one of two things badly. They either trust "
        "a single signal — most often the yield curve — which gives a deceptively precise "
        "number that has been spectacularly wrong in plausibly-distinguishable regimes; "
        "or they aggregate everything into one opaque black-box probability that you "
        "cannot interrogate.</p>"
        "<p>This dashboard takes a different approach. It surfaces <b>three independent "
        "lenses</b>: a probit ensemble across 30+ FRED series, a labor-market composite, "
        "and a yield-curve module. Each can be opened and decomposed. When they agree, the "
        "signal is strong. When they disagree, the disagreement itself is the insight.</p>"
        "<p>The headline is a 0–100 composite that weights the recession ensemble at 50% "
        "and the labor and curve lenses at 25% each. The weights are deliberate: the "
        "ensemble is the most information-rich number, but it is also the most opaque, "
        "so the simpler lenses get meaningful weight as a check.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- data sources


def _data_sources() -> None:
    _section_header("2. Data sources")
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:12px;line-height:1.6;margin-bottom:12px;">'
        "Every series below is pulled from "
        '<a href="https://fred.stlouisfed.org" style="color:#d4a574;">FRED</a> '
        "(Federal Reserve Bank of St. Louis) via the official "
        '<a href="https://github.com/mortada/fredapi" style="color:#d4a574;">fredapi</a> '
        "client. Click any FRED ID to open the series on FRED.</div>",
        unsafe_allow_html=True,
    )

    rows = []
    for key, meta in SERIES_REGISTRY.items():
        desc = SERIES_DESCRIPTIONS.get(key, {"name": key, "unit": "—"})
        rows.append(
            {
                "FRED ID": meta["fred_id"],
                "Description": desc["name"],
                "Unit": desc["unit"],
                "Native freq.": _freq_label(meta["freq"]),
                "Transform": meta["transform"],
                "Sign": _sign_label(meta.get("sign")),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True, height=560)

    st.markdown(
        f'<div style="color:{PALETTE["text_tiny"]};font-size:11px;margin-top:6px;">'
        "The <b>Sign</b> column applies only to labor indicators in the LAME composite; "
        "a sign of −1 means the indicator is inverted before z-scoring so that positive "
        "values always indicate expansion."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- transforms


def _transforms() -> None:
    _section_header("3. Transforms")
    rows = [
        (name, desc) for name, desc in TRANSFORM_DESCRIPTIONS.items()
    ]
    body = "".join(
        f'<div class="submodel-row"><span class="name" style="color:{PALETTE["accent"]};">{name}</span>'
        f'<span class="value" style="text-align:left;flex:1;margin-left:24px;color:{PALETTE["text_primary"]};">{desc}</span></div>'
        for name, desc in rows
    )
    st.markdown(
        f'<div class="panel"><div class="panel-body">{body}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:8px;">'
        "Implementation: <code>src/data/series_registry.py · transform_series()</code>."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- yield curve


def _yield_curve_section() -> None:
    _section_header("4. Yield curve module")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p><b>Spreads.</b> The three benchmark spreads are computed as simple yield differences:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "spread_10y3m = DGS10 − DGS3MO\n"
        "spread_10y2y = DGS10 − DGS2\n"
        "spread_5y2y  = DGS5  − DGS2"
        "</pre>"
        "<p><b>Term structure.</b> Snapshots use the most recent observation on or before the "
        "as-of date for each maturity. Comparison curves (3 months ago, 12 months ago) use "
        "the same as-of-lookup against an earlier date.</p>"
        "<p><b>Inversion episodes.</b> The daily 10Y−3M spread is resampled to month-end "
        "average. An <i>episode</i> is a maximal run of consecutive months where the "
        "monthly-average spread is below zero. Episodes lasting three months or fewer are "
        "treated as noise and excluded from the hit-rate statistics.</p>"
        "<p><b>Hit rate.</b> For each qualifying inversion episode, we look for an NBER "
        "recession <i>peak</i> within 36 months of the episode start. The hit rate is "
        "(episodes followed by a peak) / (qualifying episodes).</p>"
        "<p><b>Lead time.</b> Average months from episode start to the following NBER peak, "
        "across episodes that hit.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/yield_curve.py</code>."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- labor


def _labor_section() -> None:
    _section_header("5. Labor composite")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>Ten labor indicators are combined into a single z-score. Indicators with a "
        "negative sign convention (UNRATE, U6RATE, ICSA, CCSA) are inverted so that "
        "positive values always indicate expansion.</p>"
        "<p><b>Step 1 — Transform.</b> Each indicator is resampled to month-end and "
        "transformed per the registry (level, year-over-year, 3-month difference, 4-period "
        "moving average).</p>"
        "<p><b>Step 2 — Z-score.</b> Each indicator is z-scored using its own "
        "expanding-window mean and standard deviation, requiring at least 60 monthly "
        "observations before the first z-score is produced:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "z_t(i) = sign(i) · (x_t(i) − μ_{≤t}(i)) / σ_{≤t}(i)"
        "</pre>"
        "<p><b>Step 3 — Inverse-volatility weighting.</b> The rolling 60-month standard "
        "deviation of each signed z-score gives σ_t(i). Weights are normalised across "
        "indicators with available data each month:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "w_t(i) = (1/σ_t(i)) / Σ_j (1/σ_t(j))"
        "</pre>"
        "<p><b>Step 4 — Composite.</b> The labor composite at time t is:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "L_t = Σ_i w_t(i) · z_t(i)"
        "</pre>"
        "<p><b>Reference date.</b> Slow-release monthlies trail weekly claims by several "
        "weeks. The breakdown displayed on the Labor page snapshots the most recent month "
        "where indicator coverage is at least 70% of peak, so the reading is not anchored "
        "on a stub month with only two series.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/lame.py</code>."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- recession


def _recession_section(ensemble: RecessionEnsemble | None) -> None:
    _section_header("6. Recession probability ensemble")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The ensemble is the arithmetic mean of five thematic <b>probit</b> submodels. "
        "Each submodel is fit independently on a thematic subset of FRED features against "
        "the standard NBER 12-month-forward target:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "y_t = 1{any NBER month ∈ (t, t+12]}\n"
        "P(y_t = 1 | x_t) = Φ(β_0 + β'x_t)"
        "</pre>"
        "<p>where Φ is the standard normal CDF.</p>"
        "<p><b>Sign-constrained selection.</b> After an initial fit on the full feature "
        "set, any feature whose coefficient sign disagrees with the economic prior is "
        "dropped. The signs we encode are: an inverted curve raises recession risk (β &lt; 0 "
        "on spreads); rising unemployment and jobless claims raise risk (β &gt; 0); wider "
        "credit spreads raise risk; falling permits, starts, and consumption raise risk "
        "(β &lt; 0 on YoY growth); elevated VIX raises risk; rising leading indicators and "
        "equity returns lower risk.</p>"
        "<p><b>BIC-stepwise selection.</b> Starting from the sign-filtered feature set, at "
        "each step we drop the feature with the smallest absolute t-statistic, refit, and "
        "accept the drop iff (a) BIC strictly improves and (b) the remaining signs still "
        "agree with their priors. Otherwise we stop.</p>"
        "<p><b>Estimation window.</b> Expanding from <b>1976-01-01</b> — the earliest date "
        "at which most series are available.</p>"
        "<p><b>Ensemble aggregation.</b> Each fitted submodel produces a path of "
        "probabilities; the ensemble probability is the simple arithmetic mean of the five "
        "submodel probabilities at each date. Equal weighting is a deliberate choice — it "
        "avoids letting the curve dominate even when it disagrees materially with the labor "
        "or credit picture.</p>"
        "<p><b>Driver contributions.</b> The marginal effect of feature j at observation x "
        "is reported in percentage points as:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "contribution_j = β_j · (x_j − x̄_j) · φ(β_0 + β'x) · 100"
        "</pre>"
        "<p>This is the local linearization of the probit around the in-sample feature mean, "
        "expressed in percent.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    rows = [
        (
            sub.label,
            ", ".join(f.name for f in sub.features),
        )
        for sub in SUBMODELS
    ]
    body = "".join(
        f'<div class="submodel-row"><span class="name">{name}</span>'
        f'<span class="value" style="text-align:right;color:{PALETTE["text_muted"]};">{feats}</span></div>'
        for name, feats in rows
    )
    st.markdown(
        '<div class="label-small" style="margin-top:12px;">Submodel feature sets (pre-selection)</div>'
        f'<div class="panel"><div class="panel-body">{body}</div></div>',
        unsafe_allow_html=True,
    )

    # Show what survived selection in the current fit, if available.
    if ensemble is not None and getattr(ensemble, "_fitted", None):
        survivors = []
        for sub in SUBMODELS:
            fitted = ensemble._fitted.get(sub.name)
            if fitted is None:
                survivors.append((sub.label, "—"))
            else:
                survivors.append((sub.label, ", ".join(fitted.feature_names) or "—"))
        body = "".join(
            f'<div class="submodel-row"><span class="name">{name}</span>'
            f'<span class="value" style="text-align:right;color:{PALETTE["accent"]};">{feats}</span></div>'
            for name, feats in survivors
        )
        st.markdown(
            '<div class="label-small" style="margin-top:12px;">Features retained in the current fit</div>'
            f'<div class="panel"><div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/recession_ensemble.py</code>. Probit fits use "
        "<code>statsmodels.discrete.discrete_model.Probit</code> with Newton (default) "
        "and BFGS fallback if Newton fails to converge."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- composite


def _valuation_section() -> None:
    _section_header("7. Valuation context · CAPE")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The Dashboard surfaces Robert Shiller's <b>cyclically-adjusted P/E ratio (CAPE)</b> "
        "as a valuation context indicator. CAPE = S&P 500 price / 10-year inflation-adjusted "
        "earnings; it has Shiller-mainstreamed roots back to 1871.</p>"
        "<p><b>Why it's not a recession input.</b> Empirically CAPE has a poor "
        "short-horizon recession-prediction record. It was elevated for most of 2014–2024 "
        "without a recession arriving; including it in the probit ensemble would degrade "
        "out-of-sample fit. The literature is unambiguous: CAPE predicts <i>10-year forward "
        "real equity returns</i>, not 12-month recession probability.</p>"
        "<p><b>Why we surface it anyway.</b> Valuation determines the <i>magnitude</i> of "
        "potential equity damage conditional on a recession arriving. The same labor/credit/"
        "curve signal looks very different for an investor at the 90th percentile of CAPE "
        "than at the 30th percentile.</p>"
        "<p><b>Source.</b> Fetched live from <a href=\"http://www.econ.yale.edu/~shiller/data.htm\" "
        f'style="color:{PALETTE["accent"]};">Shiller\'s Yale page</a> (monthly, .xls); '
        "percentile rank is computed against the post-1950 sample to avoid structural "
        "breaks in the pre-WWII reporting cadence."
        "</div></div>",
        unsafe_allow_html=True,
    )


def _composite_section() -> None:
    _section_header("8. Composite construction")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The headline 0–100 composite blends the three lenses with fixed weights:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "composite = 0.50 · ensemble_pct\n"
        "          + 0.25 · lame_to_risk(L)\n"
        "          + 0.25 · curve_to_risk(spread_10y3m)\n\n"
        "lame_to_risk(z)         = clip(50 − 25 · z,  0, 100)\n"
        "curve_to_risk(spread)   = clip(50 − 20 · spread, 0, 100)\n\n"
        "anchors:\n"
        "  L =  +2σ  → 0   (very firm labor)\n"
        "  L =   0σ  → 50  (neutral)\n"
        "  L =  −2σ  → 100 (deeply contractionary)\n"
        "  spread = +2.5pp → 0   (steep, expansionary)\n"
        "  spread =  0.0pp → 50  (flat)\n"
        "  spread = −2.5pp → 100 (deeply inverted)"
        "</pre>"
        "<p><b>Bands.</b> 0–19 LOW · 20–39 ELEVATED · 40–59 HIGH · 60–100 CRITICAL.</p>"
        "<p><b>Missing inputs.</b> If a component is unavailable (e.g. the ensemble cannot "
        "be evaluated), its weight is redistributed proportionally across the available "
        "components so the composite still uses the full weight budget.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/composite.py</code>."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- calibration


def _calibration_section(ensemble: RecessionEnsemble | None) -> None:
    _section_header("9. Calibration")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The ensemble is scored on its full in-sample history with two metrics:</p>"
        "<ul>"
        "<li><b>Brier score</b> — mean squared error between predicted probabilities and "
        "the realised 0/1 NBER outcome. Lower is better; the unconditional base rate sets "
        "the no-skill floor.</li>"
        "<li><b>AUC</b> — area under the ROC curve, i.e. P(rank(positive) > rank(negative)). "
        "0.5 is no-skill, 1.0 is perfect ordering.</li>"
        "<li><b>Reliability diagram</b> — predicted probabilities are binned into deciles "
        "and plotted against the empirical recession frequency in each bin. A "
        "well-calibrated model sits on the 45° line.</li>"
        "</ul>"
        "<p>The ensemble is fit on the full available history, so all calibration is "
        "<i>in-sample</i>. This is appropriate for a transparency-first dashboard whose "
        "purpose is to surface the current reading, not to prove out-of-sample superiority. "
        "An out-of-sample analysis should use a walk-forward fitting protocol and is "
        "discussed in <code>notebooks/methodology.ipynb</code>.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if ensemble is not None and getattr(ensemble, "_fitted", None):
        try:
            stats = ensemble.calibration_stats()
        except Exception:
            stats = None
        if stats is not None:
            left, right = st.columns([2, 1])
            with left:
                fig = reliability_diagram(stats.get("reliability_curve", pd.DataFrame()))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                brier = stats.get("brier", float("nan"))
                auc = stats.get("auc", float("nan"))
                rows = [
                    ("Brier score (in-sample)", f"{brier:.4f}" if np.isfinite(brier) else "—"),
                    ("AUC (in-sample)", f"{auc:.3f}" if np.isfinite(auc) else "—"),
                    ("Sample start", str(ensemble.START_DATE.date())),
                ]
                body = "".join(
                    f'<div class="submodel-row"><span class="name">{label}</span>'
                    f'<span class="value">{value}</span></div>'
                    for label, value in rows
                )
                st.markdown(
                    f'<div class="panel"><div class="panel-body">{body}</div></div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------- walk-forward


def _walk_forward_section(
    ensemble: RecessionEnsemble | None,
    oos_history: pd.DataFrame | None,
    oos_stats: dict | None,
) -> None:
    """Out-of-sample backtest: how the model would have called recessions in real time."""
    _section_header("10. Out-of-sample backtest")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The headline ensemble is fit on the full sample, which means its in-sample "
        "Brier/AUC overstate what a real-time forecaster would have achieved. A walk-forward "
        "backtest fixes this: at each refit date we drop everything from that date forward, "
        "refit the submodels with the same sign-constrained BIC selection, and predict the "
        "next 12 months. No future information enters any prediction by construction.</p>"
        "<p><b>Protocol.</b> Annual refits starting <code>1985-01-01</code>. Between refits "
        "the most recent fit is used to score every month. Pandemic-exclusion (if enabled) "
        "applies to each refit's training window too.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if oos_history is None or oos_history.empty:
        st.info("Walk-forward backtest not available — model still initialising.")
        return

    # --- OOS history + true recessions -----------------------------------
    st.markdown(
        '<div class="label-small" style="margin-top:8px;">Out-of-sample ensemble probability · NBER recessions shaded</div>',
        unsafe_allow_html=True,
    )

    from src.ui.components import add_recession_shading, apply_template, reliability_diagram
    from src.data.nber import load_nber_recessions

    nber = load_nber_recessions()
    fig = go.Figure()
    s = oos_history["ensemble"].dropna()
    fig.add_trace(
        go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=PALETTE["accent"], width=1.4),
            fill="tozeroy", fillcolor="rgba(212,165,116,0.10)",
            name="OOS ensemble",
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra></extra>",
        )
    )
    fig.add_hline(y=50, line=dict(color="#3d4754", width=1, dash="dot"))
    add_recession_shading(fig, nber)
    fig.update_yaxes(title="Recession probability (%)", range=[0, 100])
    apply_template(fig, height=360, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Calibration metrics -----------------------------------
    if oos_stats is None:
        return
    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="label-small">OOS reliability diagram</div>', unsafe_allow_html=True)
        fig = reliability_diagram(oos_stats.get("reliability_curve", pd.DataFrame()))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        brier = oos_stats.get("brier", float("nan"))
        baseline = oos_stats.get("baseline_brier", float("nan"))
        skill = oos_stats.get("skill_score", float("nan"))
        auc = oos_stats.get("auc", float("nan"))
        n = oos_stats.get("n_obs", 0)
        rows = [
            ("OOS Brier", f"{brier:.4f}" if np.isfinite(brier) else "—"),
            ("Base-rate Brier", f"{baseline:.4f}" if np.isfinite(baseline) else "—"),
            ("Skill score", f"{skill:+.1%}" if np.isfinite(skill) else "—"),
            ("OOS AUC", f"{auc:.3f}" if np.isfinite(auc) else "—"),
            ("OOS observations", f"{n:,}"),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{label}</span>'
            f'<span class="value">{value}</span></div>'
            for label, value in rows
        )
        st.markdown(
            '<div class="panel"><div class="panel-header"><span>OOS calibration</span></div>'
            f'<div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )

    # --- Coefficient evolution -----------------------------------
    coef_hist = oos_history.attrs.get("coef_history", [])
    if coef_hist:
        _render_coef_evolution(coef_hist)


def _render_coef_evolution(coef_hist: list[dict]) -> None:
    """How each submodel's retained features change over the walk-forward refits."""
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Retained features across refits · '
        'shows where the model is structurally stable vs not</div>',
        unsafe_allow_html=True,
    )
    df = pd.DataFrame(coef_hist)
    if df.empty:
        return
    # For each (submodel, feature), plot coefficient time series.
    fig = go.Figure()
    feature_groups: dict[tuple[str, str], list[tuple[pd.Timestamp, float]]] = {}
    for _, row in df.iterrows():
        sub = row["submodel"]
        for feat in row["features"]:
            coef = row["coefs"].get(feat, np.nan)
            feature_groups.setdefault((sub, feat), []).append((row["refit_date"], float(coef)))

    # Take the top 12 most persistent features by appearance count, plot those.
    ranked = sorted(feature_groups.items(), key=lambda kv: -len(kv[1]))[:12]
    color_cycle = [
        PALETTE["accent"], PALETTE["submodel"]["yield_curve"], PALETTE["submodel"]["labor"],
        PALETTE["submodel"]["credit"], PALETTE["submodel"]["housing"], PALETTE["submodel"]["sentiment"],
        "#9d7aa8", "#7a8b99", "#d4a574", "#5ba3a3", "#c97c5d", "#e8b339",
    ]
    for i, ((sub, feat), pts) in enumerate(ranked):
        pts.sort(key=lambda t: t[0])
        xs, ys = zip(*pts)
        fig.add_trace(
            go.Scatter(
                x=list(xs), y=list(ys), mode="lines+markers",
                name=f"{sub} · {feat}",
                line=dict(color=color_cycle[i % len(color_cycle)], width=1.2),
                marker=dict(size=4),
                hovertemplate=f"{sub} · {feat}<br>%{{x|%Y}}: %{{y:+.3f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="#3d4754", width=1, dash="dot"))
    fig.update_yaxes(title="Probit coefficient")
    fig.update_xaxes(title="Refit date")
    from src.ui.components import apply_template
    apply_template(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Features whose coefficient is stable across refits are structurally informative; "
        "features whose coefficient swings between refits should be read with caution."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- nber


def _nber_section() -> None:
    _section_header("11. NBER recession dates")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>Recession periods are the canonical NBER business-cycle peaks and troughs "
        "from the "
        '<a href="https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions" '
        f'style="color:{PALETTE["accent"]};">NBER Business Cycle Dating Committee</a>. '
        "Dates are shipped in <code>data/nber_recessions.csv</code> and are accurate as of "
        "the last NBER announcement.</p>"
        "<p><b>Lag.</b> The NBER announces peaks roughly a year after the fact and troughs "
        "roughly 15 months after the fact. The dating is not a real-time signal; it is the "
        "ground truth against which forward-looking models like ours are scored.</p>"
        "<p><b>Forward target.</b> The probit dependent variable is</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "y_t = 1 if any month s ∈ (t, t+12] has NBER_s = True, else 0"
        "</pre>"
        "<p>This is the same construction used by Estrella & Mishkin (1996) and the NY Fed "
        "yield-curve model.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- limitations


def _limitations() -> None:
    _section_header("12. Limitations")
    points = [
        (
            "In-sample headline · mitigated.",
            "The default reading is fit on the full sample. A full walk-forward backtest "
            "(section 9 above) is computed at app startup and surfaces true out-of-sample "
            "Brier / AUC / reliability — use that for honest predictive performance, not "
            "the in-sample headline.",
        ),
        (
            "Regime shifts · partly mitigated.",
            "The 2020 pandemic is excluded from training by default; toggle the option at "
            "the top of this page to see how coefficients shift when it's included. "
            "We still use static sign priors and equal-weight aggregation — a fully "
            "regime-switching specification (e.g. Markov-switching probit) is out of scope.",
        ),
        (
            "Series availability · partial.",
            "JOLTS begin in 2000 and the BofA high-yield OAS begins in 1996. The credit "
            "submodel relies on BAA10Y and DRTSCILM pre-1996 and on all three series "
            "after that. The walk-forward backtest's earliest credit predictions are "
            "therefore based on fewer features than recent ones.",
        ),
        (
            "Look-ahead in NBER dating · partly mitigated.",
            "Section 9 reports walk-forward predictions using the NBER record as known "
            "today; we don't store NBER vintages. As a real-time check that does not "
            "depend on NBER, the Labor page also shows the Sahm Rule (FRED "
            "<code>SAHMREALTIME</code>) — a recession indicator that uses only real-time "
            "unemployment and is not revised after release.",
        ),
        (
            "Street estimates · partly automated.",
            "<b>NY Fed</b> probability is pulled live from FRED "
            "(<code>RECPROUSM156N</code>, the Chauvet–Piger smoothed model published by "
            "St Louis Fed). <b>Cleveland Fed, Bloomberg, Goldman</b> are still maintained "
            "by hand because their probabilities aren't published as machine-readable feeds.",
        ),
        (
            "Composite weights · interactive.",
            "The default 50/25/25 blend is judgmental. The <i>Dashboard</i> page now has "
            "a weight-sensitivity panel so any reader can re-weight the composite on the "
            "fly. The interactive panel shows how much each lens contributes under your "
            "chosen weighting; the default is documented above for reference.",
        ),
    ]
    body = "".join(
        f'<div style="border-top:1px solid {PALETTE["panel_border"]};padding:10px 0;">'
        f'<div style="color:{PALETTE["accent"]};font-size:12px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{PALETTE["text_primary"]};font-size:13px;line-height:1.6;">{text}</div>'
        "</div>"
        for label, text in points
    )
    st.markdown(
        f'<div class="panel"><div class="panel-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- repro


def _reproducibility() -> None:
    _section_header("13. Reproducibility")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p><b>Code.</b> Every module described above is in the open-source repo at "
        '<a href="https://github.com/SecondOrderEdge/Macro-Dashboard" '
        f'style="color:{PALETTE["accent"]};">github.com/SecondOrderEdge/Macro-Dashboard</a>. '
        "MIT licensed. No hidden constants — anything we assert here is in the code.</p>"
        "<p><b>Tests.</b> 23 deterministic pytest cases cover probability bounds, ensemble "
        "aggregation, sign-constraint dropping, z-score normalisation, weight summation, "
        "spread calculation, inversion detection, hit-rate counting, and composite banding. "
        "Tests use synthetic data and never hit the FRED API.</p>"
        "<p><b>Data attribution.</b> All macro time series © Federal Reserve Bank of "
        "St. Louis (FRED). High-yield OAS © ICE BofA. Recession dates © NBER. The S&P 500 "
        "is an index of S&P Dow Jones Indices LLC.</p>"
        "<p><b>Disclaimer.</b> This is a research and education project. It is not "
        "investment advice and carries no warranty.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- helpers


def _section_header(text: str) -> None:
    st.markdown(
        f'<div style="margin-top:32px;margin-bottom:8px;'
        f'font-family:Fraunces,serif;font-size:18px;color:{PALETTE["text_primary"]};'
        f'border-bottom:1px solid {PALETTE["panel_border"]};padding-bottom:8px;">'
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def _freq_label(code: str) -> str:
    return {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly"}.get(code, code)


def _sign_label(s) -> str:
    if s is None:
        return "—"
    return "+1" if int(s) > 0 else "−1"
