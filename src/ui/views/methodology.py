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

from src.data.revisions import REVISION_SERIES, fetch_revision_pair, revision_summary
from src.data.series_registry import SERIES_REGISTRY, label_for
from src.models.recession_probit import THRESHOLD_ELEVATED, feature_label
from src.ui.components import add_recession_shading, apply_template, reliability_diagram
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
    # External / parallel indicators (display only, not probit inputs)
    "ny_fed_prob":   {"name": "NY Fed-style recession probability (Chauvet–Piger smoothed)", "unit": "percent"},
    "sahm":          {"name": "Sahm Rule recession indicator (real-time)", "unit": "percentage points"},
    "nfci":          {"name": "Chicago Fed National Financial Conditions Index", "unit": "z-score"},
    "anfci":         {"name": "Chicago Fed Adjusted NFCI (macro-controlled)", "unit": "z-score"},
    "stlfsi":        {"name": "St Louis Fed Financial Stress Index (STLFSI4)", "unit": "z-score"},
    "cfnai":         {"name": "Chicago Fed National Activity Index", "unit": "z-score"},
    "cfnai_3ma":     {"name": "CFNAI 3-month moving average (canonical threshold: −0.7)", "unit": "z-score"},
    "wage_tracker":  {"name": "Atlanta Fed Wage Growth Tracker (median, 12-mo MA)", "unit": "percent"},
}


TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "level":   "Raw value, no transform.",
    "yoy":     "12-period percent change × 100.",
    "diff_3m": "3-period first difference (raw units).",
    "ma4":     "Trailing 4-period mean (used to smooth weekly claims).",
    "ma_3m":   "Trailing 3-month mean (sub-daily inputs are resampled to month-end first).",
    "ret_6m":  "6-month percent change × 100; sub-daily inputs use month-end last-of-period.",
}


def render(probit: dict | None = None) -> None:
    _heading()
    _print_download(probit)
    _emit_all(probit)


def _emit_all(probit: dict | None) -> None:
    """The full section sequence — shared by the on-screen render and the
    printable HTML export, so the two can never drift."""
    _philosophy()
    _data_sources()
    _transforms()
    _yield_curve_section()
    _labor_section()
    _recession_section(probit)
    _valuation_section()
    _policy_path_section()
    _composite_section()
    _calibration_section(probit)
    _walk_forward_section(probit)
    _nber_section()
    _revisions_section()
    _growth_section()
    _credit_section()
    _breadth_section()
    _early_warning_section()
    _limitations()
    _reproducibility()


# --------------------------------------------------- printable HTML export


class _NoCtx:
    """No-op context manager so st.columns can be intercepted during capture."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _build_print_html(probit: dict | None) -> str:
    """Render the whole methodology to a self-contained, light HTML string by
    intercepting the Streamlit output calls. Browser print of the live dark app
    clips to the scroll container; this downloadable file prints cleanly."""
    import datetime as _dt

    captured: list[str] = []

    def cap_md(body: str = "", **_k) -> None:
        s = str(body)
        captured.append("<hr>" if s.strip() == "---" else s)

    def cap_info(msg: str = "", **_k) -> None:
        captured.append(f'<p style="color:#555;">{msg}</p>')

    def cap_df(data=None, **_k) -> None:
        try:
            captured.append(pd.DataFrame(data).to_html(index=False, border=0))
        except Exception:  # noqa: BLE001
            pass

    def cap_cols(spec, **_k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_NoCtx() for _ in range(n)]

    saved = {n: getattr(st, n) for n in ("markdown", "plotly_chart", "dataframe", "info", "columns")}
    st.markdown, st.info, st.dataframe, st.columns = cap_md, cap_info, cap_df, cap_cols
    st.plotly_chart = lambda *a, **k: None
    try:
        _emit_all(probit)
    finally:
        for n, fn in saved.items():
            setattr(st, n, fn)

    return _PRINT_TEMPLATE.format(today=_dt.date.today().isoformat(), body="\n".join(captured))


def _print_download(probit: dict | None) -> None:
    """Offer a clean, light, complete copy as a downloadable HTML file."""
    try:
        html = _build_print_html(probit)
    except Exception:  # noqa: BLE001 - never let the export break the page
        return
    st.download_button(
        "Download printable methodology (HTML)",
        data=html.encode("utf-8"),
        file_name="macro-dashboard-methodology.html",
        mime="text/html",
        help="A clean, light, complete copy. Open the file, then Print / Save as PDF — "
             "the in-app dark page can clip when printed directly.",
    )


_PRINT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>U.S. Macro Dashboard — Methodology</title>
<style>
@page {{ margin: 1.4cm; }}
* {{ color:#111 !important; background-color: transparent !important; box-shadow:none !important; text-shadow:none !important; }}
body {{ background:#fff !important; font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
       line-height:1.6; max-width:820px; margin:24px auto; padding:0 18px; }}
h1 {{ font-size:22px; margin:0 0 2px; }}
.panel {{ border:1px solid #bdbdbd !important; border-radius:4px; margin:0 0 14px;
         break-inside:avoid; page-break-inside:avoid; }}
.panel-header {{ padding:8px 12px; border-bottom:1px solid #ddd !important; font-size:11px;
                letter-spacing:.12em; text-transform:uppercase; color:#444 !important; }}
.panel-body {{ padding:12px; font-size:13px; }}
pre {{ background:#f3f3f3 !important; border:1px solid #ddd !important; padding:8px;
      white-space:pre-wrap; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12px; }}
code {{ background:#f3f3f3 !important; padding:1px 3px; border-radius:2px;
       font-family:ui-monospace,Menlo,Consolas,monospace; }}
a {{ color:#14478f !important; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin:6px 0; }}
th,td {{ border:1px solid #ccc !important; padding:4px 8px; text-align:left; }}
hr {{ border:none; border-top:1px solid #ddd; margin:18px 0; }}
.label-small,.label-tiny {{ font-size:11px; letter-spacing:.12em; text-transform:uppercase;
                           color:#666 !important; display:block; margin:14px 0 6px; }}
</style></head>
<body>
<h1>U.S. Macro Dashboard — Methodology</h1>
<div style="color:#666 !important;font-size:12px;margin-bottom:18px;">Generated {today} · open this file, then Print / Save as PDF (Cmd/Ctrl + P)</div>
{body}
</body></html>"""


# ---------------------------------------------------------------- top


def _heading() -> None:
    st.markdown(
        '<div class="label-small">Methodology & Sources</div>'
        '<div style="font-family:Fraunces,serif;font-size:22px;color:#d4d4d0;margin-bottom:4px;">'
        "How the dashboard is built</div>"
        f'<div style="color:{PALETTE["text_muted"]};font-size:12px;letter-spacing:0.05em;">'
        "Every chart on this dashboard is built from public data — almost entirely FRED, "
        "plus the Atlanta Fed Market Probability Tracker for the Policy Path view — with the "
        "formulas below. "
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
        desc = SERIES_DESCRIPTIONS.get(key, {})
        rows.append(
            {
                "FRED ID": meta["fred_id"],
                "Description": label_for(key),
                "Unit": desc.get("unit", "—"),
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
        "<p><b>Front-end funding.</b> A panel of overnight rates — SOFR, the effective fed funds "
        "rate (EFFR), interest on reserve balances (IORB), and the 1-month T-bill — plus the "
        "<b>SOFR−EFFR spread</b> as a money-market stress monitor: SOFR drifting above EFFR/IORB "
        "flags collateral or reserve scarcity in repo. Display only, not a recession input.</p>"
        "<p><b>Yield surface.</b> A heatmap of the full Treasury curve (1mo–30y) over time, so "
        "level shifts and inversions are visible across the whole term structure at once.</p>"
        "<p><b>Principal components.</b> A covariance PCA on the raw monthly yield <i>changes</i> "
        "across maturities (not standardised) extracts three orthogonal factors that, by "
        "construction, recover the classic shape moves: "
        "<b>PC1 ≈ level</b> (parallel shift), <b>PC2 ≈ slope</b> (steepening/flattening), and "
        "<b>PC3 ≈ curvature</b> (belly vs. wings). Each component is sign-oriented so the current "
        "reading is interpretable; explained-variance shares are shown and scores are cumulated "
        "over time.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/yield_curve.py</code> (spreads, inversions); "
        "<code>src/ui/views/curve.py</code> (front-end funding, yield surface, PCA)."
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
        "<p><b>Sahm Rule.</b> Shown alongside the composite: the 3-month moving average of the "
        "unemployment rate minus its trailing 12-month minimum, which fires at <b>+0.5pp</b>. We use "
        "FRED's real-time series (<code>SAHMREALTIME</code>), so it carries no look-ahead and isn't "
        "revised. Bands: &lt;0.2 low · 0.2–0.4 warming · 0.4–0.5 watch · ≥0.5 triggered.</p>"
        "<p><b>Wage tracker.</b> The Atlanta Fed Wage Growth Tracker (median, 12-month MA) gauges "
        "wage pressure; the long-run average is ~3.5%, and sustained readings above ~4.5% have "
        "historically coincided with Fed tightening cycles.</p>"
        "<p><b>Diffusion.</b> The share of the labor indicators with a positive (expansionary) signed "
        "z-score — a breadth check that separates broad strength from a couple of series carrying "
        "the composite.</p>"
        "<p><b>Beveridge curve.</b> A scatter of JOLTS job openings against the unemployment rate, "
        "colored by era. Today's position is read against the nearest-unemployment pre-COVID "
        "(2010–19) month: openings materially above that baseline imply a structurally tighter / "
        "less efficient match. <b>Composition caveat</b> — per Cheremukhin &amp; Restrepo-Echavarria "
        "(<i>The Dual Beveridge Curve</i>, StL Fed WP 2022-021), a rising share of <i>poaching</i> "
        "vacancies (aimed at the already-employed) inflates raw openings, so the curve overstates "
        "the tightness facing the unemployed; we surface the JOLTS quits rate as a free directional "
        "proxy rather than reproduce their structural split.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/lame.py</code> (composite, diffusion, Beveridge); "
        "<code>src/models/external.py</code> (Sahm Rule); "
        "<code>src/models/conditions.py</code> (wage tracker)."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- recession


# Four forward (12-month-ahead) models that form the ensemble, plus the
# coincident benchmark shown separately.
_PROBIT_MODELS = [
    ("NY Fed", "10y-3m term spread", "Re-estimated probit", "Estrella & Mishkin (1998)"),
    ("Wright", "Spread + fed funds rate", "Re-estimated probit", "Wright (2006)"),
    ("BIC-selected", "Data-driven, sign-constrained", "Forward-stepwise BIC", "Berge (2014)"),
    ("Estrella-Mishkin", "10y-3m term spread", "Closed form, frozen 2006 params", "Estrella & Trubin (2006)"),
]
_PROBIT_BENCHMARK = ("Chauvet-Piger", "Markov-switching (coincident)", "FRED RECPROUSM156N — benchmark, not in ensemble", "Chauvet & Piger")


def _recession_section(probit: dict | None) -> None:
    _section_header("6. Recession probability ensemble")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The headline probability is the equal-weighted mean of four "
        "<b>methodologically distinct</b> 12-month-ahead models, each estimating the "
        "probability of an NBER recession 12 months ahead over a shared FRED universe. "
        "Diversifying across model structure — from a single-variable yield-curve probit to a "
        "multivariate BIC model — guards against any one specification's blind spot. A fifth "
        "series, Chauvet–Piger, is reported as a coincident benchmark but excluded from the "
        "average (see below).</p>"

        "<p><b>1 · Target.</b> The label is a <i>window</i> indicator — 1 if a recession occurs "
        "in <i>any</i> of the next twelve months, more robust than asking about a single month "
        "exactly 12 months out:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "y_t = max( USREC_{t+1} , … , USREC_{t+12} )      ∈ {0, 1}"
        "</pre>"

        "<p><b>2 · Feature universe &amp; engineering.</b> 35 raw FRED series across eight "
        "categories (activity, industrial, consumer, labor, inflation, housing, banking, yields), "
        "plus two engineered features — 37 candidates in all. Series flagged <code>yoy</code> "
        "become a 12-month percent change; the rest enter as levels. The two derived terms extend "
        "history and add momentum:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "SPREAD       = GS10 − TB3MS                          (monthly term spread, back to 1959)\n"
        "UNRATE_CHG3  = MA3(UNRATE)_t − MA3(UNRATE)_{t−12}    (Sahm-style 12-mo momentum)"
        "</pre>"
        "<p>A candidate must be observed over at least <b>80%</b> of the target window to be "
        "eligible, so a short-history series can't distort the fit.</p>"

        "<p><b>3 · The probit and how it is fit.</b> Each model maps a linear index of features "
        "through the standard-normal CDF Φ, and the coefficients are chosen by "
        "<b>maximum likelihood</b> (numerically, via BFGS):</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "P(y_t = 1 | x_t) = Φ(β0 + β′x_t)\n"
        "ℓ(β) = Σ_t [ y_t·ln Φ(β0+β′x_t) + (1−y_t)·ln(1−Φ(β0+β′x_t)) ]   → maximised"
        "</pre>"
        "<p>Models are estimated on an <b>expanding window from 1967-01</b> (minimum 120 months). "
        "Each is scored on the most recent month where <i>its own</i> features are all observed, so "
        "a lagging peripheral series can't stale the headline.</p>"

        "<p><b>4 · BIC selection (the multivariate model).</b> Features are chosen by "
        "forward-stepwise minimisation of the Bayesian Information Criterion, which penalises "
        "complexity so the model stays parsimonious:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "BIC = k·ln(n) − 2·ℓ̂    (k = #params incl. intercept, n = months, ℓ̂ = max log-likelihood)"
        "</pre>"
        "<p>Starting from the intercept-only model, at each step every remaining candidate is fit; "
        "it is kept only if it (a) does not induce <b>quasi-complete separation</b> (rejected when "
        "pseudo-R² &gt; 0.99, any |coefficient| &gt; 100, or any standard error is NaN) and "
        "(b) preserves the <b>economically correct sign</b> on constrained features (SPREAD ≤ 0, "
        "UNRATE_CHG3 ≥ 0, U. Michigan sentiment ≤ 0, C&amp;I loan growth ≤ 0). The candidate with "
        "the lowest BIC is added; selection stops when no candidate lowers BIC, or at "
        "<b>9 features</b>.</p>"

        "<p><b>5 · The four models.</b> Three are re-estimated on our data; one is frozen:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "NY Fed            Φ(β0 + β·SPREAD)                   re-estimated\n"
        "Wright (2006)     Φ(β0 + β1·SPREAD + β2·FEDFUNDS)    re-estimated\n"
        "BIC-selected      Φ(β0 + β′x_BIC)                    re-estimated, sign-constrained\n"
        "Estrella–Mishkin  Φ(−0.6045 − 0.7374·SPREAD)         frozen (Estrella–Trubin 2006)"
        "</pre>"

        "<p><b>6 · Aggregation.</b> Equal-weighted mean of the four forward probabilities — "
        "deliberately avoiding letting the yield curve dominate when it disagrees with the broader "
        "panel.</p>"

        "<p><b>7 · Uncertainty.</b> A 90% interval on the BIC model comes from a "
        "<b>pairs bootstrap</b>: resample the (y, X) rows with replacement (~300 draws), refit the "
        "probit each time, score today's feature row, and take the 5th and 95th percentiles of the "
        "resulting probabilities.</p>"

        "<p><b>Why Chauvet–Piger is a benchmark, not an input.</b> It is a <i>coincident</i> "
        "smoothed Markov-switching nowcast (FRED <code>RECPROUSM156N</code>) — it estimates "
        "whether we are in recession <i>now</i>, not 12 months ahead. Averaging a coincident "
        "nowcast with forward models would blend forecast horizons, so it is shown alongside "
        "for context but kept out of the ensemble.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    body = "".join(
        f'<div class="submodel-row"><span class="name">{name}</span>'
        f'<span class="value" style="text-align:right;color:{PALETTE["text_muted"]};">'
        f"{feats} · {method} · {ref}</span></div>"
        for name, feats, method, ref in [*_PROBIT_MODELS, _PROBIT_BENCHMARK]
    )
    st.markdown(
        '<div class="label-small" style="margin-top:12px;">Four-model ensemble + coincident benchmark</div>'
        f'<div class="panel"><div class="panel-body">{body}</div></div>',
        unsafe_allow_html=True,
    )

    # Show the BIC features and per-model probabilities from the current fit.
    if probit and "error" not in probit:
        feats = probit.get("bic_selected_features", [])
        meta = probit.get("model_metadata", {})
        feat_txt = ", ".join(f"{feature_label(f)} (<code>{f}</code>)" for f in feats) if feats else "—"
        st.markdown(
            '<div class="label-small" style="margin-top:12px;">BIC-selected features · current fit</div>'
            f'<div class="panel"><div class="panel-body">'
            f'<div class="submodel-row"><span class="name">Retained features</span>'
            f'<span class="value" style="text-align:right;color:{PALETTE["accent"]};">{feat_txt}</span></div>'
            f'<div class="submodel-row"><span class="name">Training window</span>'
            f'<span class="value">{meta.get("training_start","—")} → {meta.get("training_end","—")}</span></div>'
            f'<div class="submodel-row"><span class="name">Pseudo R² (BIC model)</span>'
            f'<span class="value">{meta.get("pseudo_r2","—")}</span></div>'
            f'<div class="submodel-row"><span class="name">Candidate features</span>'
            f'<span class="value">{meta.get("feature_count","—")}</span></div>'
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/recession_probit.py</code>. Probit fits use "
        "<code>statsmodels.discrete.discrete_model.Probit</code> (BFGS); the closed-form "
        "Estrella-Mishkin model uses the published 2006 coefficients."
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
        "<p><b>Two companion measures.</b> The panel also reports, from the same workbook, "
        "<b>TR&nbsp;CAPE</b> (total-return CAPE, which reinvests dividends and so is more "
        "comparable across eras as payouts shifted from dividends to buybacks) and the "
        "<b>Excess CAPE Yield</b> — the CAPE earnings yield (1/CAPE) minus the real 10-year "
        "Treasury yield. The ECY is Shiller's own answer to the standard objection that high "
        "CAPE is justified by low real rates: a low ECY means equities are richly priced "
        "<i>even after</i> accounting for rates, and it has a stronger forward-excess-return "
        "record than raw CAPE. Both are read directly from Shiller's file (not recomputed).</p>"
        "<p><b>Source.</b> Robert Shiller's monthly CAPE series. A scheduled GitHub Action "
        "(<code>.github/workflows/refresh-cape.yml</code>) refreshes a bundled "
        "<code>data/cape.csv</code> from "
        "<a href=\"https://shillerdata.com\" "
        f'style="color:{PALETTE["accent"]};">Shiller\'s spreadsheet</a>; the app reads that '
        "committed file and overlays the live source when reachable (an older datahub mirror "
        "supplies deep history). Percentile rank is computed against the post-1950 sample to "
        "avoid structural breaks in the pre-WWII reporting cadence."
        "</div></div>",
        unsafe_allow_html=True,
    )


def _policy_path_section() -> None:
    _section_header("8. Policy path · market-implied FOMC expectations")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The <b>Policy Path</b> tab surfaces the Atlanta Fed's "
        '<a href="https://www.atlantafed.org/cenfis/market-probability-tracker" '
        f'style="color:{PALETTE["accent"]};">Market Probability Tracker</a>, which backs out the '
        "market-implied probability distribution of the FOMC policy rate after each upcoming "
        "quarterly contract from CME options on SOFR futures. It is a forward-looking, "
        "market-priced complement to the (spot) Yield Curve module.</p>"
        "<p><b>What we show.</b> For the latest snapshot: a fan chart of the published mean path "
        "with its 25th–75th percentile band; a comparison of the mean path across recent "
        "snapshots (how expectations have re-priced); a heatmap of the probability on each 25bp "
        "target range per meeting; and the next-meeting hike/cut odds. The mean, mode, and "
        "percentiles are taken <i>directly</i> from the Atlanta Fed export — we do not re-estimate "
        "the distribution.</p>"
        "<p><b>Why it is not a recession input.</b> It measures market <i>expectations</i> of "
        "policy, not recession risk, so it is shown as context and excluded from the composite "
        "and the probit ensemble.</p>"
        "<p><b>Data handling.</b> Unlike the FRED-backed panels, the app doesn't fetch this on "
        "load; it reads a <b>bundled CSV</b> (<code>data/market_probability_tracker.csv</code>) "
        "built from the Atlanta Fed's <i>MPT Historical Data</i> (.xlsx) export. A scheduled "
        "<b>GitHub Action</b> (<code>.github/workflows/refresh-market-probability.yml</code>) "
        "attempts a daily refresh — downloading the .xlsx, validating it through this same parser, "
        "and committing the CSV only when it changes (which redeploys the app). If the source "
        "blocks automated access, the last committed snapshot is served and can be refreshed by "
        "replacing the file; the in-app <i>snapshot as-of date</i> shows how fresh the copy is.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/data/market_probability.py</code>, "
        "<code>src/ui/views/rate_path.py</code>. Source data © Federal Reserve Bank of Atlanta; "
        "derived from CME Group options on SOFR futures."
        "</div>",
        unsafe_allow_html=True,
    )


def _composite_section() -> None:
    _section_header("9. Composite construction")
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
        "<p><b>Overview-page reading aids.</b> The Macro Dashboard home carries three extras built "
        "on top of the composite. <b>Weight sensitivity</b> lets you re-weight the three lenses with "
        "sliders (auto-renormalised to 100%) to see how judgmental the 50/25/25 blend is. "
        "<b>Historical analogues</b> finds the five past months closest to today by Euclidean "
        "distance in standardised (ensemble, labor σ, 10Y−3M) space — excluding the last 24 months — "
        "and reports what followed (peak ensemble probability and whether an NBER recession hit "
        "within 12 months). <b>Financial conditions</b> shows NFCI, ANFCI, STLFSI, and the CFNAI "
        "3-month activity index as <i>parallel display indicators</i> (positive NFCI = tighter); "
        "they are deliberately <b>not</b> composite or ensemble inputs, to avoid collinearity with "
        "the credit spreads already in the models.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/composite.py</code>; overview extras in "
        "<code>src/ui/views/dashboard.py</code> (analogues, weight sensitivity, financial conditions)."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- calibration


def _calibration_section(probit: dict | None) -> None:
    _section_header("10. Calibration · in-sample")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The ensemble is scored against the realised NBER outcome with three metrics:</p>"
        "<ul>"
        "<li><b>Brier score</b> — mean squared error between predicted probabilities and the "
        "0/1 outcome. Lower is better; the unconditional base rate sets the no-skill floor.</li>"
        "<li><b>AUC</b> — area under the ROC curve. 0.5 is no-skill, 1.0 is perfect ordering.</li>"
        "<li><b>Reliability diagram</b> — predictions binned into deciles vs the empirical "
        "recession frequency in each bin. A calibrated model sits on the 45° line.</li>"
        "</ul>"
        "<p>The figures here are <i>in-sample</i> (the models see the whole history). For an "
        "honest read of predictive performance, see the walk-forward backtest in section 10.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    if probit and "error" not in probit:
        _calibration_panel(probit.get("in_sample_calibration"), "in-sample")


def _calibration_panel(stats: dict | None, label: str) -> None:
    """Reliability diagram + metric card for a calibration-stats dict."""
    if not stats or "error" in stats:
        st.info(f"Calibration ({label}) not available.")
        return
    left, right = st.columns([2, 1])
    with left:
        fig = reliability_diagram(stats.get("reliability_curve", pd.DataFrame()))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        brier = stats.get("brier", float("nan"))
        baseline = stats.get("baseline_brier", float("nan"))
        skill = stats.get("skill_score", float("nan"))
        auc = stats.get("auc", float("nan"))
        n = stats.get("n_obs", 0)
        rows = [
            (f"Brier ({label})", f"{brier:.4f}" if np.isfinite(brier) else "—"),
            ("Base-rate Brier", f"{baseline:.4f}" if np.isfinite(baseline) else "—"),
            ("Skill score", f"{skill:+.1f}%" if np.isfinite(skill) else "—"),
            (f"AUC ({label})", f"{auc:.3f}" if np.isfinite(auc) else "—"),
            ("Observations", f"{n:,}"),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{lab}</span>'
            f'<span class="value">{val}</span></div>'
            for lab, val in rows
        )
        st.markdown(
            '<div class="panel"><div class="panel-header"><span>Calibration</span></div>'
            f'<div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- walk-forward


def _walk_forward_section(probit: dict | None) -> None:
    """Out-of-sample backtest: how the ensemble would have called recessions in real time."""
    _section_header("11. Out-of-sample backtest")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>In-sample Brier/AUC overstate what a real-time forecaster would have achieved. "
        "The walk-forward backtest fixes this: at each refit date the re-estimated models "
        "(NY Fed, Wright, BIC) are fit using only observations whose 12-month-ahead outcome "
        "was already known by that date — i.e. month <code>t</code> enters training only once "
        "<code>t+12 ≤ refit date</code>, so a label that wouldn't yet have been observed can't "
        "leak in. Estrella-Mishkin (closed form) and Chauvet-Piger (a published series) are "
        "inherently out-of-sample.</p>"
        "<p><b>Protocol.</b> Annual refits from <code>1985-01-01</code>; the most recent fit "
        "scores every month until the next refit. The BIC <i>feature set</i> is selected once "
        "on the full sample (coefficients are re-estimated out-of-sample, selection is not) — "
        "reselecting features at every refit would multiply runtime without changing the "
        "headline conclusion. The OOS series therefore starts once enough labelled history has "
        "accumulated under the 12-month cutoff.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if not probit or "error" in probit:
        st.info("Walk-forward backtest not available — model still initialising.")
        return
    oos_history = probit.get("oos_history")
    if oos_history is None or oos_history.empty:
        st.info("Walk-forward backtest not available.")
        return

    st.markdown(
        '<div class="label-small" style="margin-top:8px;">Out-of-sample ensemble probability · NBER recessions shaded</div>',
        unsafe_allow_html=True,
    )
    usrec = probit.get("usrec")
    nber = (usrec > 0) if usrec is not None and not usrec.empty else None
    fig = go.Figure()
    s = oos_history.dropna()
    fig.add_trace(
        go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=PALETTE["accent"], width=1.4),
            fill="tozeroy", fillcolor="rgba(212,165,116,0.10)",
            name="OOS ensemble",
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra></extra>",
        )
    )
    fig.add_hline(y=THRESHOLD_ELEVATED, line=dict(color="#3d4754", width=1, dash="dot"))
    if nber is not None:
        add_recession_shading(fig, nber)
    fig.update_yaxes(title="Recession probability (%)", range=[0, 100])
    apply_template(fig, height=360, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="label-small">Out-of-sample calibration</div>', unsafe_allow_html=True)
    _calibration_panel(probit.get("oos_calibration"), "OOS")


# ---------------------------------------------------------------- nber


def _nber_section() -> None:
    _section_header("12. NBER recession dates")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>Recession periods are the canonical NBER business-cycle dates, sourced live from "
        "FRED's <code>USREC</code> indicator ("
        '<a href="https://fred.stlouisfed.org/series/USREC" '
        f'style="color:{PALETTE["accent"]};">NBER-based Recession Indicator</a>), so the '
        "shading auto-updates when the NBER dates a new cycle. If the FRED fetch is "
        "unavailable the dashboard falls back to the bundled "
        "<code>data/nber_recessions.csv</code>.</p>"
        "<p><b>Lag.</b> The NBER announces peaks roughly a year after the fact and troughs "
        "roughly 15 months after the fact. The dating is not a real-time signal; it is the "
        "ground truth against which forward-looking models like ours are scored.</p>"
        "<p><b>Forward target.</b> The probit dependent variable is the within-12-months "
        "(window) target</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "y_t = 1 if USREC = 1 in any month from t+1 to t+12, else 0"
        "</pre>"
        "<p>i.e. \"does a recession occur at some point in the next year?\" — matching how the "
        "headline probability is read. The point-in-time variant (recession exactly at t+12) "
        "prints lower, spikier numbers and can miss short recessions.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- limitations


def _revisions_section() -> None:
    _section_header("13. Data revisions (ALFRED)")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>Official statistics are revised — sometimes enough to rewrite the story. Payroll "
        "benchmark revisions have flipped reported job <i>growth</i> into <i>contraction</i> a year "
        "after the fact. Backtesting on today's final-revised history therefore embeds look-ahead "
        "bias: the model would appear to have known things the public didn't yet.</p>"
        "<p>We mitigate this two ways. The recession target and Sahm signal use FRED's "
        "<b>real-time</b> series (<code>USREC</code>, <code>SAHMREALTIME</code>) rather than "
        "retroactively revised ones. And below we show, via ALFRED, how far each high-revision "
        "series' <b>first public print</b> ends up moving from its <b>final-revised</b> value — the "
        "magnitude of the bias a naive backtest would absorb. (Full point-in-time re-fitting on "
        "as-of-date vintages is a batch pipeline, out of scope for the live app.)</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    summaries = []
    for sid, label, unit in REVISION_SERIES:
        pair = fetch_revision_pair(sid)
        if pair is None or pair.empty:
            continue
        if sid == "GDPC1":
            # GDP levels are periodically rebased (reference-year changes), so
            # level diffs are dominated by rebasing artifacts. Compare the
            # annualized growth rate — the meaningful real-time revision.
            first = ((pair["first"] / pair["first"].shift(1)) ** 4 - 1) * 100.0
            latest = ((pair["latest"] / pair["latest"].shift(1)) ** 4 - 1) * 100.0
            summ = revision_summary(first.dropna(), latest.dropna())
            label, unit = "Real GDP growth", "pp, annualized"
        else:
            summ = revision_summary(pair["first"], pair["latest"])
        if summ["n"]:
            summaries.append((sid, label, unit, summ))

    if not summaries:
        st.info(
            "Revision comparison unavailable (ALFRED fetch needs a live FRED key / network). "
            "The real-time series the model relies on are unaffected."
        )
        return

    header = (
        '<tr style="border-bottom:1px solid #1f2630;color:#6b7280;font-size:10px;'
        'letter-spacing:0.08em;text-transform:uppercase;">'
        "<th style='text-align:left;padding:6px 8px;'>Series</th>"
        "<th style='text-align:right;padding:6px 8px;'>Obs</th>"
        "<th style='text-align:right;padding:6px 8px;'>Median revision</th>"
        "<th style='text-align:right;padding:6px 8px;'>Mean abs revision</th>"
        "<th style='text-align:right;padding:6px 8px;'>% revised down</th></tr>"
    )
    body = []
    for sid, label, unit, summ in summaries:
        body.append(
            f'<tr style="border-bottom:1px solid #141a22;color:{PALETTE["text_primary"]};font-size:12px;">'
            f'<td style="padding:6px 8px;">{label}<span style="color:#5a6470;"> · {sid} · {unit}</span></td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{summ["n"]:,}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{summ["median_revision"]:+.1f}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{summ["mean_abs_revision"]:.1f}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{summ["share_revised_down"]:.0f}%</td></tr>'
        )
    st.markdown(
        '<div class="panel"><div class="panel-body"><table style="width:100%;border-collapse:collapse;">'
        + header + "".join(body) + "</table></div></div>",
        unsafe_allow_html=True,
    )

    # Illustrative chart: first vs final-revised for the first series with data.
    sid, label, unit, summ = summaries[0]
    df = summ["aligned"].tail(240)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["first"].values, mode="lines",
        line=dict(color=PALETTE["text_muted"], width=1, dash="dot"), name="First release",
        hovertemplate="%{x|%b %Y}<br>%{y:.0f}<extra>first</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["latest"].values, mode="lines",
        line=dict(color=PALETTE["accent"], width=1.6), name="Final revised",
        hovertemplate="%{x|%b %Y}<br>%{y:.0f}<extra>revised</extra>",
    ))
    fig.update_yaxes(title=unit)
    apply_template(fig, height=320)
    st.markdown(
        f'<div class="label-small" style="margin-top:8px;">{label} · first release vs final revised</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def _growth_section() -> None:
    _section_header("14. Growth & nowcasting")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The recession ensemble is <i>forward-looking</i> (probability of a downturn in the "
        "next 12 months). The Growth tab is the <i>coincident</i> complement — how fast output is "
        "expanding right now — because the official GDP print arrives with a long lag and is heavily "
        "revised (section 13).</p>"
        "<p><b>Surface, don't rebuild.</b> Institutional nowcasts (the Atlanta Fed's bottom-up "
        "GDPNow, the NY Fed's dynamic-factor model) are best-in-class and already published to FRED. "
        "We read them rather than re-implement a Kalman-filter / bridge-equation stack. The tab "
        "shows the official real-GDP growth rate (<code>A191RL1Q225SBEA</code>), GDPNow "
        "(<code>GDPNOW</code>), real GDI (<code>A261RL1Q225SBEA</code>), the BEA contributions to "
        "growth (consumption / investment / government / net exports / inventories, the "
        "<code>…RY2Q224SBEA</code> family), and the Dallas Fed Weekly Economic Index "
        "(<code>WEI</code>). Any series FRED rejects is skipped, never fatal.</p>"
        "<p><b>GDP vs GDI.</b> Income- and expenditure-side measures of the same economy should "
        "match but don't; GDI has historically led GDP at turning points, so a wide GDP−GDI gap is a "
        "flag that the headline may be revised toward the weaker side.</p>"
        "<p><b>Coincident growth factor (the one original model).</b> Deliberately the simplest "
        "thing that works: take payrolls (<code>PAYEMS</code>), industrial production "
        "(<code>INDPRO</code>), retail sales (<code>RSAFS</code>), and real consumption "
        "(<code>PCEC96</code>) year-over-year; z-score each over the common sample; average the "
        "available z-scores per month. The result is a unitless momentum gauge (0 = trend, positive "
        "= above trend), not a GDP forecast — a transparent corroboration for the nowcast, not a "
        "black-box DFM. The tab validates it two ways: a scatter of the factor against the GDP "
        "print (it should slope up), and an overlay against the 12-month recession probability "
        "(they move inversely — weak current momentum coincides with elevated forward risk).</p>"
        "<p><b>Revisions, measured right.</b> The real-time-vs-revised panel compares the "
        "first-published vs latest-revised <i>annualized growth rate</i>, not the level. GDP "
        "levels are periodically rebased (the chained-dollar reference year shifts), so level "
        "differences are dominated by rebasing artifacts; growth-rate revisions isolate the "
        "genuine real-time uncertainty (typically under ~1.5 pp).</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _credit_section() -> None:
    _section_header("15. Credit & funding stress")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>Credit conditions lead the cycle: lenders tighten and spreads widen before output and "
        "employment roll over. The Credit tab distils this into a standardized <b>stress "
        "composite</b> — high-yield and investment-grade OAS, the Baa–10y spread, the Chicago Fed "
        "conditions index (NFCI), the St. Louis stress index (STLFSI), and the SLOOS net share of "
        "banks tightening C&amp;I standards. Each is resampled to month-start and oriented so higher "
        "= more stress, then z-scored over the common 1997+ sample and averaged:</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "z_i = (x_i − mean(x_i)) / std(x_i)     over the common 1997+ sample\n"
        "stress_t = mean_i z_i(t)               (averaged over the series available that month)"
        "</pre>"
        "<p>The quarterly SLOOS series is forward-filled up to two months so it aligns with the "
        "monthly inputs. Severity is a percentile of the composite against its own history: "
        "<b>≥90</b> stressed · <b>70–90</b> elevated · <b>40–70</b> moderate · <b>&lt;40</b> calm.</p>"
        "<p><b>Overlap is intentional.</b> NFCI already embeds credit spreads, so the inputs are "
        "not independent — the composite is a robust summary of one underlying stress factor, not "
        "a multi-signal model. It's plotted against the 12-month recession probability, where a "
        "positive correlation is expected (stress and forward risk rise together).</p>"
        "<p><b>CLO supply gauge — and honest scope.</b> The CLO panel reads the Fed's quarterly "
        "Z.1 Financial-Accounts estimate of CLO liabilities outstanding and leveraged loans held "
        "by CLOs (added 2019; most US CLOs are offshore-domiciled). It's a coarse, ~10-week-lagged "
        "<i>supply</i> trend — is the CLO machine expanding or contracting — <b>not</b> tranche "
        "analytics. Timely CLO demand, AAA spreads, and speculative-grade default/distress rates "
        "live behind paid vendors (JPM, Barclays, LCD, Moody's, S&amp;P) and are out of scope. "
        "Funding/liquidity series (SOFR, Fed balance sheet, reserves, ON RRP, M2) sit beside the "
        "composite as context rather than inside it, since their stress sign is regime-dependent.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _breadth_section() -> None:
    _section_header("16. Breadth, diffusion & market-implied (Pulse)")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The Pulse tab answers a question the ensemble can't on its own: <i>is the weakness "
        "broad or narrow?</i> A few indicators rolling over is noise; a majority rolling over "
        "together is how real downturns begin. It is descriptive, not a second forecast.</p>"
        "<p><b>Breadth / diffusion.</b> Over the labor-indicator signed z-score panel we compute "
        "the <b>share below trend</b> (indicators with z &lt; 0) and the <b>momentum breadth</b> "
        "(share whose z fell over the last 3 months). Both skip months below 50% indicator "
        "coverage so the ragged right edge doesn't whipsaw the reading, and the snapshot uses each "
        "indicator's own freshest value.</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "below-trend % = 100 · #{ i : z_i < 0 } / N            (N = indicators reporting)\n"
        "momentum %    = 100 · #{ i : z_i(t) − z_i(t−3) < 0 } / N"
        "</pre>"
        "<p>Bands on the below-trend share: &lt;40% broad strength · 40–55 mixed · 55–70 "
        "weakening · ≥70 broad weakness.</p>"
        "<p><b>CFNAI diffusion.</b> FRED's CFNAI Diffusion Index summarises how broadly the ~85 "
        "CFNAI components are contributing; sustained readings below about −0.35 have historically "
        "accompanied recessions.</p>"
        "<p><b>Market-implied.</b> A panel of forward expectations priced into traded assets — 5y5y "
        "forward inflation, the 10y breakeven, the 10y real (TIPS) yield, the 10y–3m term spread, "
        "the high-yield spread, and the St. Louis stress index. For each we show the latest level, "
        "the 1-month change, and the trailing 5-year percentile. These are continuous and never "
        "revised — a complement to the lagged official data, shown as context.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/breadth.py</code>, "
        "<code>src/models/market_implied.py</code>, <code>src/ui/views/pulse.py</code>."
        "</div>",
        unsafe_allow_html=True,
    )


def _early_warning_section() -> None:
    _section_header("17. Early-warning ladder")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p>The Early Warning tab re-frames signals computed elsewhere as a single "
        "<b>lead-time-ordered ladder</b> answering one question: <i>how close is trouble?</i> "
        "Rungs are ordered from the earliest/slowest macro signal to the fastest financial one, so "
        "the depth of the lit rungs is the proximity read. Slow macro trouble climbs down from the "
        "top; fast financial trouble lights the bottom directly — one view, both kinds.</p>"
        '<pre style="background:#0d1117;padding:10px;color:#d4d4d0;font-size:12px;">'
        "rung                         fires when                         typical lead\n"
        "Yield-curve inversion        10y−3m ≤ 0                         ~12–18 mo\n"
        "Bank lending standards       SLOOS net tightening > 0          ~6–12 mo\n"
        "Housing permits              permits YoY < −2%                 ~6–9 mo\n"
        "Labor: Sahm rule             Sahm ≥ 0.5                        0–6 mo\n"
        "Recession ensemble           12-mo probability > 30%          12-mo model\n"
        "Labor breadth                ≥ 55% of indicators below trend  weeks–mo\n"
        "Financial conditions (NFCI)  NFCI > 0 (tighter than avg)      weeks\n"
        "Acute stress (VIX)           VIX ≥ 25                         days–wks"
        "</pre>"
        "<p><b>Proximity stage.</b> The deepest lit rung (lowest in the order) sets the headline: "
        "lit only near the top → <i>EARLY · distant</i>; into the middle → <i>BUILDING · "
        "approaching</i>; into the bottom → <i>ACUTE · near/here</i>.</p>"
        "<p><b>Early warning, not a forecast.</b> Lead times are stylized averages that vary "
        "widely; thresholds are judgmental (shown on each rung); and the sequence is <b>not</b> "
        "deterministic — 2020 lit the bottom rungs (conditions, vol) with no curve-led runway. "
        "Every input is already shown on another tab; this view only re-orders them by lead time, "
        "and adds no new data or model.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:4px;">'
        "Implementation: <code>src/models/early_warning.py</code>, "
        "<code>src/ui/views/early_warning.py</code>."
        "</div>",
        unsafe_allow_html=True,
    )


def _limitations() -> None:
    _section_header("18. Limitations")
    points = [
        (
            "In-sample headline · mitigated.",
            "The default reading is fit on the full sample. The walk-forward backtest "
            "(section 10) is computed at app startup and surfaces true out-of-sample "
            "Brier / AUC / reliability — use that for honest predictive performance, not "
            "the in-sample headline.",
        ),
        (
            "Coincident benchmark · separate.",
            "<b>Chauvet–Piger</b> is a <i>coincident</i> smoothed nowcast — it answers "
            "\"are we in recession now?\", not \"within 12 months?\" — so it is reported beside "
            "the ensemble as a benchmark and excluded from the average, keeping the headline a "
            "single-horizon (12-month) number.",
        ),
        (
            "Target definition · window, with one frozen exception.",
            "The dependent variable is <code>y_t = 1</code> if an NBER recession occurs in any "
            "month from <code>t+1</code> to <code>t+12</code> — the \"within 12 months\" reading "
            "the headline implies. The re-estimated models (NY Fed, Wright, BIC) are trained on "
            "this window target; the closed-form <b>Estrella–Mishkin</b> model keeps its frozen "
            "2006 point-in-time coefficients, so it sits on a slightly different basis within the "
            "ensemble. The point-in-time variant prints lower, spikier numbers and can miss "
            "short recessions; the Boston Fed has documented material dispersion between the two.",
        ),
        (
            "Feature selection · in-sample.",
            "BIC forward selection runs once on the full sample; the walk-forward backtest "
            "re-estimates coefficients out-of-sample but holds that feature set fixed. Features "
            "covering less than 80% of the target window (e.g. JOLTS from 2000) are dropped so "
            "short-history series don't shrink the estimation sample.",
        ),
        (
            "Bootstrap CI · approximate.",
            "The 90% interval resamples observations i.i.d.; because recession data is serially "
            "correlated, an i.i.d. bootstrap understates true uncertainty somewhat. Read the "
            "interval as indicative, not exact — a block bootstrap would widen it.",
        ),
        (
            "NBER dating · revised, not real-time.",
            "Recession shading and the training target use FRED <code>USREC</code> as known "
            "today; we don't store NBER vintages, and the NBER dates cycles with a long lag. As "
            "a real-time cross-check that doesn't depend on NBER, the Labor page also shows the "
            "Sahm Rule (FRED <code>SAHMREALTIME</code>), which uses only real-time unemployment "
            "and is not revised after release.",
        ),
        (
            "Model comparison · fully live.",
            "The model comparison is computed live from FRED on every rebuild — no "
            "hand-entered street estimates. The Chauvet–Piger reading is FRED's smoothed "
            "Markov-switching series (<code>RECPROUSM156N</code>); the others are probit "
            "specifications re-estimated on the FRED panel.",
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
    _section_header("19. Reproducibility")
    st.markdown(
        '<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;'
        f'color:{PALETTE["text_primary"]};">'
        "<p><b>Code.</b> Every module described above is in the open-source repo at "
        '<a href="https://github.com/SecondOrderEdge/Macro-Dashboard" '
        f'style="color:{PALETTE["accent"]};">github.com/SecondOrderEdge/Macro-Dashboard</a>. '
        "MIT licensed. No hidden constants — anything we assert here is in the code.</p>"
        "<p><b>Tests.</b> 63 deterministic pytest cases cover probability bounds, the "
        "four-model ensemble, BIC selection and sign constraints, walk-forward calibration, "
        "z-score normalisation, weight summation, spread calculation, inversion detection, "
        "composite banding, and the Market Probability Tracker CSV parser. Tests use synthetic "
        "or bundled data and never hit external APIs.</p>"
        "<p><b>Data attribution.</b> All macro time series © Federal Reserve Bank of "
        "St. Louis (FRED). High-yield OAS © ICE BofA. Recession dates © NBER. The S&P 500 "
        "is an index of S&P Dow Jones Indices LLC. Market-implied policy-rate distributions "
        "© Federal Reserve Bank of Atlanta (Market Probability Tracker), derived from CME "
        "Group options on SOFR futures.</p>"
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
