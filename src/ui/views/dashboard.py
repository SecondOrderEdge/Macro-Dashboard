"""Dashboard view — top-level page combining all three modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.composite import composite_risk
from src.models.lame import LAME
from src.models.yield_curve import YieldCurve
from src.ui.components import (
    add_recession_shading,
    apply_template,
    metric_card,
    sparkline_svg,
)
from src.ui.theme import PALETTE, risk_color


def render(
    current: dict,
    history: pd.DataFrame,
    lame: LAME,
    panel: pd.DataFrame,
    nber: pd.Series,
    market_prob: pd.DataFrame | None = None,
) -> None:
    yc = YieldCurve(panel)
    spreads = yc.spreads_history()
    lame_hist = lame.history()

    ensemble_now = float(current["ensemble"])
    lame_now = float(lame_hist.iloc[-1]) if not lame_hist.empty else float("nan")
    curve_now = (
        float(spreads["spread_10y3m"].dropna().iloc[-1])
        if "spread_10y3m" in spreads.columns and not spreads["spread_10y3m"].dropna().empty
        else float("nan")
    )
    composite = composite_risk(ensemble_now, lame_now, curve_now)

    _row_one(current, history, lame_hist, spreads)
    _row_policy_path(market_prob)
    _row_two(history, lame_hist, spreads, nber)
    _row_three(ensemble_now, lame_now, curve_now, composite, current)
    _row_weight_sensitivity(ensemble_now, lame_now, curve_now)
    _row_financial_conditions(panel, nber)
    _row_valuation_cape(nber)
    _row_four_analogues(history, lame_hist, spreads, nber, ensemble_now, lame_now, curve_now)


def _row_one(current: dict, history: pd.DataFrame, lame_hist: pd.Series, spreads: pd.DataFrame) -> None:
    cols = st.columns([2, 1, 1])

    with cols[0]:
        _recession_card(current, history)
        if st.button("Drill into Recession →", key="drill_recession"):
            st.session_state.pending_nav = "Recession"
            st.rerun()

    with cols[1]:
        _lame_card(lame_hist)
        if st.button("Drill into Labor →", key="drill_lame"):
            st.session_state.pending_nav = "Labor"
            st.rerun()

    with cols[2]:
        _curve_card(spreads)
        if st.button("Drill into Curve →", key="drill_curve"):
            st.session_state.pending_nav = "Yield Curve"
            st.rerun()


def _recession_card(current: dict, history: pd.DataFrame) -> None:
    ensemble_now = float(current["ensemble"])
    band = "LOW" if ensemble_now < 20 else "ELEVATED" if ensemble_now < 40 else "HIGH" if ensemble_now < 60 else "CRITICAL"
    color = risk_color(band)

    spark = ""
    if "ensemble" in history.columns:
        spark = sparkline_svg(history["ensemble"].dropna().tail(180).values, color=color, width=320)

    sub_rows = ""
    for name, prob in current["submodels"].items():
        sub_color = PALETTE["submodel"].get(name, PALETTE["text_muted"])
        label_pretty = name
        sub_rows += (
            f'<div class="submodel-row"><span class="name" style="color:{sub_color};">{label_pretty}</span>'
            f'<span class="value">{prob:.0f}%</span></div>'
        )

    html = f"""
<div class="panel" style="height:100%;">
  <div class="panel-header">
    <span>Recession Probability · 12-month forward</span>
    <span class="risk-badge" style="color:{color};">{band}</span>
  </div>
  <div class="panel-body">
    <div style="display:flex;align-items:flex-start;gap:24px;">
      <div>
        <div class="metric-big data-font" style="color:{color};">{ensemble_now:.0f}<span class="metric-unit">%</span></div>
        <div class="metric-sub">4-model ensemble</div>
        <div style="margin-top:10px;">{spark}</div>
      </div>
      <div style="flex:1;min-width:0;">
        {sub_rows}
      </div>
    </div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def _lame_card(lame_hist: pd.Series) -> None:
    if lame_hist.empty:
        st.markdown(metric_card("Labor Composite", "—", "σ"), unsafe_allow_html=True)
        return
    val = float(lame_hist.iloc[-1])
    color = (
        PALETTE["risk_critical"] if val < -1.0
        else PALETTE["risk_high"] if val < -0.5
        else PALETTE["risk_elevated"] if val < 0.5
        else PALETTE["risk_low"]
    )
    badge = "CONTRACTIONARY" if val < -0.5 else "FIRM" if val > 0.5 else "NEUTRAL"
    spark = sparkline_svg(lame_hist.tail(120).values, color=color)
    st.markdown(
        metric_card(
            label="Labor Composite",
            value=f"{val:+.2f}",
            unit="σ",
            risk_color_hex=color,
            sparkline_html=spark,
            badge=badge,
            subline="10-indicator composite",
        ),
        unsafe_allow_html=True,
    )


def _curve_card(spreads: pd.DataFrame) -> None:
    if "spread_10y3m" not in spreads.columns:
        st.markdown(metric_card("10Y − 3M", "—", "pp"), unsafe_allow_html=True)
        return
    s = spreads["spread_10y3m"].dropna()
    if s.empty:
        st.markdown(metric_card("10Y − 3M", "—", "pp"), unsafe_allow_html=True)
        return
    val = float(s.iloc[-1])
    inverted = val < 0
    color = PALETTE["risk_critical"] if inverted else PALETTE["risk_low"]
    badge = "INVERTED" if inverted else "POSITIVE"
    spark = sparkline_svg(s.tail(252).values, color=color)
    st.markdown(
        metric_card(
            label="10Y − 3M Spread",
            value=f"{val:+.2f}",
            unit="pp",
            risk_color_hex=color,
            sparkline_html=spark,
            badge=badge,
            subline="curve recession signal",
        ),
        unsafe_allow_html=True,
    )


def _row_policy_path(market_prob: pd.DataFrame | None) -> None:
    """Headline card: market-implied FOMC policy path (Atlanta Fed tracker)."""
    if market_prob is None or market_prob.empty:
        return

    from src.data.market_probability import (
        directional_probs,
        latest_snapshot,
        rate_path,
    )

    rp = rate_path(market_prob)
    if rp.empty:
        return
    snap = latest_snapshot(market_prob)["snapshot_date"].max()
    front = rp.index.min()
    front_rate = float(rp.loc[front, "mean"]) / 100.0

    dirs = directional_probs(market_prob)
    dirs = dirs[dirs["snapshot_date"] == snap].sort_values("meeting_date") if not dirs.empty else dirs
    hike = cut = float("nan")
    if not dirs.empty:
        hike = float(dirs.iloc[0]["prob_hike"]) if pd.notna(dirs.iloc[0]["prob_hike"]) else float("nan")
        cut = float(dirs.iloc[0]["prob_cut"]) if pd.notna(dirs.iloc[0]["prob_cut"]) else float("nan")
    if np.isfinite(hike) and np.isfinite(cut):
        if hike > cut + 5:
            lean, lean_color = "HIKES PRICED", PALETTE["risk_elevated"]
        elif cut > hike + 5:
            lean, lean_color = "CUTS PRICED", PALETTE["risk_low"]
        else:
            lean, lean_color = "ON HOLD", PALETTE["text_muted"]
    else:
        lean, lean_color = "—", PALETTE["text_muted"]

    st.markdown(
        '<div class="label-small" style="margin-top:24px;">'
        'Market-implied policy path · Atlanta Fed Market Probability Tracker</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])
    with left:
        hike_s = f"{hike:.0f}%" if np.isfinite(hike) else "—"
        cut_s = f"{cut:.0f}%" if np.isfinite(cut) else "—"
        st.markdown(
            '<div class="panel" style="height:100%;">'
            '<div class="panel-header"><span>Implied rate · next meeting</span>'
            f'<span class="risk-badge" style="color:{lean_color};">{lean}</span></div>'
            '<div class="panel-body">'
            f'<div class="metric-big data-font" style="color:{PALETTE["accent"]};">{front_rate:.2f}<span class="metric-unit">%</span></div>'
            f'<div class="metric-sub">mean · {front.strftime("%b %Y")} meeting</div>'
            f'<div style="margin-top:10px;">'
            f'<div class="submodel-row"><span class="name" style="color:{PALETTE["risk_elevated"]};">Hike</span><span class="value">{hike_s}</span></div>'
            f'<div class="submodel-row"><span class="name" style="color:{PALETTE["risk_low"]};">Cut</span><span class="value">{cut_s}</span></div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Drill into Policy Path →", key="drill_policy"):
            st.session_state.pending_nav = "Policy Path"
            st.rerun()

    with right:
        band = rp.dropna(subset=["p25", "p75"])
        fig = go.Figure()
        if not band.empty:
            fig.add_trace(
                go.Scatter(
                    x=band.index, y=(band["p75"] / 100).values, mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=band.index, y=(band["p25"] / 100).values, mode="lines",
                    line=dict(width=0), fill="tonexty", fillcolor=_cape_fade(PALETTE["accent"], 0.12),
                    name="25th–75th pct", hoverinfo="skip",
                )
            )
        mean = rp.dropna(subset=["mean"])
        fig.add_trace(
            go.Scatter(
                x=mean.index, y=(mean["mean"] / 100).values, mode="lines+markers",
                line=dict(color=PALETTE["accent"], width=1.8), marker=dict(size=4, color=PALETTE["accent"]),
                name="Mean",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>mean</extra>",
            )
        )
        fig.update_yaxes(title="Implied rate (%)")
        apply_template(fig, height=260, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)


def _row_two(
    history: pd.DataFrame,
    lame_hist: pd.Series,
    spreads: pd.DataFrame,
    nber: pd.Series,
) -> None:
    st.markdown(
        '<div class="label-small" style="margin-top:24px;">Three lenses · last 30 years</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=30)

    if "ensemble" in history.columns:
        s = history["ensemble"].dropna()
        s = s.loc[s.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines", name="Recession ensemble (%)",
                line=dict(color=PALETTE["accent"], width=1.6),
                fill="tozeroy", fillcolor="rgba(212,165,116,0.10)",
                yaxis="y1",
                hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra>Ensemble</extra>",
            )
        )

    if not lame_hist.empty:
        s = lame_hist.loc[lame_hist.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines", name="Labor (σ)",
                line=dict(color=PALETTE["submodel"]["labor"], width=1.2),
                yaxis="y2",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}σ<extra>Labor</extra>",
            )
        )

    if "spread_10y3m" in spreads.columns:
        s = spreads["spread_10y3m"].dropna()
        s = s.loc[s.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines", name="10Y−3M (pp)",
                line=dict(color=PALETTE["submodel"]["yield_curve"], width=1.0),
                yaxis="y2",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}pp<extra>10Y−3M</extra>",
            )
        )

    add_recession_shading(fig, nber.loc[nber.index >= cutoff])
    fig.update_layout(
        yaxis=dict(title="Recession probability (%)", side="left", range=[0, 100]),
        yaxis2=dict(title="Labor (σ) · spread (pp)", overlaying="y", side="right", showgrid=False),
    )
    apply_template(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


def _row_three(ensemble_now, lame_now, curve_now, composite, current) -> None:
    left, right = st.columns(2)

    with left:
        rows = [
            ("Recession ensemble (50%)", f"{ensemble_now:.0f}%", PALETTE["accent"]),
            ("Labor-risk (25%)", f"{composite['contributions'].get('lame', 0)*4:.0f} → {composite['contributions'].get('lame', 0):.0f} pts", PALETTE["submodel"]["labor"]),
            ("Curve-risk (25%)", f"{composite['contributions'].get('curve', 0)*4:.0f} → {composite['contributions'].get('curve', 0):.0f} pts", PALETTE["submodel"]["yield_curve"]),
            ("Composite", f"{composite['composite']} · {composite['band']}", risk_color(composite["band"])),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{label}</span>'
            f'<span class="value" style="color:{color};">{value}</span></div>'
            for label, value, color in rows
        )
        st.markdown(
            '<div class="panel"><div class="panel-header"><span>Composite construction</span></div>'
            f'<div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        text = _todays_read(ensemble_now, lame_now, curve_now, current)
        st.markdown(
            '<div class="panel"><div class="panel-header"><span>Today\'s read</span></div>'
            f'<div class="panel-body" style="font-family:\'Fraunces\',serif;font-size:15px;line-height:1.7;color:{PALETTE["text_primary"]};">{text}</div></div>',
            unsafe_allow_html=True,
        )


def _todays_read(ensemble_now: float, lame_now: float, curve_now: float, current: dict) -> str:
    parts: list[str] = []
    if np.isfinite(ensemble_now):
        parts.append(
            f"The ensemble puts <b>12-month recession probability at {ensemble_now:.0f}%</b>."
        )
    if np.isfinite(curve_now):
        if curve_now < 0:
            parts.append(
                f"The 10Y−3M curve is <b>inverted at {curve_now:+.2f}pp</b>, "
                "the most reliable single recession signal we have."
            )
        else:
            parts.append(
                f"The 10Y−3M curve is positive at {curve_now:+.2f}pp, not flagging recession on its own."
            )
    if np.isfinite(lame_now):
        if lame_now < -0.5:
            parts.append(
                f"Labor is contractionary at {lame_now:+.2f}σ — "
                "the soft-landing thesis is on thinner ice."
            )
        else:
            parts.append(
                f"Labor reads {lame_now:+.2f}σ, broadly consistent with continued expansion."
            )
    if not parts:
        parts.append("Not enough data is available to form a reading.")
    return " ".join(parts)


# --------------------------------------------------------- historical analogues


def _row_four_analogues(
    history: pd.DataFrame,
    lame_hist: pd.Series,
    spreads: pd.DataFrame,
    nber: pd.Series,
    ensemble_now: float,
    lame_now: float,
    curve_now: float,
) -> None:
    """Top-N historical dates most similar to today, plus what came next.

    Similarity is Euclidean distance in (ensemble, labor, 10Y-3M) space after
    each series is z-scored against its own history — so all three are on a
    comparable scale. For each analogue date we look up the realized outcome
    in the following 12 months: peak ensemble probability and whether any
    NBER recession month occurred in that window.
    """
    st.markdown(
        '<div class="label-small" style="margin-top:24px;">Historical analogues · '
        'when the dashboard last looked like today</div>',
        unsafe_allow_html=True,
    )

    ensemble_hist = history["ensemble"].dropna() if "ensemble" in history.columns else pd.Series(dtype=float)
    curve_hist = spreads["spread_10y3m"].dropna() if "spread_10y3m" in spreads.columns else pd.Series(dtype=float)

    # Align all three on a common monthly index.
    ensemble_m = ensemble_hist.copy()
    ensemble_m.index = pd.DatetimeIndex(ensemble_m.index).to_period("M").to_timestamp()
    labor_m = lame_hist.copy()
    labor_m.index = pd.DatetimeIndex(labor_m.index).to_period("M").to_timestamp()
    curve_m = curve_hist.resample("ME").last() if not curve_hist.empty else curve_hist
    curve_m.index = pd.DatetimeIndex(curve_m.index).to_period("M").to_timestamp()

    df = pd.concat(
        [ensemble_m.rename("ensemble"), labor_m.rename("labor"), curve_m.rename("curve_10y3m")],
        axis=1,
    ).dropna()

    if df.empty:
        st.info("Not enough overlapping history for analogue search.")
        return

    # Standardise each column against its own full sample, then compute the
    # Euclidean distance to today's standardised triplet.
    means = df.mean()
    stds = df.std().replace(0, 1.0)
    standardised = (df - means) / stds
    today_std = pd.Series(
        {
            "ensemble": (ensemble_now - means["ensemble"]) / stds["ensemble"],
            "labor": (lame_now - means["labor"]) / stds["labor"],
            "curve_10y3m": (curve_now - means["curve_10y3m"]) / stds["curve_10y3m"],
        }
    )
    distances = np.sqrt(((standardised - today_std) ** 2).sum(axis=1))

    # Exclude dates within 24 months of today, plus the today row itself, so
    # we get genuinely different historical regimes.
    excluded = distances.index >= (distances.index.max() - pd.DateOffset(months=24))
    candidates = distances[~excluded].sort_values().head(5)
    if candidates.empty:
        st.info("Not enough non-recent history to surface analogues.")
        return

    # Outcomes: max ensemble in next 12m, did NBER recession occur?
    nber_monthly = nber.copy()
    nber_monthly.index = pd.DatetimeIndex(nber_monthly.index).to_period("M").to_timestamp()

    rows = []
    for ts, dist in candidates.items():
        snap = df.loc[ts]
        end_ts = ts + pd.DateOffset(months=12)
        next12_ensemble = ensemble_m.loc[(ensemble_m.index > ts) & (ensemble_m.index <= end_ts)]
        peak = float(next12_ensemble.max()) if not next12_ensemble.empty else float("nan")
        nber_window = nber_monthly.loc[(nber_monthly.index > ts) & (nber_monthly.index <= end_ts)]
        rec_in_12m = bool(nber_window.any()) if not nber_window.empty else False
        rows.append(
            {
                "date": ts.strftime("%b %Y"),
                "distance": f"{dist:.2f}",
                "ensemble": f"{snap['ensemble']:.0f}%",
                "labor (σ)": f"{snap['labor']:+.2f}",
                "10Y−3M (pp)": f"{snap['curve_10y3m']:+.2f}",
                "peak ensemble next 12m": f"{peak:.0f}%" if np.isfinite(peak) else "—",
                "NBER recession in next 12m": "YES" if rec_in_12m else "no",
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Verdict
    hit_count = sum(1 for r in rows if r["NBER recession in next 12m"] == "YES")
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        f"Of the five closest historical analogues to today's reading, "
        f"<b>{hit_count} of 5</b> were followed by an NBER recession within twelve months. "
        "Analogue distance is Euclidean in standardised (ensemble probability, labor σ, 10Y−3M spread) space "
        "and excludes dates within the last 24 months so the comparison surfaces genuinely earlier regimes."
        "</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------- weight sensitivity


def _row_weight_sensitivity(ensemble_now: float, lame_now: float, curve_now: float) -> None:
    """Interactive sliders that let the user re-weight the composite."""
    from src.models.composite import composite_risk, lame_to_risk, curve_to_risk

    st.markdown(
        '<div class="label-small" style="margin-top:24px;">'
        'Composite weight sensitivity · re-weight on the fly</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:12px;margin-bottom:8px;">'
        "The default 50/25/25 blend (recession / labor / curve) is judgmental — a reasonable "
        "analyst could weight the curve more heavily, or rely on labor alone. Drag the sliders "
        "to see how the composite would change. Weights are renormalised to 100% automatically."
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        w_ens = st.slider("Recession ensemble", 0, 100, 50, 5, key="w_ens")
    with cols[1]:
        w_lame = st.slider("Labor", 0, 100, 25, 5, key="w_lame")
    with cols[2]:
        w_curve = st.slider("Yield curve", 0, 100, 25, 5, key="w_curve")

    total = max(w_ens + w_lame + w_curve, 1)
    ne, nl, nc = w_ens / total, w_lame / total, w_curve / total

    ensemble_risk = float(np.clip(ensemble_now, 0, 100)) if np.isfinite(ensemble_now) else 0.0
    lame_risk_v = lame_to_risk(lame_now) if np.isfinite(lame_now) else 0.0
    curve_risk_v = curve_to_risk(curve_now) if np.isfinite(curve_now) else 0.0

    custom_score = ne * ensemble_risk + nl * lame_risk_v + nc * curve_risk_v
    default_score = composite_risk(ensemble_now, lame_now, curve_now)["composite"]

    band = (
        "LOW" if custom_score < 20 else
        "ELEVATED" if custom_score < 40 else
        "HIGH" if custom_score < 60 else
        "CRITICAL"
    )
    color = risk_color(band)

    with cols[3]:
        st.markdown(
            '<div class="panel" style="height:100%;">'
            '<div class="panel-header"><span>Custom-weighted composite</span></div>'
            '<div class="panel-body">'
            f'<div style="display:flex;align-items:baseline;gap:24px;">'
            f'<div class="metric-big data-font" style="color:{color};">{custom_score:.0f}</div>'
            f'<div>'
            f'<div class="risk-badge" style="color:{color};">{band}</div>'
            f'<div class="metric-sub">default (50/25/25): {default_score}</div>'
            f'</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # Show contribution breakdown
    rows = [
        ("Ensemble", f"{ne:.0%}", f"{ensemble_risk:.0f}", f"{ne * ensemble_risk:.1f}", PALETTE["accent"]),
        ("Labor",    f"{nl:.0%}", f"{lame_risk_v:.0f}",   f"{nl * lame_risk_v:.1f}",   PALETTE["submodel"]["labor"]),
        ("Curve",    f"{nc:.0%}", f"{curve_risk_v:.0f}",  f"{nc * curve_risk_v:.1f}",  PALETTE["submodel"]["yield_curve"]),
    ]
    body = "".join(
        f'<div class="submodel-row">'
        f'<span class="name" style="color:{c};">{name}</span>'
        f'<span class="value">{wt} weight · {sc} risk · {pts} pts</span>'
        '</div>'
        for name, wt, sc, pts, c in rows
    )
    st.markdown(
        '<div class="panel"><div class="panel-header"><span>Contribution breakdown</span></div>'
        f'<div class="panel-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------- valuation (CAPE)


def _row_valuation_cape(nber: pd.Series) -> None:
    """CAPE ratio panel — equity-market valuation context.

    Framed deliberately as a *valuation* indicator, not a recession signal.
    CAPE has a poor short-horizon recession-prediction record but a strong
    long-horizon equity-return record, and extreme readings materially
    raise the conditional drawdown if recession does arrive.
    """
    from src.data.cape import fetch_cape_history, cape_summary, cape_band

    cape = fetch_cape_history()
    st.markdown(
        '<div class="label-small" style="margin-top:24px;">'
        'Valuation context · Shiller CAPE ratio</div>',
        unsafe_allow_html=True,
    )

    if cape is None or cape.empty:
        log = (cape.attrs.get("fetch_log") if cape is not None else None) or []
        log_html = ""
        if log:
            items = "".join(f"<li>{ln}</li>" for ln in log)
            log_html = (
                f'<details style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
                "<summary>fetch attempts</summary>"
                f'<ul style="margin:6px 0 0 18px;padding:0;">{items}</ul>'
                "</details>"
            )
        st.markdown(
            f'<div class="panel"><div class="panel-body" style="font-size:12px;color:{PALETTE["text_muted"]};">'
            "CAPE data temporarily unavailable. Primary source: "
            '<a href="https://github.com/datasets/s-and-p-500" '
            f'style="color:{PALETTE["accent"]};">datasets/s-and-p-500</a>'
            " · fallback: "
            '<a href="http://www.econ.yale.edu/~shiller/data.htm" '
            f'style="color:{PALETTE["accent"]};">Robert Shiller / Yale</a>.'
            f"{log_html}"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    summary = cape_summary(cape)
    if not summary:
        return

    today = summary["today"]
    pct = summary["modern_percentile"]
    median = summary["modern_median"]
    yr_ago = summary["one_year_ago"]
    label, severity = cape_band(pct)
    color = {
        "low":      PALETTE["risk_low"],
        "elevated": PALETTE["risk_elevated"],
        "high":     PALETTE["risk_high"],
        "critical": PALETTE["risk_critical"],
    }[severity]

    left, right = st.columns([1, 3])

    with left:
        spark = sparkline_svg(cape.tail(300).values, color=color)
        months_stale = (pd.Timestamp.today() - summary["as_of"]).days // 30
        staleness = "" if months_stale <= 2 else f" · {months_stale}mo stale"
        subline = (
            f"{pct:.0f}th percentile since 1950 · "
            f"as of {summary['as_of'].strftime('%b %Y')}{staleness}"
        )
        st.markdown(
            metric_card(
                label="Shiller CAPE",
                value=f"{today:.1f}",
                unit="×",
                risk_color_hex=color,
                sparkline_html=spark,
                badge=label,
                subline=subline,
            ),
            unsafe_allow_html=True,
        )

        rows = [
            ("Today",                f"{today:.1f}×"),
            ("12 months ago",        f"{yr_ago:.1f}×" if np.isfinite(yr_ago) else "—"),
            ("Median (post-1950)",   f"{median:.1f}×"),
            ("2000 dot-com peak",    f"{summary['peaks']['2000 dot-com peak']:.1f}×" if np.isfinite(summary['peaks'].get('2000 dot-com peak', float('nan'))) else "—"),
            ("2007 peak",            f"{summary['peaks']['2007 peak']:.1f}×" if np.isfinite(summary['peaks'].get('2007 peak', float('nan'))) else "—"),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{lbl}</span>'
            f'<span class="value">{val}</span></div>'
            for lbl, val in rows
        )
        st.markdown(
            f'<div class="panel"><div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        modern = cape.loc[cape.index >= "1950-01-01"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=modern.index, y=modern.values, mode="lines",
                line=dict(color=color, width=1.2),
                fill="tozeroy", fillcolor=_cape_fade(color, 0.08),
                name="CAPE", showlegend=False,
                hovertemplate="%{x|%b %Y}<br>%{y:.1f}×<extra></extra>",
            )
        )
        fig.add_hline(
            y=median, line=dict(color=PALETTE["text_tiny"], width=1, dash="dot"),
            annotation_text=f"median {median:.0f}", annotation_position="bottom right",
            annotation_font=dict(color=PALETTE["text_tiny"], size=10),
        )
        # Mark famous peaks
        for label_, dt in [("dot-com 2000", "2000-01-01"), ("2007 peak", "2007-10-01"), ("1929", "1929-09-01")]:
            ts = pd.Timestamp(dt)
            if ts in cape.index or (cape.index.min() <= ts <= cape.index.max()):
                val = float(cape.asof(ts)) if hasattr(cape, "asof") else float("nan")
                if np.isfinite(val):
                    fig.add_annotation(
                        x=ts, y=val,
                        text=label_,
                        showarrow=True, arrowhead=2, ax=0, ay=-22,
                        arrowcolor=PALETTE["text_muted"], arrowwidth=1,
                        font=dict(color=PALETTE["text_muted"], size=9),
                        bgcolor="rgba(10,13,18,0.6)", borderpad=2,
                    )
        add_recession_shading(fig, nber.loc[nber.index >= "1950-01-01"])
        fig.update_yaxes(title="CAPE multiple")
        apply_template(fig, height=300, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Interpretation panel --------------------------------------------
    if pct >= 85:
        verdict = (
            "Equity valuations sit in the <b>top 15%</b> of their post-1950 distribution. "
            "CAPE at this level has historically preceded multi-year equity underperformance "
            "rather than a near-term recession; the practical read is that the conditional "
            "drawdown <i>if</i> a recession arrives is materially larger than it would be at "
            "average valuations."
        )
    elif pct >= 60:
        verdict = (
            "Valuations are <b>above average but not extreme</b>. CAPE is not a useful "
            "short-horizon recession signal at this level, but worth monitoring as a context "
            "factor for the magnitude of potential drawdowns."
        )
    elif pct >= 25:
        verdict = (
            "Valuations are <b>around their post-1950 median</b>. Forward equity returns at "
            "this CAPE level have historically been roughly average; valuation is not signalling "
            "anything notable in either direction."
        )
    else:
        verdict = (
            "Valuations are in the <b>bottom quartile</b> of post-1950 history. Forward 10-year "
            "real equity returns from these levels have historically been strong; a recession "
            "would still hurt but the conditional drawdown floor is structurally less alarming."
        )

    st.markdown(
        f'<div class="panel"><div class="panel-header"><span>How to read this</span></div>'
        f'<div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        "<p>CAPE divides today's S&P 500 price by trailing <b>10-year inflation-adjusted "
        "earnings</b>. It's a valuation gauge, not a recession indicator — CAPE was elevated "
        "for most of 2014–2024 without a recession arriving. We surface it here because "
        "valuation determines the <i>magnitude</i> of potential equity damage if a recession "
        "does arrive.</p>"
        f"<p>{verdict}</p>"
        f'<p style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:8px;">'
        'Sources: <a href="http://www.econ.yale.edu/~shiller/data.htm" '
        f'style="color:{PALETTE["accent"]};">Robert Shiller / Yale</a> (primary, fresh) and '
        '<a href="https://github.com/datasets/s-and-p-500" '
        f'style="color:{PALETTE["accent"]};">datasets/s-and-p-500</a> (fallback, long history). '
        f'<span style="color:{PALETTE["text_tiny"]};">data path: {summary.get("source", "?")}</span>'
        "</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _cape_fade(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------- financial conditions


def _row_financial_conditions(panel: pd.DataFrame, nber: pd.Series) -> None:
    """Four parallel display indicators: NFCI / ANFCI / STLFSI4 / CFNAIMA3.

    Framed as a *real-time financial+activity nowcast* row — separate from
    the recession ensemble (no probit input). These complement the Sahm Rule
    (labor-only nowcast) with broader financial and activity coverage.
    """
    from src.models.conditions import (
        nfci, anfci, stlfsi, cfnai_3ma,
        stress_band, cfnai_band,
    )

    nfci_s    = nfci(panel)
    anfci_s   = anfci(panel)
    stlfsi_s  = stlfsi(panel)
    cfnai_s   = cfnai_3ma(panel)

    if nfci_s.empty and anfci_s.empty and stlfsi_s.empty and cfnai_s.empty:
        return  # nothing to render

    st.markdown(
        '<div class="label-small" style="margin-top:24px;">'
        'Financial conditions & activity · real-time nowcasts</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)

    def _stress_card(col, label, series, source_note):
        if series.empty:
            with col:
                st.markdown(metric_card(label, "—", ""), unsafe_allow_html=True)
            return
        latest = float(series.iloc[-1])
        as_of = series.index[-1].strftime("%b %Y")
        band, sev = stress_band(latest)
        color = _sev_color(sev)
        spark = sparkline_svg(series.tail(260).values, color=color)
        with col:
            st.markdown(
                metric_card(
                    label=label,
                    value=f"{latest:+.2f}",
                    unit="σ",
                    risk_color_hex=color,
                    sparkline_html=spark,
                    badge=band,
                    subline=f"{source_note} · {as_of}",
                ),
                unsafe_allow_html=True,
            )

    def _activity_card(col, label, series, source_note):
        if series.empty:
            with col:
                st.markdown(metric_card(label, "—", ""), unsafe_allow_html=True)
            return
        latest = float(series.iloc[-1])
        as_of = series.index[-1].strftime("%b %Y")
        band, sev = cfnai_band(latest)
        color = _sev_color(sev)
        spark = sparkline_svg(series.tail(120).values, color=color)
        with col:
            st.markdown(
                metric_card(
                    label=label,
                    value=f"{latest:+.2f}",
                    unit="",
                    risk_color_hex=color,
                    sparkline_html=spark,
                    badge=band,
                    subline=f"{source_note} · {as_of}",
                ),
                unsafe_allow_html=True,
            )

    _stress_card  (cols[0], "Financial Conditions",        nfci_s,    "Chicago Fed · NFCI")
    _stress_card  (cols[1], "Adjusted Financial Conditions", anfci_s, "Chicago Fed · ANFCI")
    _stress_card  (cols[2], "Financial Stress",            stlfsi_s,  "St Louis Fed · STLFSI4")
    _activity_card(cols[3], "Economic Activity (3-mo)",    cfnai_s,   "Chicago Fed · CFNAI")

    # Combined chart with dual y-axis (stress on left, activity on right).
    fig = go.Figure()
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=25)

    stress_pairs = [
        ("NFCI",   nfci_s,   PALETTE["accent"]),
        ("ANFCI",  anfci_s,  PALETTE["submodel"]["labor"]),
        ("STLFSI", stlfsi_s, PALETTE["submodel"]["sentiment"]),
    ]
    for name, s, color in stress_pairs:
        if s.empty:
            continue
        s_window = s.loc[s.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=s_window.index, y=s_window.values, mode="lines",
                line=dict(color=color, width=1.2),
                name=name, yaxis="y1",
                hovertemplate=f"%{{x|%b %Y}}<br>%{{y:+.2f}}σ<extra>{name}</extra>",
            )
        )

    if not cfnai_s.empty:
        s_window = cfnai_s.loc[cfnai_s.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=s_window.index, y=s_window.values, mode="lines",
                line=dict(color=PALETTE["submodel"]["yield_curve"], width=1.4, dash="dot"),
                name="CFNAI 3-mo (right axis)", yaxis="y2",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}<extra>CFNAI 3-mo</extra>",
            )
        )

    # Threshold reference lines
    fig.add_hline(y=0, line=dict(color="#3d4754", width=1, dash="dot"))
    fig.add_hline(
        y=1.0, line=dict(color=PALETTE["risk_high"], width=1, dash="dash"),
        annotation_text="stress threshold", annotation_position="bottom right",
        annotation_font=dict(color=PALETTE["risk_high"], size=9),
    )

    add_recession_shading(fig, nber.loc[nber.index >= cutoff])
    fig.update_layout(
        yaxis=dict(title="Financial conditions (σ)", side="left"),
        yaxis2=dict(title="CFNAI 3-mo", overlaying="y", side="right", showgrid=False),
    )
    apply_template(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

    # Plain-English read
    _financial_conditions_read(nfci_s, cfnai_s)


def _financial_conditions_read(nfci_s: pd.Series, cfnai_s: pd.Series) -> None:
    from src.models.conditions import stress_band, cfnai_band

    parts: list[str] = []
    if not nfci_s.empty:
        latest = float(nfci_s.iloc[-1])
        band, _ = stress_band(latest)
        if latest > 1.0:
            parts.append(
                f"<b>Financial conditions are stressed</b> — NFCI is {latest:+.2f}σ, "
                "above the +1.0 historical stress threshold. Credit spreads, equity "
                "volatility, and funding markets are simultaneously tighter than average."
            )
        elif latest > 0:
            parts.append(
                f"Financial conditions are <b>modestly tighter</b> than average "
                f"(NFCI {latest:+.2f}σ). Not at stress levels."
            )
        else:
            parts.append(
                f"Financial conditions are <b>easier than average</b> (NFCI {latest:+.2f}σ). "
                "Credit-driven recession scenarios are not gathering momentum."
            )

    if not cfnai_s.empty:
        cval = float(cfnai_s.iloc[-1])
        cband, _ = cfnai_band(cval)
        if cval < -0.7:
            parts.append(
                f"The Chicago Fed activity composite (CFNAI 3-mo) reads {cval:+.2f} — "
                "<b>below its -0.7 recession-signal threshold</b>. This is the canonical "
                "real-time activity contraction signal."
            )
        elif cval < 0:
            parts.append(
                f"CFNAI 3-mo is {cval:+.2f}, slightly below trend but above the -0.7 "
                "recession threshold."
            )
        else:
            parts.append(
                f"CFNAI 3-mo is {cval:+.2f}, above its long-run trend — activity is "
                "running at or above potential growth."
            )

    if not parts:
        return
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        + " ".join(parts) +
        f'<p style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:10px;">'
        "These indicators are <i>not</i> inputs to the recession ensemble — they're "
        "parallel real-time nowcasts. NFCI/ANFCI/STLFSI4 sources: Chicago Fed and "
        "St Louis Fed via FRED. CFNAI source: Chicago Fed via FRED."
        "</p></div></div>",
        unsafe_allow_html=True,
    )


def _sev_color(severity: str) -> str:
    return {
        "low":      PALETTE["risk_low"],
        "elevated": PALETTE["risk_elevated"],
        "high":     PALETTE["risk_high"],
        "critical": PALETTE["risk_critical"],
    }.get(severity, PALETTE["text_muted"])
