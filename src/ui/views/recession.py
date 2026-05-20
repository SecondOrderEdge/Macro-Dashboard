"""Recession page — five-model academic probit ensemble.

Three tabs surface the same analytics the weekly investment-committee email
reports: The Reading (headline + history + drivers), Under the Hood (the five
models, comparison, bootstrap CI, indicator percentiles), and Watchlist
(trigger levels, adverse scenario, what-would-change-our-view).

The report dict is produced by :mod:`src.models.recession_probit`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from src.models.recession_probit import (
    THRESHOLD_ELEVATED,
    THRESHOLD_WARNING,
    feature_label,
    scenario_probability,
)
from src.ui.components import (
    add_recession_shading,
    apply_template,
    metric_card,
    sparkline_svg,
)
from src.ui.theme import PALETTE


_TAB_STYLES = {
    "container": {"background-color": "#0a0d12", "padding": "0", "margin-top": "8px"},
    "nav-link": {
        "font-size": "10px",
        "letter-spacing": "0.15em",
        "text-transform": "uppercase",
        "color": "#6b7280",
        "background-color": "transparent",
        "padding": "8px 16px",
    },
    "nav-link-selected": {
        "color": "#d4a574",
        "background-color": "transparent",
        "border-bottom": "2px solid #d4a574",
    },
}


def render(report: dict | None, nber: pd.Series) -> None:
    if not report:
        st.error("The recession probit ensemble is unavailable (no report was built).")
        return
    if "error" in report:
        st.error(f"Recession probit ensemble failed to build: {report['error']}")
        return

    selected = option_menu(
        menu_title=None,
        options=["The Reading", "Under the Hood", "Watchlist", "Scenario"],
        icons=["graph-up", "sliders", "bullseye", "toggles"],
        orientation="horizontal",
        default_index=0,
        key="recession_tab",
        styles=_TAB_STYLES,
    )

    if selected == "The Reading":
        _render_reading(report, nber)
    elif selected == "Under the Hood":
        _render_under_hood(report)
    elif selected == "Watchlist":
        _render_watchlist(report)
    else:
        _render_scenario(report)


# --------------------------------------------------------------- shared helpers


def _prob_color(p: float) -> str:
    if not np.isfinite(p):
        return PALETTE["text_muted"]
    if p < 20:
        return PALETTE["risk_low"]
    if p < THRESHOLD_WARNING:
        return PALETTE["risk_elevated"]
    if p < THRESHOLD_ELEVATED:
        return PALETTE["risk_high"]
    return PALETTE["risk_critical"]


def _consistent_with(p: float) -> str:
    if p < 20:
        return "expansion-phase conditions"
    if p < THRESHOLD_WARNING:
        return "late-cycle conditions worth monitoring"
    if p < THRESHOLD_ELEVATED:
        return "elevated, transition-phase risk"
    return "conditions that historically precede contraction"


def _fade(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _direction(report: dict) -> str:
    ta = report.get("trend_attribution") or {}
    change = ta.get("prob_change_pp")
    if change is None:
        return "stable"
    if change > 2:
        return "rising"
    if change < -2:
        return "falling"
    return "stable"


# ----------------------------------------------------------------- The Reading


def _render_reading(report: dict, nber: pd.Series) -> None:
    ens = report["ensemble_probability"]
    color = _prob_color(ens)
    lo, hi = report.get("ci_lower"), report.get("ci_upper")
    probs = report["model_probabilities"]
    p_lo, p_hi = min(probs.values()), max(probs.values())

    ci_txt = (
        f"90% CI {lo:.0f}–{hi:.0f}%" if lo is not None and hi is not None else "CI unavailable"
    )
    spark = sparkline_svg(
        report["ensemble_history"].tail(60).values, color=color, width=240, height=44
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            metric_card(
                label="12-month recession probability · 5-model ensemble",
                value=f"{ens:.0f}",
                unit="%",
                risk_color_hex=color,
                sparkline_html=spark,
                badge=report["signal"],
                subline=f"{ci_txt} · models range {p_lo:.0f}–{p_hi:.0f}% · consensus {report['consensus']}",
            ),
            unsafe_allow_html=True,
        )
    with right:
        meta = report["model_metadata"]
        rows = [
            ("Direction (24m)", _direction(report)),
            ("Data through", report["data_through"]),
            ("Training window", f"{meta['training_start']} → {meta['training_end']}"),
            ("Pseudo R²", f"{meta['pseudo_r2']:.3f}"),
            ("BIC features", f"{len(report['bic_selected_features'])}"),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{k}</span>'
            f'<span class="value">{v}</span></div>'
            for k, v in rows
        )
        st.markdown(
            '<div class="panel"><div class="panel-header"><span>Snapshot</span></div>'
            f'<div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )

    # History: ensemble + BIC fitted, NBER shaded.
    ens_hist = report["ensemble_history"]
    bic_hist = report["bic_history"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bic_hist.index, y=bic_hist.values, mode="lines",
            line=dict(color=PALETTE["text_muted"], width=1, dash="dot"),
            name="BIC-selected",
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra>BIC</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ens_hist.index, y=ens_hist.values, mode="lines",
            line=dict(color=PALETTE["accent"], width=1.6),
            fill="tozeroy", fillcolor=_fade(PALETTE["accent"], 0.12),
            name="5-model ensemble",
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra>Ensemble</extra>",
        )
    )
    fig.add_hline(y=THRESHOLD_ELEVATED, line=dict(color=PALETTE["risk_critical"], width=1, dash="dash"))
    fig.add_hline(y=THRESHOLD_WARNING, line=dict(color=PALETTE["risk_elevated"], width=1, dash="dot"))
    add_recession_shading(fig, nber)
    fig.update_yaxes(title="Recession probability (%)", range=[0, 100])
    apply_template(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Plain-English read + trend attribution.
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="label-small">Today\'s read</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel"><div class="panel-body" '
            f'style="font-family:\'Fraunces\',serif;font-size:15px;line-height:1.7;color:{PALETTE["text_primary"]};">'
            f"{_reading_text(report)}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="label-small">What moved the probability (24m)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel"><div class="panel-body">{_attribution_block(report)}</div></div>',
            unsafe_allow_html=True,
        )


def _reading_text(report: dict) -> str:
    ens = report["ensemble_probability"]
    probs = report["model_probabilities"]
    p_lo, p_hi = min(probs.values()), max(probs.values())
    consensus = report["consensus"].lower()
    parts = [
        f"Our five-model ensemble estimates a <b>{ens:.0f}%</b> probability of a U.S. "
        f"recession within 12 months — consistent with <b>{_consistent_with(ens)}</b>."
    ]
    lo, hi = report.get("ci_lower"), report.get("ci_upper")
    if lo is not None and hi is not None:
        parts.append(f"The BIC model's bootstrap 90% interval spans {lo:.0f}–{hi:.0f}%.")
    parts.append(
        f"Individual models range from {p_lo:.0f}% to {p_hi:.0f}%, "
        f"a {consensus} consensus across specifications."
    )
    ta = report.get("trend_attribution") or {}
    if "prob_change_pp" in ta:
        ch = ta["prob_change_pp"]
        verb = "risen" if ch > 0 else "fallen" if ch < 0 else "held"
        parts.append(f"Over the past 24 months the BIC reading has {verb} by {abs(ch):.0f}pp.")
    return " ".join(parts)


def _attribution_block(report: dict) -> str:
    ta = report.get("trend_attribution") or {}
    effects = ta.get("partial_effects")
    if not effects:
        return '<div class="submodel-row"><span class="name">—</span><span class="value">attribution unavailable</span></div>'
    ordered = sorted(effects.items(), key=lambda x: x[1], reverse=True)
    up = [(f, v) for f, v in ordered if v > 0][:3]
    down = [(f, v) for f, v in ordered if v < 0][-3:][::-1]

    def rows(items, col):
        if not items:
            return '<div class="submodel-row"><span class="name">—</span><span class="value">—</span></div>'
        return "".join(
            f'<div class="submodel-row"><span class="name">{feature_label(f)}</span>'
            f'<span class="value" style="color:{col};">{v:+.1f} pp</span></div>'
            for f, v in items
        )

    return (
        '<div style="margin-bottom:10px;"><div class="label-tiny" style="margin-bottom:4px;">Pushed risk up</div>'
        f"{rows(up, PALETTE['risk_high'])}</div>"
        '<div><div class="label-tiny" style="margin-bottom:4px;">Pulled risk down</div>'
        f"{rows(down, PALETTE['risk_low'])}</div>"
    )


# ----------------------------------------------------------------- Under the Hood


def _render_under_hood(report: dict) -> None:
    probs = report["model_probabilities"]
    ens = report["ensemble_probability"]

    # Model cards: ensemble first, then the five specifications.
    order = ["NY Fed", "Wright", "BIC-selected", "Estrella-Mishkin", "Chauvet-Piger"]
    cards = [("5-model ensemble", ens, True)] + [
        (name, probs[name], False) for name in order if name in probs
    ]
    cols = st.columns(len(cards))
    for col, (name, val, is_ens) in zip(cols, cards):
        with col:
            st.markdown(
                metric_card(
                    label=name,
                    value=f"{val:.0f}",
                    unit="%",
                    risk_color_hex=PALETTE["accent"] if is_ens else _prob_color(val),
                    subline="ensemble" if is_ens else "12m probability",
                ),
                unsafe_allow_html=True,
            )

    # Model comparison bar.
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Model comparison · do the specifications agree?</div>',
        unsafe_allow_html=True,
    )
    s = pd.Series(probs).sort_values()
    colors = [PALETTE["accent"] if n == "BIC-selected" else PALETTE["text_muted"] for n in s.index]
    fig = go.Figure(
        go.Bar(
            x=s.values, y=s.index, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in s.values], textposition="outside",
            textfont=dict(color=PALETTE["text_primary"]),
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
        )
    )
    fig.add_vline(x=ens, line=dict(color=PALETTE["accent"], width=1, dash="dash"))
    fig.add_vline(x=THRESHOLD_WARNING, line=dict(color=PALETTE["risk_elevated"], width=1, dash="dot"))
    fig.update_xaxes(title="12-month recession probability (%)", range=[0, max(s.max() * 1.25, 40)])
    apply_template(fig, height=300, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:12px;color:{PALETTE["text_primary"]};line-height:1.6;">'
        "All five probabilities are computed live from FRED — none are hand-entered. "
        "<b>NY Fed</b> and <b>Estrella-Mishkin</b> use the 10y-3m term spread; <b>Wright</b> adds the fed funds rate; "
        "<b>BIC-selected</b> is a sign-constrained multivariate probit; <b>Chauvet-Piger</b> is FRED's smoothed "
        "Markov-switching series (<code>RECPROUSM156N</code>). The ensemble is their equal-weighted average."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Indicator percentile dashboard (BIC features).
    _render_percentiles(report)


def _render_percentiles(report: dict) -> None:
    readings = report["indicator_readings"]
    if not readings:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:24px;">Indicator dashboard · where each BIC driver sits historically</div>',
        unsafe_allow_html=True,
    )
    feats = list(readings.keys())
    pctiles = [readings[f]["percentile"] for f in feats]
    labels = [f"{feature_label(f)}  ·  {readings[f]['category']}" for f in feats]

    def color(p):
        if p <= 10 or p >= 90:
            return PALETTE["risk_critical"]
        if p <= 25 or p >= 75:
            return PALETTE["risk_elevated"]
        return PALETTE["risk_low"]

    fig = go.Figure(
        go.Bar(
            x=pctiles, y=labels, orientation="h",
            marker=dict(color=[color(p) for p in pctiles], line=dict(width=0)),
            text=[f"{p:.0f}th" for p in pctiles], textposition="outside",
            textfont=dict(color=PALETTE["text_primary"]),
            hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
        )
    )
    fig.add_vline(x=50, line=dict(color=PALETTE["text_muted"], width=1, dash="dash"))
    fig.update_xaxes(title="Historical percentile", range=[0, 108])
    fig.update_yaxes(autorange="reversed")
    apply_template(fig, height=max(260, len(feats) * 48), show_legend=False)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------- Watchlist


def _render_watchlist(report: dict) -> None:
    sens = report.get("sensitivity") or []
    if not sens:
        st.info("No watchlist data available.")
        return

    # Rank by proximity to the 30% trigger (closest first).
    def distance_key(s):
        d = s.get("distance_30")
        return abs(d) if d is not None else float("inf")

    ranked = sorted(sens, key=distance_key)

    st.markdown(
        '<div class="label-small">Watchlist · how far each driver is from triggering a higher reading</div>',
        unsafe_allow_html=True,
    )
    rows_html = [
        '<tr style="border-bottom:1px solid #1f2630;color:#6b7280;font-size:10px;'
        'letter-spacing:0.08em;text-transform:uppercase;">'
        "<th style='text-align:left;padding:6px 8px;'>Indicator</th>"
        "<th style='text-align:right;padding:6px 8px;'>Current</th>"
        "<th style='text-align:right;padding:6px 8px;'>→30% at</th>"
        "<th style='text-align:right;padding:6px 8px;'>Distance</th>"
        "<th style='text-align:right;padding:6px 8px;'>→50% at</th>"
        "<th style='text-align:right;padding:6px 8px;'>±1SD impact</th></tr>"
    ]
    for s in ranked:
        t30 = f"{s['trigger_30']:.2f}" if s.get("trigger_30") is not None else "—"
        d30 = f"{s['distance_30']:+.2f}" if s.get("distance_30") is not None else "—"
        t50 = f"{s['trigger_50']:.2f}" if s.get("trigger_50") is not None else "—"
        rows_html.append(
            f'<tr style="border-bottom:1px solid #141a22;color:{PALETTE["text_primary"]};font-size:12px;">'
            f'<td style="padding:6px 8px;">{feature_label(s["feature"])}'
            f'<span style="color:#5a6470;font-size:10px;"> · {s["category"]} · {s["feature"]}</span></td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{s["current_value"]:.2f}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{t30}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;color:{PALETTE["accent"]};">{d30}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{t50}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{s["impact_pp"]:+.1f} pp</td></tr>'
        )
    st.markdown(
        '<div class="panel"><div class="panel-body"><table style="width:100%;border-collapse:collapse;">'
        + "".join(rows_html)
        + "</table></div></div>",
        unsafe_allow_html=True,
    )

    nearest = ranked[0]
    nearest_txt = (
        f"<b>{feature_label(nearest['feature'])}</b> is closest to its 30% trigger — "
        f"{abs(nearest['distance_30']):.2f} away from {nearest['trigger_30']:.2f} "
        f"(currently {nearest['current_value']:.2f})."
        if nearest.get("distance_30") is not None
        else "No driver is currently within range of its 30% trigger."
    )

    # Adverse scenario + what-would-change-our-view.
    c1, c2 = st.columns([1, 1])
    with c1:
        adverse = report["adverse_scenario_probability"]
        st.markdown('<div class="label-small">Adverse scenario</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
            f"If every BIC driver simultaneously deteriorated by one standard deviation in its "
            f"risk-increasing direction, the BIC model would print <b>{adverse:.0f}%</b>. "
            "This is a tail construction — it requires uncorrelated indicators to move together, "
            f"which historically happens only in systemic stress. {nearest_txt}"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="label-small">What would change our view</div>', unsafe_allow_html=True)
        items = [
            "Term spread falls below −1.0% — historically associated with hard-landing risk.",
            "Core CPI drops below 1.5% — demand destruction outpacing supply normalization.",
            "Housing starts decline more than 15% YoY — mortgage-rate transmission accelerating.",
            "Initial claims sustain above 300K (4-week avg) — early labor-demand deterioration.",
            "Ensemble rises above 20% for two consecutive updates — model-convergence signal.",
        ]
        body = "".join(
            f'<div style="font-size:12px;line-height:1.6;color:{PALETTE["text_primary"]};margin-bottom:6px;">'
            f'<span style="color:{PALETTE["accent"]};">{i+1}.</span> {t}</div>'
            for i, t in enumerate(items)
        )
        st.markdown(f'<div class="panel"><div class="panel-body">{body}</div></div>', unsafe_allow_html=True)

    # Data currency notice.
    lagged = report.get("lagged_series") or []
    lagged_txt = ", ".join(lagged) if lagged else "None"
    st.markdown(
        f'<div class="panel" style="margin-top:16px;border-color:{PALETTE["panel_border"]};">'
        f'<div class="panel-body" style="font-size:11px;color:{PALETTE["text_muted"]};line-height:1.6;">'
        f"<b>Data currency.</b> Reflects FRED data through {report['data_through']}. "
        f"Series with publication lags over 30 days: {lagged_txt}. "
        "Probabilities refresh automatically on the next cached rebuild."
        "</div></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------- Scenario


def _render_scenario(report: dict) -> None:
    sens = report.get("sensitivity") or []
    coefs = report.get("bic_coefficients") or {}
    if not sens or not coefs:
        st.info("Scenario tool unavailable — the BIC model has no tunable drivers.")
        return

    baseline = report["bic_probability"]

    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        "<p>Move the drivers and watch the probability respond. This perturbs the "
        "<b>BIC-selected multivariate model</b> — the one specification of the five with "
        "multiple tunable inputs — recomputing Φ(β·x) directly from its fitted coefficients "
        "(no refit). Every non-moved driver is held at its current value, so each reading is a "
        "<i>ceteris paribus</i> what-if, not a forecast of joint moves.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Order drivers by influence (|coef · 1SD|) so the most impactful sit on top.
    rows = sorted(sens, key=lambda s: abs(s.get("coef", 0.0) * s.get("std_dev", 0.0)), reverse=True)

    left, right = st.columns([3, 2])
    overrides: dict[str, float] = {}
    with left:
        st.markdown('<div class="label-small">Drivers</div>', unsafe_allow_html=True)
        for s in rows:
            feat = s["feature"]
            lo, hi = float(s["hist_min"]), float(s["hist_max"])
            cur = float(s["current_value"])
            if not all(np.isfinite(v) for v in (lo, hi, cur)):
                continue
            # Always include the current reading (it can be a fresh extreme beyond
            # the training-sample min/max) so the slider default stays in range.
            lo, hi = min(lo, cur), max(hi, cur)
            span = (hi - lo) or abs(cur) or 1.0
            lo_s, hi_s = lo - 0.05 * span, hi + 0.05 * span
            step = max(span / 200.0, 1e-4)
            val = st.slider(
                f"{feature_label(feat)}",
                min_value=float(round(lo_s, 4)),
                max_value=float(round(hi_s, 4)),
                value=float(cur),
                step=float(step),
                key=f"scenario_{feat}",
                help=f"{feat} · current {cur:g} · historical range [{lo:g}, {hi:g}]",
            )
            overrides[feat] = val

    scenario_prob = scenario_probability(report, overrides)
    delta = scenario_prob - baseline
    color = _prob_color(scenario_prob)

    with right:
        st.markdown('<div class="label-small">Scenario probability</div>', unsafe_allow_html=True)
        st.markdown(
            metric_card(
                label="BIC model · this scenario",
                value=f"{scenario_prob:.0f}",
                unit="%",
                risk_color_hex=color,
                subline=f"baseline {baseline:.0f}% · {delta:+.0f} pp vs current",
            ),
            unsafe_allow_html=True,
        )
        # How far the scenario sits from the warning / elevated thresholds.
        st.markdown(
            f'<div class="panel" style="margin-top:8px;"><div class="panel-body" '
            f'style="font-size:12px;color:{PALETTE["text_primary"]};line-height:1.6;">'
            f"Warning threshold {THRESHOLD_WARNING}% · elevated {THRESHOLD_ELEVATED}%. "
            + (
                f"This scenario is <b>{scenario_prob - THRESHOLD_WARNING:+.0f} pp</b> relative to the "
                "warning line."
                if np.isfinite(scenario_prob) else ""
            )
            + " Reset by dragging sliders back, or reload the page."
            "</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("Reset to current", key="scenario_reset"):
            for s in rows:
                st.session_state.pop(f"scenario_{s['feature']}", None)
            st.rerun()

    st.markdown(
        f'<div class="panel" style="margin-top:12px;border-color:{PALETTE["panel_border"]};">'
        f'<div class="panel-body" style="font-size:11px;color:{PALETTE["text_muted"]};line-height:1.6;">'
        "<b>Reading this honestly.</b> This is the BIC model alone, not the five-model ensemble "
        f"headline ({report['ensemble_probability']:.0f}%). It assumes each driver moves "
        "independently — real downturns move them together, so a realistic joint move would "
        "typically push the probability higher than any single-slider change implies. Slider "
        "ranges span each driver's historical min–max over the training sample."
        "</div></div>",
        unsafe_allow_html=True,
    )
