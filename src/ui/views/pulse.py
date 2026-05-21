"""Pulse view — real-time breadth & diffusion monitor.

The recession ensemble is monthly and calibrated; this tab answers a different
question between releases: *how broad is the weakness?* It tracks diffusion /
breadth — what share of indicators are below trend and deteriorating — which is
the tell that separates a real downturn from a one-off wobble. Descriptive, not
a second forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.breadth import (
    below_trend_breadth,
    breadth_snapshot,
    breadth_state,
    cfnai_diffusion_band,
    momentum_breadth,
)
from src.models.lame import LAME
from src.models.market_implied import MARKET_IMPLIED_SIGNALS, signal_summary
from src.ui.components import add_recession_shading, apply_template, metric_card, sparkline_svg
from src.ui.theme import PALETTE


def _sev_color(sev: str) -> str:
    return {
        "low": PALETTE["risk_low"],
        "elevated": PALETTE["risk_elevated"],
        "high": PALETTE["risk_high"],
        "critical": PALETTE["risk_critical"],
    }.get(sev, PALETTE["text_muted"])


def _fade(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render(panel: pd.DataFrame, nber: pd.Series, lame: LAME) -> None:
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        "<b>Real-time breadth monitor.</b> The recession model updates monthly; this tracks how "
        "<i>broad</i> the weakness is right now. A few indicators rolling over is noise — a "
        "majority rolling over together is how downturns start. This is descriptive, not a "
        "second forecast."
        "</div></div>",
        unsafe_allow_html=True,
    )

    zscores = getattr(lame, "_zscores", None)
    snap = breadth_snapshot(zscores) if zscores is not None else breadth_snapshot(pd.DataFrame())

    # --- cards -----------------------------------------------------------
    cols = st.columns(3)

    # CFNAI diffusion index (authoritative macro breadth, ~85 components).
    with cols[0]:
        diff_series = panel["CFNAIDIFF"].dropna() if "CFNAIDIFF" in panel.columns else pd.Series(dtype=float)
        if diff_series.empty:
            st.markdown(metric_card("Activity Diffusion (CFNAI)", "—", ""), unsafe_allow_html=True)
        else:
            val = float(diff_series.iloc[-1])
            label, sev = cfnai_diffusion_band(val)
            color = _sev_color(sev)
            st.markdown(
                metric_card(
                    label="Activity Diffusion (CFNAI)",
                    value=f"{val:+.2f}",
                    unit="",
                    risk_color_hex=color,
                    sparkline_html=sparkline_svg(diff_series.tail(120).values, color=color),
                    badge=label,
                    subline=f"~85 components · {diff_series.index[-1].strftime('%b %Y')}",
                ),
                unsafe_allow_html=True,
            )

    # Labor breadth — share below trend.
    with cols[1]:
        below, total = snap["below_trend"], snap["total"]
        label, sev = breadth_state(snap["below_trend_pct"])
        color = _sev_color(sev)
        st.markdown(
            metric_card(
                label="Labor breadth · below trend",
                value=f"{below}/{total}" if total else "—",
                unit="",
                risk_color_hex=color,
                badge=label,
                subline="indicators below their own trend",
            ),
            unsafe_allow_html=True,
        )

    # Labor momentum — share deteriorating over 3 months.
    with cols[2]:
        falling, ftotal = snap["falling"], snap.get("falling_total", 0)
        pct = snap["falling_pct"]
        color = _sev_color(breadth_state(pct)[1])
        st.markdown(
            metric_card(
                label="Labor momentum · deteriorating",
                value=f"{falling}/{ftotal}" if ftotal else "—",
                unit="",
                risk_color_hex=color,
                subline="indicators falling over 3 months",
            ),
            unsafe_allow_html=True,
        )

    # --- breadth over time ----------------------------------------------
    if zscores is None or zscores.empty:
        st.info("Breadth history unavailable — labor composite not ready.")
        return

    below_hist = below_trend_breadth(zscores)
    mom_hist = momentum_breadth(zscores)
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=30)
    below_hist = below_hist.loc[below_hist.index >= cutoff]
    mom_hist = mom_hist.loc[mom_hist.index >= cutoff]

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Labor breadth over time · share of indicators below trend / deteriorating</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mom_hist.index, y=mom_hist.values, mode="lines",
            line=dict(color=PALETTE["text_muted"], width=1, dash="dot"),
            name="Deteriorating (3m, %)",
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra>falling</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=below_hist.index, y=below_hist.values, mode="lines",
            line=dict(color=PALETTE["accent"], width=1.6),
            fill="tozeroy", fillcolor=_fade(PALETTE["accent"], 0.12),
            name="Below trend (%)",
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra>below trend</extra>",
        )
    )
    fig.add_hline(y=50, line=dict(color="#3d4754", width=1, dash="dot"))
    add_recession_shading(fig, nber.loc[nber.index >= cutoff])
    fig.update_yaxes(title="% of labor indicators", range=[0, 100])
    apply_template(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)

    # --- read ------------------------------------------------------------
    latest_below = float(below_hist.iloc[-1]) if not below_hist.empty else float("nan")
    state_label, _ = breadth_state(latest_below)
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        f"<b>{snap['below_trend']} of {snap['total']}</b> labor indicators are below their own trend "
        f"and <b>{snap['falling']} of {snap.get('falling_total', 0)}</b> have deteriorated over the "
        f"past three months — a <b>{state_label.lower()}</b> reading. Breadth above ~70% (the dotted "
        "line is 50%) historically coincides with the early innings of recessions; isolated weakness "
        "in one or two series rarely does. Read this alongside the monthly ensemble: broad breadth "
        "deterioration is what turns a soft patch into a downturn."
        "</div></div>",
        unsafe_allow_html=True,
    )

    _render_market_implied(panel)


def _render_market_implied(panel: pd.DataFrame) -> None:
    """What the market is pricing — forward expectations from traded assets."""
    rows = []
    for fred_id, label, unit, fmt in MARKET_IMPLIED_SIGNALS:
        if fred_id not in panel.columns:
            continue
        s = signal_summary(panel[fred_id])
        if s["as_of"] is None:
            continue
        rows.append((label, unit, fmt, s))
    if not rows:
        return

    st.markdown(
        '<div class="label-small" style="margin-top:24px;">Market-implied · what traded assets are pricing</div>',
        unsafe_allow_html=True,
    )

    header = (
        '<tr style="border-bottom:1px solid #1f2630;color:#6b7280;font-size:10px;'
        'letter-spacing:0.08em;text-transform:uppercase;">'
        "<th style='text-align:left;padding:6px 8px;'>Signal</th>"
        "<th style='text-align:right;padding:6px 8px;'>Latest</th>"
        "<th style='text-align:right;padding:6px 8px;'>1m change</th>"
        "<th style='text-align:right;padding:6px 8px;'>5y percentile</th></tr>"
    )
    body = []
    for label, unit, fmt, s in rows:
        val = fmt.format(s["latest"]) + (f" {unit}" if unit else "")
        chg = f"{s['change']:+.2f}" if np.isfinite(s["change"]) else "—"
        chg_color = PALETTE["text_muted"] if not np.isfinite(s["change"]) else (
            PALETTE["risk_high"] if s["change"] > 0 else PALETTE["risk_low"] if s["change"] < 0 else PALETTE["text_muted"]
        )
        pct = f"{s['percentile']:.0f}th" if np.isfinite(s["percentile"]) else "—"
        body.append(
            f'<tr style="border-bottom:1px solid #141a22;color:{PALETTE["text_primary"]};font-size:12px;">'
            f'<td style="padding:6px 8px;">{label}'
            f'<span style="color:#5a6470;font-size:10px;"> · {s["as_of"].strftime("%d %b %Y")}</span></td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{val}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;color:{chg_color};">{chg}</td>'
            f'<td style="text-align:right;padding:6px 8px;font-variant-numeric:tabular-nums;">{pct}</td></tr>'
        )
    st.markdown(
        '<div class="panel"><div class="panel-body"><table style="width:100%;border-collapse:collapse;">'
        + header + "".join(body)
        + "</table></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="panel" style="margin-top:8px;border-color:{PALETTE["panel_border"]};">'
        f'<div class="panel-body" style="font-size:11px;color:{PALETTE["text_muted"]};line-height:1.6;">'
        "These are priced into traded assets in real time and are <b>not revised</b> — the cleanest, "
        "most current read on what the market expects for inflation, real rates, growth (term spread), "
        "and credit/funding stress. The 1-month change and 5-year percentile show direction and how "
        "stretched each reading is versus its recent range."
        "</div></div>",
        unsafe_allow_html=True,
    )
