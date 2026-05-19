"""Cockpit view — top-level dashboard combining all three modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.composite import composite_risk
from src.models.lame import LAME
from src.models.recession_ensemble import RecessionEnsemble
from src.models.yield_curve import YieldCurve
from src.ui.components import (
    add_recession_shading,
    apply_template,
    metric_card,
    sparkline_svg,
)
from src.ui.theme import PALETTE, risk_color


def render(
    model: RecessionEnsemble,
    lame: LAME,
    panel: pd.DataFrame,
    nber: pd.Series,
) -> None:
    current = model.predict_current()
    history = model.predict_history()
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
    _row_two(history, lame_hist, spreads, nber)
    _row_three(ensemble_now, lame_now, curve_now, composite, current)


def _row_one(current: dict, history: pd.DataFrame, lame_hist: pd.Series, spreads: pd.DataFrame) -> None:
    cols = st.columns([2, 1, 1])

    with cols[0]:
        _recession_card(current, history)
        if st.button("Drill into Recession →", key="drill_recession"):
            st.session_state.view = "Recession"
            st.rerun()

    with cols[1]:
        _lame_card(lame_hist)
        if st.button("Drill into LAME →", key="drill_lame"):
            st.session_state.view = "LAME · Labor"
            st.rerun()

    with cols[2]:
        _curve_card(spreads)
        if st.button("Drill into Curve →", key="drill_curve"):
            st.session_state.view = "Yield Curve"
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
        label_pretty = name.replace("_", " ").title()
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
        <div class="metric-sub">5-submodel ensemble</div>
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
        st.markdown(metric_card("LAME", "—", "σ"), unsafe_allow_html=True)
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
            label="LAME · Labor",
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
                x=s.index, y=s.values, mode="lines", name="LAME (σ)",
                line=dict(color=PALETTE["submodel"]["labor"], width=1.2),
                yaxis="y2",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}σ<extra>LAME</extra>",
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
        yaxis2=dict(title="LAME (σ) · spread (pp)", overlaying="y", side="right", showgrid=False),
    )
    apply_template(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


def _row_three(ensemble_now, lame_now, curve_now, composite, current) -> None:
    left, right = st.columns(2)

    with left:
        rows = [
            ("Recession ensemble (50%)", f"{ensemble_now:.0f}%", PALETTE["accent"]),
            ("LAME-risk (25%)", f"{composite['contributions'].get('lame', 0)*4:.0f} → {composite['contributions'].get('lame', 0):.0f} pts", PALETTE["submodel"]["labor"]),
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
                f"Labor (LAME) is contractionary at {lame_now:+.2f}σ — "
                "the soft-landing thesis is on thinner ice."
            )
        else:
            parts.append(
                f"Labor (LAME) reads {lame_now:+.2f}σ, broadly consistent with continued expansion."
            )
    if not parts:
        parts.append("Not enough data is available to form a reading.")
    return " ".join(parts)
