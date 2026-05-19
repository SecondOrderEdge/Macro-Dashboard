"""Yield curve view: term structure, spreads, inversion stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.yield_curve import YieldCurve
from src.ui.components import (
    add_recession_shading,
    apply_template,
    metric_card,
    sparkline_svg,
)
from src.ui.theme import PALETTE


def render(panel: pd.DataFrame, nber: pd.Series) -> None:
    yc = YieldCurve(panel)
    spreads = yc.spreads_history()
    if spreads.empty:
        st.warning("No yield curve data available.")
        return

    stats = yc.inversion_stats(nber)
    _render_spread_cards(spreads, stats)
    _render_term_structure(yc)
    _render_history_and_stats(spreads, nber, stats)


def _render_spread_cards(spreads: pd.DataFrame, stats: dict) -> None:
    cols = st.columns(3)
    labels = [
        ("spread_10y3m", "10Y − 3M"),
        ("spread_10y2y", "10Y − 2Y"),
        ("spread_5y2y", "5Y − 2Y"),
    ]
    for col, (key, label) in zip(cols, labels):
        if key not in spreads.columns:
            continue
        series = spreads[key].dropna()
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        inverted = latest < 0
        color = PALETTE["risk_critical"] if inverted else PALETTE["risk_low"]
        badge = "INVERTED" if inverted else "POSITIVE"
        spark = sparkline_svg(series.tail(252), color=color)
        subline = None
        if key == "spread_10y3m" and stats.get("months_inverted", 0) > 0:
            subline = f"{stats['months_inverted']} consec. inverted months"
        with col:
            st.markdown(
                metric_card(
                    label=label,
                    value=f"{latest:+.2f}",
                    unit="pp",
                    risk_color_hex=color,
                    sparkline_html=spark,
                    badge=badge,
                    subline=subline,
                ),
                unsafe_allow_html=True,
            )


def _render_term_structure(yc: YieldCurve) -> None:
    st.markdown('<div class="label-small" style="margin-top:8px;">Term Structure</div>', unsafe_allow_html=True)
    ts = yc.term_structure()
    if ts.empty:
        st.info("Term structure unavailable.")
        return

    fig = go.Figure()
    for col, name, color in [
        ("m12", "12 months ago", PALETTE["text_tiny"]),
        ("m3", "3 months ago", PALETTE["text_muted"]),
        ("current", "Current", PALETTE["accent"]),
    ]:
        if col not in ts.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=ts["maturity"],
                y=ts[col],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=1.6 if col == "current" else 1.0),
                marker=dict(size=7 if col == "current" else 5, color=color),
                hovertemplate=f"{name}<br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_yaxes(title="Yield (%)")
    fig.update_xaxes(title="Maturity")
    apply_template(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


def _render_history_and_stats(spreads: pd.DataFrame, nber: pd.Series, stats: dict) -> None:
    left, right = st.columns([3, 1])

    with left:
        st.markdown(
            '<div class="label-small" style="margin-top:8px;">Spread History · 10Y−3M and 10Y−2Y</div>',
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        if "spread_10y3m" in spreads.columns:
            s = spreads["spread_10y3m"].dropna()
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values, mode="lines",
                    line=dict(color=PALETTE["accent"], width=1.2),
                    name="10Y − 3M",
                    hovertemplate="%{x|%b %Y}<br>%{y:+.2f}pp<extra>10Y−3M</extra>",
                )
            )
        if "spread_10y2y" in spreads.columns:
            s = spreads["spread_10y2y"].dropna()
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values, mode="lines",
                    line=dict(color=PALETTE["submodel"]["labor"], width=1.0),
                    name="10Y − 2Y",
                    hovertemplate="%{x|%b %Y}<br>%{y:+.2f}pp<extra>10Y−2Y</extra>",
                )
            )
        fig.add_hline(y=0, line=dict(color="#3d4754", width=1, dash="dot"))
        add_recession_shading(fig, nber)
        fig.update_yaxes(title="Spread (pp)")
        apply_template(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="label-small" style="margin-top:8px;">Inversion Statistics</div>',
            unsafe_allow_html=True,
        )
        hits, total = stats.get("hit_rate", (0, 0))
        depth = stats.get("max_depth_current", float("nan"))
        lead = stats.get("avg_lead_to_recession", float("nan"))
        months = stats.get("months_inverted", 0)
        rows = [
            ("Months inverted (current)", f"{months}"),
            ("Max depth (current run)", f"{depth:+.2f} pp" if not np.isnan(depth) else "—"),
            ("Avg lead to NBER peak", f"{lead:.0f} months" if not np.isnan(lead) else "—"),
            ("Inversion hit rate", f"{hits} / {total}" if total else "—"),
        ]
        body_rows = "".join(
            f'<div class="submodel-row"><span class="name">{label}</span>'
            f'<span class="value">{value}</span></div>'
            for label, value in rows
        )
        st.markdown(
            f'<div class="panel"><div class="panel-body">{body_rows}</div></div>',
            unsafe_allow_html=True,
        )

        interp = _interpretation(months, depth, hits, total)
        st.markdown(
            f'<div class="panel"><div class="panel-header"><span>Interpretation</span></div>'
            f'<div class="panel-body" style="font-size:12px;color:{PALETTE["text_primary"]};line-height:1.6;">{interp}</div></div>',
            unsafe_allow_html=True,
        )


def _interpretation(months: int, depth: float, hits: int, total: int) -> str:
    if months == 0:
        return (
            "The 10Y−3M spread is not currently inverted. Curve-implied recession risk "
            "is muted; watch for re-inversion if the front end re-prices higher."
        )
    base = (
        f"The curve has been inverted for {months} consecutive months, reaching a "
        f"trough of {depth:+.2f}pp during the current episode. "
    )
    if total:
        base += (
            f"Historically, {hits} of the last {total} sustained inversions (>3 months) "
            "preceded an NBER recession within three years."
        )
    return base
