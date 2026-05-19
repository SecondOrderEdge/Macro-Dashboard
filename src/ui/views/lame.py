"""LAME view: composite reading, indicator breakdown, Beveridge curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.lame import LAME
from src.ui.components import (
    add_recession_shading,
    apply_template,
    line_chart,
    metric_card,
    sparkline_svg,
)
from src.ui.theme import PALETTE


_BANDS = [
    ("CONTRACTIONARY", -np.inf, -1.0, PALETTE["risk_critical"]),
    ("SOFTENING", -1.0, -0.5, PALETTE["risk_high"]),
    ("NEUTRAL", -0.5, 0.5, PALETTE["risk_elevated"]),
    ("FIRM", 0.5, 1.0, PALETTE["risk_low"]),
    ("HOT", 1.0, np.inf, PALETTE["risk_low"]),
]


def render(panel: pd.DataFrame, nber: pd.Series, model: LAME | None = None) -> None:
    model = model or LAME()
    if model._composite is None:
        model.compute(panel)
    history = model.history()
    if history.empty:
        st.warning("Insufficient labor data for LAME.")
        return

    _render_top(history, model, nber)
    _render_breakdown(model)
    _render_beveridge(panel, nber)


def _render_top(history: pd.Series, model: LAME, nber: pd.Series) -> None:
    left, right = st.columns([1, 2])
    latest = float(history.iloc[-1])
    band_name, color = _band(latest)

    with left:
        spark = sparkline_svg(history.tail(120).values, color=color)
        subline = f"as of {history.index[-1].strftime('%b %Y')}"
        st.markdown(
            metric_card(
                label="LAME Composite",
                value=f"{latest:+.2f}",
                unit="σ",
                risk_color_hex=color,
                sparkline_html=spark,
                badge=band_name,
                subline=subline,
            ),
            unsafe_allow_html=True,
        )

        rows = "".join(
            f'<div class="submodel-row"><span class="name">{name}</span>'
            f'<span class="value" style="color:{c};">{lo:+.1f} → {hi:+.1f}</span></div>'
            for name, lo, hi, c in _BANDS
        )
        st.markdown(
            '<div class="panel"><div class="panel-header"><span>Regime bands</span></div>'
            f'<div class="panel-body">{rows}</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        fig = line_chart(
            history.rename("LAME"),
            color=color,
            height=380,
            nber=nber,
            zero_line=True,
            fill=True,
            yaxis_title="z-score",
        )
        fig.add_hline(y=-0.5, line=dict(color=PALETTE["risk_high"], width=1, dash="dash"))
        st.plotly_chart(fig, use_container_width=True)


def _render_breakdown(model: LAME) -> None:
    st.markdown(
        '<div class="label-small" style="margin-top:8px;">Indicator breakdown · 10 series</div>',
        unsafe_allow_html=True,
    )
    df = model.current_breakdown().copy()
    if df.empty:
        return

    # Sort by absolute contribution for readability.
    df = df.iloc[df["contribution"].abs().sort_values(ascending=False).index].reset_index(drop=True)

    fig = go.Figure()
    colors = [PALETTE["risk_low"] if v >= 0 else PALETTE["risk_high"] for v in df["z_score"]]
    fig.add_trace(
        go.Bar(
            x=df["z_score"],
            y=df["name"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate=(
                "%{y}<br>z: %{x:+.2f}σ<br>weight: %{customdata[0]:.1%}<br>"
                "contribution: %{customdata[1]:+.2f}σ<extra></extra>"
            ),
            customdata=np.stack([df["weight"].values, df["contribution"].values], axis=1),
        )
    )
    fig.add_vline(x=0, line=dict(color="#3d4754", width=1))
    fig.update_xaxes(title="signed z-score")
    fig.update_yaxes(autorange="reversed")
    apply_template(fig, height=360, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Compact table
    display = df[["name", "current_value", "z_score", "weight", "contribution"]].copy()
    display["current_value"] = display["current_value"].map(lambda v: f"{v:,.2f}")
    display["z_score"] = display["z_score"].map(lambda v: f"{v:+.2f}σ")
    display["weight"] = display["weight"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    display["contribution"] = display["contribution"].map(lambda v: f"{v:+.3f}")
    display.columns = ["indicator", "value", "z", "weight", "contribution"]
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_beveridge(panel: pd.DataFrame, nber: pd.Series) -> None:
    if "UNRATE" not in panel.columns or "JTSJOL" not in panel.columns:
        return
    if "CLF16OV" in panel.columns:
        pass  # not used; openings rate computed from JTSJOL if available
    # JTSJOL is openings (level). For a proper Beveridge we want the rate; we
    # approximate by normalizing openings against its own history.
    unrate = panel["UNRATE"].dropna().resample("ME").last()
    jol = panel["JTSJOL"].dropna().resample("ME").last()
    df = pd.concat([unrate.rename("unrate"), jol.rename("openings")], axis=1).dropna()
    if df.empty:
        return

    nber_monthly = nber.copy()
    nber_monthly.index = pd.DatetimeIndex(nber_monthly.index).to_period("M").to_timestamp()
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    df["recession"] = nber_monthly.reindex(df.index).fillna(False).astype(bool)

    recent_cutoff = df.index.max() - pd.DateOffset(months=24)
    recent = df.loc[df.index >= recent_cutoff]
    older = df.loc[df.index < recent_cutoff]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=older["unrate"], y=older["openings"], mode="markers",
            marker=dict(size=4, color=PALETTE["text_tiny"], opacity=0.5),
            name="History",
            hovertemplate="UR %{x:.1f}%<br>Openings %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent["unrate"], y=recent["openings"], mode="lines+markers",
            marker=dict(size=6, color=PALETTE["accent"]),
            line=dict(color=PALETTE["accent"], width=1.4),
            name="Last 24 months",
            hovertemplate="%{customdata|%b %Y}<br>UR %{x:.1f}%<br>Openings %{y:,.0f}<extra></extra>",
            customdata=recent.index,
        )
    )
    if not recent.empty:
        last = recent.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last["unrate"]], y=[last["openings"]], mode="markers",
                marker=dict(size=12, color=PALETTE["accent"], line=dict(color="#0a0d12", width=2)),
                name="Latest", showlegend=False,
                hovertemplate="Latest<br>UR %{x:.1f}%<br>Openings %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Unemployment rate (%)")
    fig.update_yaxes(title="Job openings (thousands)")
    apply_template(fig, height=380)
    st.markdown(
        '<div class="label-small" style="margin-top:8px;">Beveridge curve · openings vs. unemployment</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def _band(z: float) -> tuple[str, str]:
    for name, lo, hi, color in _BANDS:
        if lo <= z < hi:
            return name, color
    return "NEUTRAL", PALETTE["risk_elevated"]
