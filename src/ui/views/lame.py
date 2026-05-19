"""Labor view: composite reading, indicator breakdown, diffusion,
small-multiples grid, and the Beveridge curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.models.lame import LAME
from src.ui.components import (
    add_recession_shading,
    apply_template,
    line_chart,
    metric_card,
    sparkline_svg,
    stats_table_html,
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
        st.warning("Insufficient labor data.")
        return

    _render_top(history, model, nber)
    _render_sahm_rule(panel, nber)
    _render_breakdown(model)
    _render_diffusion(model, nber)
    _render_small_multiples(model, nber)
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
                label="Labor Composite",
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
            history.rename("Labor"),
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
    ref = model.reference_date()
    title_suffix = f" · contributions as of {ref.strftime('%b %Y')}" if ref is not None else ""
    st.markdown(
        f'<div class="label-small" style="margin-top:8px;">Indicator breakdown · 10 series{title_suffix}</div>',
        unsafe_allow_html=True,
    )
    df = model.current_breakdown().copy()
    if df.empty:
        return

    # Sort by absolute z-score so every indicator (even those with missing
    # contribution at the reference date) gets ranked sensibly.
    df = df.iloc[df["z_score"].abs().fillna(-1).sort_values(ascending=False).index].reset_index(drop=True)

    fig = go.Figure()
    z_vals = df["z_score"].fillna(0)
    colors = [PALETTE["risk_low"] if v >= 0 else PALETTE["risk_high"] for v in z_vals]
    fig.add_trace(
        go.Bar(
            x=z_vals,
            y=df["name"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate=(
                "%{y}<br>z: %{x:+.2f}σ<br>weight: %{customdata[0]}<br>"
                "contribution: %{customdata[1]}<extra></extra>"
            ),
            customdata=np.stack(
                [
                    [f"{w:.1%}" if pd.notna(w) else "—" for w in df["weight"]],
                    [f"{c:+.2f}σ" if pd.notna(c) else "—" for c in df["contribution"]],
                ],
                axis=1,
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color="#3d4754", width=1))
    fig.update_xaxes(title="signed z-score")
    fig.update_yaxes(autorange="reversed")
    apply_template(fig, height=360, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Compact table: each indicator shows its own latest value + date, then the
    # weight/contribution computed at the reference date.
    display = df[["name", "as_of", "current_value", "z_score", "weight", "contribution"]].copy()
    display["as_of"] = display["as_of"].map(
        lambda d: pd.to_datetime(d).strftime("%Y-%m") if pd.notna(d) else "—"
    )
    display["current_value"] = display["current_value"].map(
        lambda v: f"{v:,.2f}" if pd.notna(v) else "—"
    )
    display["z_score"] = display["z_score"].map(
        lambda v: f"{v:+.2f}σ" if pd.notna(v) else "—"
    )
    display["weight"] = display["weight"].map(
        lambda v: f"{v:.1%}" if pd.notna(v) else "—"
    )
    display["contribution"] = display["contribution"].map(
        lambda v: f"{v:+.3f}" if pd.notna(v) else "—"
    )
    display.columns = ["indicator", "as of", "value", "z", "weight", "contribution"]
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_sahm_rule(panel: pd.DataFrame, nber: pd.Series) -> None:
    """Sahm Rule: real-time recession indicator with zero NBER look-ahead.

    The Sahm Rule fires when the 3-month moving average of the unemployment
    rate rises by 0.5pp or more above its 12-month low. It has historically
    triggered at the start of every U.S. recession since 1970 with no false
    positives. Crucially, it uses real-time UNRATE data and is not revised
    after release — so it has no look-ahead in the way the NBER target does.
    """
    from src.models.external import sahm_rule, sahm_state

    sahm = sahm_rule(panel)
    if sahm.empty:
        return

    latest = float(sahm.iloc[-1])
    label, severity = sahm_state(latest)
    color = {
        "low": PALETTE["risk_low"],
        "elevated": PALETTE["risk_elevated"],
        "high": PALETTE["risk_high"],
        "critical": PALETTE["risk_critical"],
    }[severity]

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Sahm Rule · real-time recession indicator</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 3])
    with left:
        spark = sparkline_svg(sahm.tail(120).values, color=color)
        subline = f"as of {sahm.index[-1].strftime('%b %Y')}"
        st.markdown(
            metric_card(
                label="Sahm Rule",
                value=f"{latest:.2f}",
                unit="pp",
                risk_color_hex=color,
                sparkline_html=spark,
                badge=label,
                subline=subline,
            ),
            unsafe_allow_html=True,
        )

    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sahm.index, y=sahm.values, mode="lines",
                line=dict(color=color, width=1.4),
                fill="tozeroy", fillcolor=_fade(color, 0.10),
                name="Sahm Rule",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}pp<extra></extra>",
            )
        )
        fig.add_hline(
            y=0.5, line=dict(color=PALETTE["risk_critical"], width=1, dash="dash"),
            annotation_text="trigger 0.5pp", annotation_position="top right",
            annotation_font=dict(color=PALETTE["risk_critical"], size=10),
        )
        add_recession_shading(fig, nber)
        fig.update_yaxes(title="UNRATE 3m-MA minus 12m-min (pp)")
        apply_template(fig, height=300, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:12px;line-height:1.6;color:{PALETTE["text_primary"]};">'
        "The Sahm Rule is a complement to the labor composite. It uses only the "
        "unemployment rate (in real time, with no look-ahead) and has triggered at the "
        "onset of every U.S. recession since 1970. A reading near or above 0.5pp is the "
        "headline real-time recession signal. Source: <code>SAHMREALTIME</code> on FRED."
        "</div></div>",
        unsafe_allow_html=True,
    )


def _render_diffusion(model: LAME, nber: pd.Series) -> None:
    """Percentage of labor indicators with a positive (expansionary) z-score.

    A classic NBER-style diffusion: when the share of indicators above zero
    rolls over below 50%, the labor market is, on net, contracting. The
    diffusion line often turns several months before the LAME composite
    itself crosses zero.
    """
    z = model._zscores
    if z is None or z.empty:
        return
    valid = z.notna().sum(axis=1)
    positive = (z > 0).sum(axis=1)
    diffusion = (positive / valid.replace(0, np.nan)) * 100.0
    diffusion = diffusion.dropna()
    if diffusion.empty:
        return

    latest = float(diffusion.iloc[-1])
    color = (
        PALETTE["risk_critical"] if latest < 30
        else PALETTE["risk_high"] if latest < 50
        else PALETTE["risk_elevated"] if latest < 65
        else PALETTE["risk_low"]
    )

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Diffusion · share of labor indicators with positive z-score</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=diffusion.index, y=diffusion.values, mode="lines",
                line=dict(color=color, width=1.4),
                fill="tozeroy", fillcolor=_fade(color, 0.10),
                name="Share positive",
                hovertemplate="%{x|%b %Y}<br>%{y:.0f}%<extra></extra>",
            )
        )
        fig.add_hline(y=50, line=dict(color="#3d4754", width=1, dash="dot"))
        add_recession_shading(fig, nber)
        fig.update_yaxes(title="% positive", range=[0, 100])
        apply_template(fig, height=320, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        rows = [
            ("Today",                    f"{latest:.0f}%"),
            ("12-month change",          f"{(latest - _asof_pct(diffusion, 12)):+.0f}pp" if np.isfinite(_asof_pct(diffusion, 12)) else "—"),
            ("Median (all history)",     f"{float(diffusion.median()):.0f}%"),
            ("Min during NBER episodes", f"{_recession_min(diffusion, nber):.0f}%" if np.isfinite(_recession_min(diffusion, nber)) else "—"),
        ]
        st.markdown(stats_table_html(rows), unsafe_allow_html=True)

        verdict = (
            "Broad-based labor contraction — most indicators are below their own historical norms."
            if latest < 30 else
            "Net softening — a majority of labor indicators are now negative."
            if latest < 50 else
            "Mixed — the labor market is sending split signals."
            if latest < 65 else
            "Broad-based expansion — most indicators are above their own historical norms."
        )
        st.markdown(
            f'<div class="panel"><div class="panel-header"><span>Read</span></div>'
            f'<div class="panel-body" style="font-size:12px;line-height:1.6;color:{PALETTE["text_primary"]};">{verdict}</div></div>',
            unsafe_allow_html=True,
        )


def _render_small_multiples(model: LAME, nber: pd.Series) -> None:
    """A 5×2 grid of mini z-score charts, one per indicator."""
    z = model._zscores
    if z is None or z.empty:
        return
    cols = [c for c in LAME.INDICATORS if c in z.columns]
    if not cols:
        return

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Per-indicator z-score history · last 25 years</div>',
        unsafe_allow_html=True,
    )

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=25)
    n_cols = 5
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=cols,
        shared_yaxes=False,
        horizontal_spacing=0.04, vertical_spacing=0.18,
    )

    for i, name in enumerate(cols):
        r, c = i // n_cols + 1, i % n_cols + 1
        s = z[name].dropna()
        s = s.loc[s.index >= cutoff]
        if s.empty:
            continue
        latest = float(s.iloc[-1])
        color = PALETTE["risk_low"] if latest >= 0 else PALETTE["risk_high"]
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines",
                line=dict(color=color, width=1.0),
                fill="tozeroy", fillcolor=_fade(color, 0.10),
                showlegend=False,
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}σ<extra></extra>",
            ),
            row=r, col=c,
        )
        # Zero line + current value marker
        fig.add_hline(y=0, line=dict(color="#3d4754", width=1, dash="dot"), row=r, col=c)
        fig.add_annotation(
            x=s.index[-1], y=latest,
            text=f"{latest:+.2f}σ",
            showarrow=False,
            font=dict(color=color, size=10),
            xanchor="right", yanchor="bottom",
            row=r, col=c,
        )

    fig.update_layout(
        paper_bgcolor="#0a0d12", plot_bgcolor="#0a0d12",
        font=dict(family="JetBrains Mono", color="#6b7280", size=9),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300 * n_rows,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, color="#6b7280", tickfont=dict(size=8))
    fig.update_yaxes(gridcolor="#1f2630", zerolinecolor="#3d4754", color="#6b7280", tickfont=dict(size=8))
    # Subplot title styling. fig.layout.annotations is a tuple of plotly
    # Annotation objects which support attribute access but not .get(); use
    # try/except so any layout-side annotation we didn't expect is skipped.
    for ann in fig.layout.annotations:
        try:
            text = ann.text
        except AttributeError:
            continue
        if text in cols:
            ann.update(
                font=dict(family="JetBrains Mono", color=PALETTE["accent"], size=10),
                xanchor="left",
                x=0.0,
            )
    st.plotly_chart(fig, use_container_width=True)


def _fade(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _asof_pct(series: pd.Series, months_back: int) -> float:
    target = series.index[-1] - pd.DateOffset(months=months_back)
    s = series.loc[:target]
    return float(s.iloc[-1]) if not s.empty else float("nan")


def _recession_min(diffusion: pd.Series, nber: pd.Series) -> float:
    nber_aligned = nber.copy()
    nber_aligned.index = pd.DatetimeIndex(nber_aligned.index)
    diffusion_aligned = diffusion.copy()
    diffusion_aligned.index = pd.DatetimeIndex(diffusion_aligned.index)
    in_rec = nber_aligned.reindex(diffusion_aligned.index, method="ffill").fillna(False).astype(bool)
    if not in_rec.any():
        return float("nan")
    return float(diffusion_aligned.loc[in_rec].min())


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
