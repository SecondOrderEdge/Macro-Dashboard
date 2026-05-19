"""Reusable UI building blocks: Plotly chart helpers and HTML metric cards."""

from __future__ import annotations

from html import escape
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ui.theme import PALETTE, risk_color


PLOTLY_TEMPLATE: dict = {
    "layout": {
        "paper_bgcolor": "#0a0d12",
        "plot_bgcolor": "#0a0d12",
        "font": {"family": "JetBrains Mono", "color": "#6b7280", "size": 10},
        "xaxis": {
            "gridcolor": "#1f2630",
            "zerolinecolor": "#3d4754",
            "linecolor": "#1f2630",
            "tickfont": {"family": "JetBrains Mono", "color": "#6b7280"},
        },
        "yaxis": {
            "gridcolor": "#1f2630",
            "zerolinecolor": "#3d4754",
            "linecolor": "#1f2630",
            "tickfont": {"family": "JetBrains Mono", "color": "#6b7280"},
        },
        "margin": {"l": 60, "r": 30, "t": 20, "b": 40},
        "hoverlabel": {
            "bgcolor": "#0d1117",
            "bordercolor": "#1f2630",
            "font": {"family": "JetBrains Mono", "color": "#d4d4d0"},
        },
        "showlegend": True,
        "legend": {"font": {"color": "#6b7280", "size": 10}, "bgcolor": "rgba(0,0,0,0)"},
    }
}


def apply_template(fig: go.Figure, *, height: int | None = None, show_legend: bool | None = None) -> go.Figure:
    """Apply the dark template; optionally set height and legend visibility."""
    layout = dict(PLOTLY_TEMPLATE["layout"])
    if height is not None:
        layout["height"] = height
    if show_legend is not None:
        layout["showlegend"] = show_legend
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------- recession shading


def add_recession_shading(
    fig: go.Figure,
    nber: pd.Series,
    *,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Shade NBER recession periods on a time-series chart."""
    if nber is None or nber.empty:
        return fig

    arr = nber.values.astype(bool)
    idx = pd.DatetimeIndex(nber.index)
    in_run = False
    start = None
    for i, val in enumerate(arr):
        if val and not in_run:
            start = idx[i]
            in_run = True
        elif not val and in_run:
            end = idx[i - 1]
            _add_vrect(fig, start, end, row=row, col=col)
            in_run = False
    if in_run and start is not None:
        _add_vrect(fig, start, idx[-1], row=row, col=col)
    return fig


def _add_vrect(fig, start, end, row=None, col=None):
    kwargs = dict(
        x0=start,
        x1=end,
        fillcolor="#3d4754",
        opacity=0.18,
        line_width=0,
        layer="below",
    )
    if row is not None and col is not None:
        kwargs["row"] = row
        kwargs["col"] = col
    fig.add_vrect(**kwargs)


# ----------------------------------------------------------------- panel HTML


def panel_open(title: str, right_text: str | None = None) -> str:
    right = f"<span>{escape(right_text)}</span>" if right_text else ""
    return (
        '<div class="panel">'
        f'<div class="panel-header"><span>{escape(title)}</span>{right}</div>'
        '<div class="panel-body">'
    )


def panel_close() -> str:
    return "</div></div>"


# -------------------------------------------------------------- metric cards


def metric_card(
    label: str,
    value: str,
    unit: str = "",
    risk_color_hex: str | None = None,
    sparkline_html: str | None = None,
    badge: str | None = None,
    subline: str | None = None,
) -> str:
    """Self-contained metric card. Returns an HTML fragment.

    Pass an SVG string for ``sparkline_html`` (see :func:`sparkline_svg`).
    """
    color = risk_color_hex or PALETTE["text_primary"]
    badge_html = ""
    if badge:
        badge_html = f'<span class="risk-badge" style="color:{color};margin-left:8px;">{escape(badge)}</span>'
    subline_html = f'<div class="metric-sub">{escape(subline)}</div>' if subline else ""
    spark_html = f'<div style="margin-top:12px;">{sparkline_html}</div>' if sparkline_html else ""
    unit_html = f'<span class="metric-unit">{escape(unit)}</span>' if unit else ""

    return f"""
<div class="panel" style="height:100%;">
  <div class="panel-header">
    <span>{escape(label)}</span>
    {badge_html}
  </div>
  <div class="panel-body">
    <div class="metric-big data-font" style="color:{color};">{escape(value)}{unit_html}</div>
    {subline_html}
    {spark_html}
  </div>
</div>
"""


def sparkline_svg(
    values: Iterable[float],
    *,
    width: int = 200,
    height: int = 40,
    color: str | None = None,
) -> str:
    """Compact inline SVG sparkline."""
    color = color or PALETTE["accent"]
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return ""
    xs = np.linspace(2, width - 2, len(arr))
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        ys = np.full_like(arr, height / 2)
    else:
        ys = height - 2 - (arr - lo) / (hi - lo) * (height - 4)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{pts}"/></svg>'
    )


# -------------------------------------------------------- common chart types


def line_chart(
    series: pd.Series | pd.DataFrame,
    *,
    color: str | dict | None = None,
    height: int = 320,
    nber: pd.Series | None = None,
    zero_line: bool = False,
    fill: bool = False,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Single- or multi-line dark-themed time-series chart."""
    fig = go.Figure()
    if isinstance(series, pd.Series):
        items = [(series.name or "value", series)]
    else:
        items = list(series.items())

    for name, s in items:
        s = s.dropna()
        c = color if isinstance(color, str) else (color or {}).get(name, PALETTE["accent"])
        kwargs = dict(
            x=s.index, y=s.values, mode="lines",
            name=str(name), line=dict(color=c, width=1.4),
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra></extra>",
        )
        if fill:
            kwargs["fill"] = "tozeroy"
            kwargs["fillcolor"] = _fade(c, 0.18)
        fig.add_trace(go.Scatter(**kwargs))

    if zero_line:
        fig.add_hline(y=0, line=dict(color="#3d4754", width=1, dash="dot"))

    if nber is not None:
        add_recession_shading(fig, nber)

    if yaxis_title:
        fig.update_yaxes(title=yaxis_title)
    return apply_template(fig, height=height)


def _fade(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def horizontal_bar(
    values: pd.Series,
    *,
    color: str | list[str] | None = None,
    height: int = 320,
    xaxis_title: str | None = None,
) -> go.Figure:
    """Horizontal bar chart, sorted by value."""
    values = values.dropna().sort_values()
    color = color or PALETTE["accent"]
    fig = go.Figure(
        go.Bar(
            x=values.values,
            y=values.index,
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    if xaxis_title:
        fig.update_xaxes(title=xaxis_title)
    return apply_template(fig, height=height, show_legend=False)


def signed_bar(
    values: pd.Series,
    *,
    pos_color: str | None = None,
    neg_color: str | None = None,
    height: int = 320,
    xaxis_title: str | None = None,
) -> go.Figure:
    """Diverging bar chart for z-scores or contributions."""
    pos_color = pos_color or PALETTE["risk_low"]
    neg_color = neg_color or PALETTE["risk_high"]
    values = values.dropna()
    colors = [pos_color if v >= 0 else neg_color for v in values.values]
    fig = go.Figure(
        go.Bar(
            x=values.values,
            y=values.index,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#3d4754", width=1))
    if xaxis_title:
        fig.update_xaxes(title=xaxis_title)
    return apply_template(fig, height=height, show_legend=False)


def reliability_diagram(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#3d4754", width=1, dash="dot"),
            name="Perfect", hoverinfo="skip", showlegend=False,
        )
    )
    if not curve.empty:
        fig.add_trace(
            go.Scatter(
                x=curve["predicted"], y=curve["actual"], mode="lines+markers",
                line=dict(color=PALETTE["accent"], width=1.6),
                marker=dict(size=8, color=PALETTE["accent"]),
                hovertemplate="pred %{x:.2f} → actual %{y:.2f}<extra></extra>",
                name="Model",
            )
        )
    fig.update_xaxes(title="Predicted probability", range=[0, 1])
    fig.update_yaxes(title="Actual frequency", range=[0, 1])
    return apply_template(fig, height=300, show_legend=False)


def composite_html(score: int, band: str, *, large: bool = True) -> str:
    color = risk_color(band)
    size = "48px" if large else "32px"
    return (
        f'<div class="composite-readout">'
        f'<div class="label-tiny">Composite Risk</div>'
        f'<div class="composite-number" style="color:{color};font-size:{size};">{score}</div>'
        f'<div class="risk-badge" style="color:{color};margin-top:6px;">{escape(band)}</div>'
        f'</div>'
    )
