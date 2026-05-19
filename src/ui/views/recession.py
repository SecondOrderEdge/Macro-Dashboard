"""Recession ensemble view: Under the Hood / The Reading / vs. Street."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from src.models.recession_ensemble import RecessionEnsemble, SUBMODELS
from src.ui.components import (
    add_recession_shading,
    apply_template,
    line_chart,
    metric_card,
    reliability_diagram,
    sparkline_svg,
)
from src.ui.theme import PALETTE


_STREET_PATH = Path(__file__).resolve().parents[3] / "data" / "street_estimates.csv"


def render(model: RecessionEnsemble, nber: pd.Series) -> None:
    try:
        current = model.predict_current()
        history = model.predict_history()
        calibration = model.calibration_stats()
    except RuntimeError as exc:
        st.error(f"Recession ensemble not ready: {exc}")
        return

    selected = option_menu(
        menu_title=None,
        options=["Under the Hood", "The Reading", "vs. Street"],
        icons=["sliders", "graph-up", "bar-chart"],
        orientation="horizontal",
        default_index=1,
        key="recession_tab",
        styles=_TAB_STYLES,
    )

    if selected == "Under the Hood":
        _render_under_hood(model, current, history, calibration)
    elif selected == "The Reading":
        _render_reading(current, history, nber)
    else:
        _render_vs_street(current)


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


# ----------------------------------------------------------------- Under the Hood


def _render_under_hood(model: RecessionEnsemble, current: dict, history: pd.DataFrame, calibration: dict) -> None:
    submodel_probs = current["submodels"]
    cols = st.columns(5)
    for col, spec in zip(cols, SUBMODELS):
        if spec.name not in submodel_probs:
            continue
        prob = submodel_probs[spec.name]
        color = PALETTE["submodel"][spec.name]
        spark_series = history[spec.name].dropna().tail(180) if spec.name in history.columns else None
        spark = sparkline_svg(spark_series.values, color=color) if spark_series is not None and len(spark_series) else ""
        with col:
            st.markdown(
                metric_card(
                    label=spec.label,
                    value=f"{prob:.0f}",
                    unit="%",
                    risk_color_hex=color,
                    sparkline_html=spark,
                    subline="12m forward probability",
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Feature contributions · current observation</div>',
        unsafe_allow_html=True,
    )
    contributions: list[tuple[str, str, float, str]] = []
    for spec in SUBMODELS:
        drivers = current["drivers"].get(spec.name, [])
        color = PALETTE["submodel"][spec.name]
        for feat_name, contrib_pp in drivers:
            contributions.append((spec.label, feat_name, contrib_pp, color))

    if contributions:
        df = pd.DataFrame(contributions, columns=["submodel", "feature", "contribution", "color"])
        df = df.iloc[df["contribution"].abs().sort_values(ascending=False).index].head(15).reset_index(drop=True)
        fig = go.Figure(
            go.Bar(
                x=df["contribution"],
                y=[f"{r['feature']}  ·  {r['submodel']}" for _, r in df.iterrows()],
                orientation="h",
                marker=dict(color=df["color"], line=dict(width=0)),
                hovertemplate="%{y}<br>contribution: %{x:+.2f} pp<extra></extra>",
            )
        )
        fig.add_vline(x=0, line=dict(color="#3d4754", width=1))
        fig.update_xaxes(title="Contribution to probability (pp)")
        fig.update_yaxes(autorange="reversed")
        apply_template(fig, height=420, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="label-small">Reliability diagram</div>', unsafe_allow_html=True)
        fig = reliability_diagram(calibration.get("reliability_curve", pd.DataFrame()))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        brier = calibration.get("brier", float("nan"))
        auc = calibration.get("auc", float("nan"))
        rows = [
            ("Brier score", f"{brier:.4f}" if not np.isnan(brier) else "—"),
            ("AUC", f"{auc:.3f}" if not np.isnan(auc) else "—"),
            ("Fitted submodels", f"{len(submodel_probs)} / 5"),
            ("Sample start", str(model.START_DATE.date())),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{label}</span>'
            f'<span class="value">{value}</span></div>'
            for label, value in rows
        )
        st.markdown(
            '<div class="panel"><div class="panel-header"><span>Calibration</span></div>'
            f'<div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------- The Reading


def _render_reading(current: dict, history: pd.DataFrame, nber: pd.Series) -> None:
    ensemble_now = current["ensemble"]
    submodels = current["submodels"]

    if "ensemble" in history.columns:
        fig = line_chart(
            history["ensemble"].rename("Ensemble"),
            color=PALETTE["accent"],
            height=420,
            nber=nber,
            fill=True,
            yaxis_title="Recession probability (%)",
        )
        fig.add_hline(y=50, line=dict(color="#3d4754", width=1, dash="dot"))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="label-small">Today\'s read</div>', unsafe_allow_html=True)
        text = _plain_english(ensemble_now, submodels)
        st.markdown(
            f'<div class="panel"><div class="panel-body" '
            f'style="font-family:\'Fraunces\',serif;font-size:15px;line-height:1.7;color:{PALETTE["text_primary"]};">'
            f"{text}</div></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="label-small">Driver decomposition</div>', unsafe_allow_html=True)
        push_up, pull_down = _split_drivers(current.get("drivers", {}))
        watch_next = _watch_next(submodels)
        block = _driver_block("Push up", push_up, PALETTE["risk_high"])
        block += _driver_block("Pull down", pull_down, PALETTE["risk_low"])
        block += _driver_block("Watch next", watch_next, PALETTE["accent"])
        st.markdown(
            f'<div class="panel"><div class="panel-body">{block}</div></div>',
            unsafe_allow_html=True,
        )


def _driver_block(title: str, items: list[tuple[str, str]], color: str) -> str:
    rows = "".join(
        f'<div class="submodel-row"><span class="name">{name}</span>'
        f'<span class="value" style="color:{color};">{value}</span></div>'
        for name, value in items
    ) or '<div class="submodel-row"><span class="name">—</span><span class="value">—</span></div>'
    return (
        f'<div style="margin-bottom:10px;">'
        f'<div class="label-tiny" style="margin-bottom:4px;">{title}</div>'
        f"{rows}</div>"
    )


def _plain_english(ensemble_now: float, submodels: dict[str, float]) -> str:
    if not np.isfinite(ensemble_now):
        return "The ensemble could not be evaluated for the most recent observation."
    band = (
        "subdued" if ensemble_now < 20
        else "elevated" if ensemble_now < 40
        else "high" if ensemble_now < 60
        else "extreme"
    )
    spread = max(submodels.values()) - min(submodels.values()) if submodels else 0.0
    agree = "broad agreement" if spread < 15 else "noticeable disagreement"
    top = max(submodels, key=submodels.get) if submodels else None
    bottom = min(submodels, key=submodels.get) if submodels else None
    parts = [
        f"The 12-month forward recession probability is <b>{ensemble_now:.0f}%</b>, "
        f"which we classify as <b>{band}</b>."
    ]
    if top and bottom and top != bottom:
        parts.append(
            f"There is {agree} across submodels: the <b>{top}</b> reading sits at "
            f"{submodels[top]:.0f}% while <b>{bottom}</b> sits at {submodels[bottom]:.0f}%."
        )
    parts.append(
        "Submodel divergence is informative — when the curve and credit lenses disagree, "
        "the ensemble's hedged answer tends to be more accurate than either alone."
    )
    return " ".join(parts)


def _split_drivers(drivers: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    flat: list[tuple[str, float]] = []
    for sub, items in drivers.items():
        for feat, contrib in items:
            flat.append((feat, contrib))
    flat.sort(key=lambda t: t[1], reverse=True)
    push_up = [(name, f"+{v:.2f} pp") for name, v in flat if v > 0][:3]
    pull_down = [(name, f"{v:.2f} pp") for name, v in flat if v < 0][-3:][::-1]
    return push_up, pull_down


def _watch_next(submodels: dict[str, float]) -> list[tuple[str, str]]:
    if not submodels:
        return []
    sorted_subs = sorted(submodels.items(), key=lambda t: t[1], reverse=True)
    return [(name, f"{val:.0f}%") for name, val in sorted_subs[:3]]


# ----------------------------------------------------------------- vs. Street


def _render_vs_street(current: dict) -> None:
    ensemble_now = current["ensemble"]
    try:
        street = pd.read_csv(_STREET_PATH, parse_dates=["date"])
    except FileNotFoundError:
        st.info("Street estimates file not found.")
        return
    if street.empty:
        st.info("No street estimates available.")
        return

    latest = street.sort_values("date").iloc[-1]
    rows = {
        "This ensemble": ensemble_now,
        "Cleveland Fed": float(latest["cleveland_fed"]),
        "NY Fed": float(latest["ny_fed"]),
        "Bloomberg": float(latest["bloomberg"]),
        "Goldman Sachs": float(latest["goldman"]),
    }
    series = pd.Series(rows).sort_values()

    colors = [PALETTE["accent"] if name == "This ensemble" else PALETTE["text_muted"] for name in series.index]
    fig = go.Figure(
        go.Bar(
            x=series.values,
            y=series.index,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in series.values],
            textposition="outside",
            textfont=dict(color=PALETTE["text_primary"]),
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title="12-month recession probability (%)", range=[0, max(series.max() * 1.25, 50)])
    apply_template(fig, height=320, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="panel"><div class="panel-header"><span>How to read this</span></div>'
        f'<div class="panel-body" style="font-size:12px;color:{PALETTE["text_primary"]};line-height:1.6;">'
        "Street estimates (Cleveland/NY Fed, Bloomberg, Goldman) are pulled from public "
        "research and refreshed manually — see <code>data/street_estimates.csv</code>. "
        "Our ensemble tends to print below the Fed yield-curve models because labor and "
        "credit conditions damp the curve signal, and above pure equity-vol indicators "
        "when sentiment is unusually calm relative to the rates picture."
        "</div></div>",
        unsafe_allow_html=True,
    )
