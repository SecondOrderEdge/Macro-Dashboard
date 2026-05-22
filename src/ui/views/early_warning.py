"""Early-warning view — the "how close is trouble?" ladder.

A lead-time-ordered staircase of binary warning signals, slowest/earliest at the
top (yield-curve inversion) down to fastest/latest at the bottom (vol spikes).
How far down the lit rungs reach tells you how close trouble is — and it covers
both slow macro trouble (climbs from the top) and fast financial trouble (lights
the bottom directly). Descriptive, not a forecast.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.early_warning import build_ladder, ladder_summary
from src.ui.theme import PALETTE

_SEV_COLOR = {
    "low": PALETTE["risk_low"],
    "elevated": PALETTE["risk_elevated"],
    "high": PALETTE["risk_high"],
    "critical": PALETTE["risk_critical"],
}
_TRACK_COLOR = {"macro": PALETTE["text_muted"], "financial": PALETTE["submodel"]["sentiment"]}


def render(panel: pd.DataFrame, probit: dict | None = None, lame=None) -> None:
    st.markdown(
        '<div class="label-small" style="margin-top:8px;">Early warning · how close is trouble?</div>',
        unsafe_allow_html=True,
    )

    rungs = build_ladder(panel, probit, lame)
    if not rungs:
        st.markdown(
            f'<div class="panel"><div class="panel-body" style="font-size:12px;color:{PALETTE["text_muted"]};">'
            "Early-warning signals temporarily unavailable (this view reads live FRED series — "
            "yield curve, SLOOS, housing, Sahm, NFCI, VIX — plus the recession ensemble)."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    summary = ladder_summary(rungs)
    _row_summary(summary)
    _row_ladder(rungs)
    _row_interpretation()


def _row_summary(summary: dict) -> None:
    color = _SEV_COLOR.get(summary["severity"], PALETTE["text_muted"])
    deepest = summary.get("deepest")
    deepest_html = (
        f'<div class="metric-sub">deepest signal lit · {deepest}</div>' if deepest else
        '<div class="metric-sub">no warning signals lit</div>'
    )
    st.markdown(
        '<div class="panel"><div class="panel-header"><span>Trouble proximity</span>'
        f'<span class="risk-badge" style="color:{color};">{summary["stage"]}</span></div>'
        '<div class="panel-body">'
        f'<div class="metric-big data-font" style="color:{color};">{summary["n_lit"]}'
        f'<span class="metric-unit">/ {summary["n_total"]} lit</span></div>'
        f'{deepest_html}'
        '</div></div>',
        unsafe_allow_html=True,
    )


def _row_ladder(rungs: list[dict]) -> None:
    st.markdown(
        '<div class="label-tiny" style="margin-top:6px;">Escalation ladder · earliest / slowest at top → '
        'fastest / latest at bottom</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for r in rungs:
        lit = r["lit"]
        color = _SEV_COLOR.get(r["severity"], PALETTE["text_muted"]) if lit else PALETTE["panel_border"]
        name_color = PALETTE["text_primary"] if lit else PALETTE["text_muted"]
        marker = "●" if lit else "○"
        tcolor = _TRACK_COLOR.get(r["track"], PALETTE["text_muted"])
        rows.append(
            f'<div style="display:flex;align-items:center;gap:12px;padding:10px 12px;'
            f'border-left:3px solid {color};border-bottom:1px solid {PALETTE["panel_border"]};">'
            f'<span style="color:{color};font-size:14px;width:14px;">{marker}</span>'
            f'<div style="flex:1;">'
            f'<div style="color:{name_color};font-size:13px;">{r["label"]}'
            f'<span style="color:{tcolor};font-size:9px;letter-spacing:.12em;text-transform:uppercase;'
            f'margin-left:8px;">{r["track"]}</span></div>'
            f'<div style="color:{PALETTE["text_tiny"]};font-size:11px;margin-top:2px;">{r["detail"]}</div>'
            f'</div>'
            f'<div style="text-align:right;min-width:96px;">'
            f'<div style="color:{name_color};font-variant-numeric:tabular-nums;font-size:13px;">{r["value_str"]}</div>'
            f'<div style="color:{PALETTE["text_tiny"]};font-size:10px;letter-spacing:.1em;">lead {r["lead"]}</div>'
            f'</div></div>'
        )
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="padding:0;">{"".join(rows)}</div></div>',
        unsafe_allow_html=True,
    )


def _row_interpretation() -> None:
    st.markdown(
        '<div class="panel"><div class="panel-header"><span>How to read this</span></div>'
        f'<div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        "<p><b>What it is.</b> A checklist of warning signals that historically fire in a rough "
        "sequence, ordered by how far <i>ahead</i> they tend to lead. Reading top-to-bottom traces "
        "the path trouble usually travels; <b>how far down the lit rungs reach is how close trouble "
        "is.</b> Slow macro trouble climbs down from the top; fast financial trouble lights the "
        "bottom directly — so the ladder covers both.</p>"
        "<p><b>It is early warning, not a forecast.</b> Lead times are stylized averages that vary "
        "widely; thresholds are judgmental (shown on each rung); and the sequence is <b>not</b> "
        "deterministic. 2020 is the clean counterexample — an exogenous shock lit the bottom "
        "(conditions, vol) almost coincidentally, with no curve-led runway. Treat lit rungs as "
        "context, never as a countdown.</p>"
        f'<p style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:8px;">'
        "Inputs are the same series shown elsewhere in the app (Yield Curve, Credit, Labor, "
        "Pulse) plus the recession ensemble — this tab only re-frames them by lead time.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
