"""Yield curve view: term structure, by-maturity drill-down, spreads, inversions.

Four sub-tabs:
- Term Structure — current curve vs. 3m/12m ago, plus spread cards.
- By Maturity   — pick a maturity; see history, distribution, stats.
- Spreads       — for each benchmark spread: history, distribution, recession-conditional.
- Inversions    — current stats + table of historical episodes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from src.models.yield_curve import YieldCurve, _runs
from src.ui.components import (
    add_recession_shading,
    apply_template,
    distribution_chart,
    metric_card,
    percentile_rank,
    sparkline_svg,
    stats_table_html,
)
from src.ui.theme import PALETTE


_MATURITY_OPTIONS = [
    ("1 Month", "DGS1MO"),
    ("3 Month", "DGS3MO"),
    ("6 Month", "DGS6MO"),
    ("1 Year", "DGS1"),
    ("2 Year", "DGS2"),
    ("3 Year", "DGS3"),
    ("5 Year", "DGS5"),
    ("7 Year", "DGS7"),
    ("10 Year", "DGS10"),
    ("20 Year", "DGS20"),
    ("30 Year", "DGS30"),
]


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


def render(panel: pd.DataFrame, nber: pd.Series) -> None:
    yc = YieldCurve(panel)
    spreads = yc.spreads_history()
    if spreads.empty:
        st.warning("No yield curve data available.")
        return

    stats = yc.inversion_stats(nber)
    _render_spread_cards(spreads, stats)

    selected = option_menu(
        menu_title=None,
        options=["Term Structure", "By Maturity", "Spreads", "Inversions"],
        icons=["bezier2", "search", "bar-chart-line", "exclamation-triangle"],
        orientation="horizontal",
        default_index=0,
        key="curve_tab",
        styles=_TAB_STYLES,
    )

    if selected == "Term Structure":
        _render_term_structure(yc)
    elif selected == "By Maturity":
        _render_by_maturity(panel, nber)
    elif selected == "Spreads":
        _render_spreads_tab(spreads, nber)
    elif selected == "Inversions":
        _render_inversions_tab(spreads, nber, stats)


# ------------------------------------------------------------------ top cards


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
        pct = percentile_rank(series, latest)
        subline = f"{pct:.0f}th percentile of history"
        if key == "spread_10y3m" and stats.get("months_inverted", 0) > 0:
            subline = f"{stats['months_inverted']} consec. inverted months · " + subline
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


# ---------------------------------------------------------------- term structure


def _render_term_structure(yc: YieldCurve) -> None:
    st.markdown('<div class="label-small">Term structure · current vs. 3m / 12m ago</div>', unsafe_allow_html=True)
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
                marker=dict(size=8 if col == "current" else 6, color=color),
                hovertemplate=f"{name}<br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_yaxes(title="Yield (%)")
    fig.update_xaxes(title="Maturity")
    apply_template(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Steepness shifts: 10Y-2Y change vs 12 months ago
    if {"current", "m12"}.issubset(ts.columns):
        try:
            cur_10 = float(ts.set_index("maturity").loc["10Y", "current"])
            cur_2 = float(ts.set_index("maturity").loc["2Y", "current"])
            old_10 = float(ts.set_index("maturity").loc["10Y", "m12"])
            old_2 = float(ts.set_index("maturity").loc["2Y", "m12"])
            cur_slope = cur_10 - cur_2
            old_slope = old_10 - old_2
            steepening = cur_slope - old_slope
            direction = "steepened" if steepening > 0 else "flattened"
            st.markdown(
                f'<div class="panel"><div class="panel-body" style="font-size:13px;color:{PALETTE["text_primary"]};line-height:1.6;">'
                f"Over the last 12 months the curve has <b>{direction}</b> by "
                f"<b>{abs(steepening):.2f}pp</b> (10Y−2Y went from {old_slope:+.2f}pp to {cur_slope:+.2f}pp). "
                f"The front end (2Y) {'fell' if cur_2 < old_2 else 'rose'} by {abs(cur_2 - old_2):.2f}pp; "
                f"the belly (10Y) {'fell' if cur_10 < old_10 else 'rose'} by {abs(cur_10 - old_10):.2f}pp."
                "</div></div>",
                unsafe_allow_html=True,
            )
        except (KeyError, ValueError):
            pass


# ---------------------------------------------------------------- by maturity


def _render_by_maturity(panel: pd.DataFrame, nber: pd.Series) -> None:
    available = [(label, col) for label, col in _MATURITY_OPTIONS if col in panel.columns]
    if not available:
        st.warning("No yield series available.")
        return

    labels = [a[0] for a in available]
    default_label = "10 Year" if "10 Year" in labels else labels[0]
    chosen_label = st.selectbox(
        "Maturity",
        labels,
        index=labels.index(default_label),
        key="curve_maturity",
        label_visibility="collapsed",
    )
    col = dict(available)[chosen_label]
    series = panel[col].dropna().sort_index()
    if series.empty:
        st.info(f"No history for {chosen_label}.")
        return

    today = float(series.iloc[-1])
    today_date = series.index[-1]

    # Key reference points
    yr_ago = _asof(series, today_date - pd.DateOffset(years=1))
    m3_ago = _asof(series, today_date - pd.DateOffset(months=3))
    m1_ago = _asof(series, today_date - pd.DateOffset(months=1))
    max_v, max_d = float(series.max()), series.idxmax()
    min_v, min_d = float(series.min()), series.idxmin()
    pct = percentile_rank(series, today)

    color = PALETTE["accent"]
    cols = st.columns([1, 2])
    with cols[0]:
        spark = sparkline_svg(series.tail(252).values, color=color)
        subline = f"{pct:.0f}th percentile · since {series.index[0].year}"
        st.markdown(
            metric_card(
                label=f"{chosen_label} Treasury",
                value=f"{today:.2f}",
                unit="%",
                risk_color_hex=color,
                sparkline_html=spark,
                subline=subline,
            ),
            unsafe_allow_html=True,
        )

        rows = [
            ("As of", today_date.strftime("%Y-%m-%d")),
            ("1 month ago", f"{m1_ago:.2f}%" if np.isfinite(m1_ago) else "—"),
            ("3 months ago", f"{m3_ago:.2f}%" if np.isfinite(m3_ago) else "—"),
            ("12 months ago", f"{yr_ago:.2f}%" if np.isfinite(yr_ago) else "—"),
            ("All-time high", f"{max_v:.2f}% ({max_d.strftime('%Y-%m')})"),
            ("All-time low",  f"{min_v:.2f}% ({min_d.strftime('%Y-%m')})"),
            ("Median",        f"{series.median():.2f}%"),
            ("Std dev",       f"{series.std():.2f}pp"),
        ]
        st.markdown(stats_table_html(rows), unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="label-small">History · all available data</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.values, mode="lines",
                line=dict(color=color, width=1.2),
                fill="tozeroy", fillcolor="rgba(212,165,116,0.06)",
                name=chosen_label,
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
        add_recession_shading(fig, nber)
        fig.update_yaxes(title="Yield (%)")
        apply_template(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Distribution
    st.markdown(
        f'<div class="label-small" style="margin-top:8px;">Distribution · {chosen_label} '
        f'today vs. history</div>',
        unsafe_allow_html=True,
    )
    fig = distribution_chart(
        series,
        today_value=today,
        xaxis_title="Yield (%)",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain-English read
    bucket = (
        "an unusually low" if pct < 10
        else "a low" if pct < 33
        else "a middling" if pct < 67
        else "a high" if pct < 90
        else "an unusually high"
    )
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:13px;color:{PALETTE["text_primary"]};line-height:1.6;">'
        f"The current {chosen_label} yield of <b>{today:.2f}%</b> sits at the "
        f"<b>{pct:.0f}th percentile</b> of its history — {bucket} reading by historical standards. "
        f"It has moved <b>{(today - yr_ago):+.2f}pp</b> over the last year and "
        f"<b>{(today - m3_ago):+.2f}pp</b> over the last three months."
        "</div></div>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------- spreads


def _render_spreads_tab(spreads: pd.DataFrame, nber: pd.Series) -> None:
    nber_daily = _to_daily_mask(nber)

    labels = [
        ("spread_10y3m", "10Y − 3M", "The Estrella–Mishkin benchmark; the single best-tested recession signal."),
        ("spread_10y2y", "10Y − 2Y", "Quoted as the curve in market commentary; cleaner of front-end policy noise."),
        ("spread_5y2y",  "5Y − 2Y",  "Belly steepness; reflects medium-term growth expectations relative to policy."),
    ]

    st.markdown('<div class="label-small">Spread history · with NBER recessions shaded</div>', unsafe_allow_html=True)
    fig = go.Figure()
    palette = {
        "spread_10y3m": PALETTE["accent"],
        "spread_10y2y": PALETTE["submodel"]["labor"],
        "spread_5y2y":  PALETTE["submodel"]["sentiment"],
    }
    for key, label, _ in labels:
        if key not in spreads.columns:
            continue
        s = spreads[key].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines",
                line=dict(color=palette[key], width=1.2),
                name=label,
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}pp<extra>" + label + "</extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="#3d4754", width=1, dash="dot"))
    add_recession_shading(fig, nber)
    fig.update_yaxes(title="Spread (pp)")
    apply_template(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    for key, label, gloss in labels:
        if key not in spreads.columns:
            continue
        s = spreads[key].dropna()
        if s.empty:
            continue
        today = float(s.iloc[-1])
        pct = percentile_rank(s, today)
        median = float(s.median())
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        rec_mask = nber_daily.reindex(s.index).fillna(False).astype(bool)
        rec_median = float(s.loc[rec_mask].median()) if rec_mask.any() else float("nan")
        exp_median = float(s.loc[~rec_mask].median())

        st.markdown(
            f'<div class="label-small" style="margin-top:16px;">{label} · distribution</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:{PALETTE["text_muted"]};font-size:11px;margin-bottom:6px;">{gloss}</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns([2, 1])
        with cols[0]:
            fig = distribution_chart(
                s,
                today_value=today,
                xaxis_title=f"{label} (pp)",
                conditional=rec_mask,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with cols[1]:
            rows = [
                ("Today",                  f"{today:+.2f} pp"),
                ("Percentile rank",        f"{pct:.0f}th"),
                ("Median (all history)",   f"{median:+.2f} pp"),
                ("IQR (25th–75th)",        f"{q1:+.2f} → {q3:+.2f}"),
                ("Median during NBER",     f"{rec_median:+.2f} pp" if np.isfinite(rec_median) else "—"),
                ("Median outside NBER",    f"{exp_median:+.2f} pp"),
                ("Sample start",           s.index[0].strftime("%Y-%m")),
                ("n observations",         f"{len(s):,}"),
            ]
            st.markdown(stats_table_html(rows), unsafe_allow_html=True)


# ----------------------------------------------------------------- inversions


def _render_inversions_tab(spreads: pd.DataFrame, nber: pd.Series, stats: dict) -> None:
    if "spread_10y3m" not in spreads.columns:
        st.warning("10Y−3M spread unavailable.")
        return

    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="label-small">10Y−3M spread · inversion episodes shaded</div>', unsafe_allow_html=True)
        s = spreads["spread_10y3m"].dropna()
        monthly = s.resample("ME").mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly.index, y=monthly.values, mode="lines",
                line=dict(color=PALETTE["accent"], width=1.2),
                name="10Y − 3M (monthly)",
                hovertemplate="%{x|%b %Y}<br>%{y:+.2f}pp<extra></extra>",
            )
        )
        # Shade inversion episodes
        inverted = monthly < 0
        for start_i, end_i in _runs(inverted):
            if end_i - start_i + 1 <= 3:
                continue
            fig.add_vrect(
                x0=monthly.index[start_i], x1=monthly.index[end_i],
                fillcolor=PALETTE["risk_critical"], opacity=0.18, line_width=0,
                layer="below",
            )
        fig.add_hline(y=0, line=dict(color="#3d4754", width=1, dash="dot"))
        add_recession_shading(fig, nber)
        fig.update_yaxes(title="Spread (pp)")
        apply_template(fig, height=420, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="label-small">Current run</div>', unsafe_allow_html=True)
        rows = [
            ("Months inverted", str(stats.get("months_inverted", 0))),
            ("Max depth (current)", f"{stats['max_depth_current']:+.2f} pp" if np.isfinite(stats.get('max_depth_current', np.nan)) else "—"),
            ("Avg lead to NBER peak", f"{stats['avg_lead_to_recession']:.0f} months" if np.isfinite(stats.get('avg_lead_to_recession', np.nan)) else "—"),
            ("Hit rate (>3m episodes)", f"{stats['hit_rate'][0]} / {stats['hit_rate'][1]}" if stats.get("hit_rate", (0, 0))[1] else "—"),
        ]
        st.markdown(stats_table_html(rows), unsafe_allow_html=True)

        interp = _interpretation(
            stats.get("months_inverted", 0),
            stats.get("max_depth_current", float("nan")),
            *stats.get("hit_rate", (0, 0)),
        )
        st.markdown(
            f'<div class="panel"><div class="panel-header"><span>Interpretation</span></div>'
            f'<div class="panel-body" style="font-size:12px;color:{PALETTE["text_primary"]};line-height:1.6;">{interp}</div></div>',
            unsafe_allow_html=True,
        )

    # Historical episodes table
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">Historical inversion episodes · 10Y−3M</div>',
        unsafe_allow_html=True,
    )
    episodes = _episode_table(spreads["spread_10y3m"], nber)
    if episodes.empty:
        st.info("No sustained inversions on record.")
    else:
        st.dataframe(episodes, hide_index=True, use_container_width=True)


def _episode_table(spread: pd.Series, nber: pd.Series) -> pd.DataFrame:
    monthly = spread.dropna().resample("ME").mean()
    inverted = monthly < 0
    nber_monthly = nber.copy()
    nber_monthly.index = pd.DatetimeIndex(nber_monthly.index).to_period("M").to_timestamp()
    # NBER peaks
    peaks: list[pd.Timestamp] = []
    prev = False
    for ts, val in nber_monthly.items():
        if val and not prev:
            peaks.append(ts)
        prev = bool(val)

    rows = []
    for start_i, end_i in _runs(inverted):
        if end_i - start_i + 1 <= 3:
            continue
        start = monthly.index[start_i]
        end = monthly.index[end_i]
        depth = float(monthly.iloc[start_i : end_i + 1].min())
        # Lead to next NBER peak within 36 months
        future = [p for p in peaks if 0 <= (p - start).days / 30.5 <= 36]
        if future:
            lead = (future[0] - start).days / 30.5
            outcome = f"recession ({future[0].strftime('%b %Y')})"
            lead_str = f"{lead:.0f}"
        else:
            outcome = "no recession in 36m"
            lead_str = "—"
        rows.append(
            {
                "started": start.strftime("%b %Y"),
                "ended": end.strftime("%b %Y"),
                "duration (months)": end_i - start_i + 1,
                "max depth (pp)": f"{depth:+.2f}",
                "lead to NBER peak (months)": lead_str,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


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


# ----------------------------------------------------------------- helpers


def _asof(series: pd.Series, ts: pd.Timestamp) -> float:
    try:
        s = series.loc[:ts]
    except KeyError:
        return float("nan")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])


def _to_daily_mask(nber_monthly: pd.Series) -> pd.Series:
    """Convert the month-indexed NBER flag to a daily reindexed boolean series."""
    if nber_monthly.empty:
        return nber_monthly
    daily = nber_monthly.copy()
    daily.index = pd.DatetimeIndex(daily.index)
    daily_idx = pd.date_range(daily.index.min(), pd.Timestamp.today(), freq="D")
    return daily.reindex(daily_idx, method="ffill").astype(bool)
