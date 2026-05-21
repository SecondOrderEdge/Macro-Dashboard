"""Policy Path view — market-implied FOMC rate expectations.

Sourced from the Atlanta Fed's Market Probability Tracker, which backs out the
probability distribution of the policy rate after each upcoming quarterly
contract expiry from CME options on SOFR futures. This is a forward-looking,
market-priced complement to the dashboard's spot-rate and term-structure views:
it answers "where does the market think the FOMC is going?" rather than "where
are rates now?".
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.market_probability import (
    bucket_matrix,
    directional_probs,
    latest_snapshot,
    nearest_snapshot,
    rate_path,
)
from src.ui.components import apply_template
from src.ui.theme import PALETTE


def _fade(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render(market_prob: pd.DataFrame, nber: pd.Series | None = None) -> None:
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        "<b>Market-implied policy path.</b> The Atlanta Fed's Market Probability Tracker "
        "infers, from CME options on SOFR futures, the probability distribution of the FOMC "
        "policy rate after each upcoming quarterly contract. This is what the market is "
        "<i>pricing</i> — distinct from the spot rate and term structure on the Yield Curve tab."
        "</div></div>",
        unsafe_allow_html=True,
    )

    if market_prob is None or market_prob.empty:
        st.info(
            "Market Probability Tracker data unavailable. Refresh "
            "`data/market_probability_tracker.csv` with a current export from "
            "https://www.atlantafed.org/cenfis/market-probability-tracker."
        )
        return

    latest = latest_snapshot(market_prob)
    snap = latest["snapshot_date"].max()
    rp = rate_path(market_prob)
    dirs = directional_probs(market_prob)
    dirs_latest = dirs[dirs["snapshot_date"] == snap].sort_values("meeting_date") if not dirs.empty else dirs

    _cards(rp, dirs_latest, snap)
    _fan_chart(rp)
    _path_comparison(market_prob, snap)
    _heatmap(market_prob, snap)
    _how_to_read(snap)


def _cards(rp: pd.DataFrame, dirs_latest: pd.DataFrame, snap: pd.Timestamp) -> None:
    if rp.empty:
        return
    front_meeting = rp.index.min()
    far_meeting = rp.index.max()
    front_rate = float(rp.loc[front_meeting, "mean"]) / 100.0
    far_rate = float(rp.loc[far_meeting, "mean"]) / 100.0

    hike = cut = float("nan")
    if not dirs_latest.empty:
        row = dirs_latest.iloc[0]
        hike = float(row["prob_hike"]) if pd.notna(row["prob_hike"]) else float("nan")
        cut = float(row["prob_cut"]) if pd.notna(row["prob_cut"]) else float("nan")

    if pd.notna(hike) and pd.notna(cut):
        if hike > cut + 5:
            lean, lean_color = "HIKES PRICED", PALETTE["risk_elevated"]
        elif cut > hike + 5:
            lean, lean_color = "CUTS PRICED", PALETTE["risk_low"]
        else:
            lean, lean_color = "ON HOLD", PALETTE["text_muted"]
    else:
        lean, lean_color = "—", PALETTE["text_muted"]

    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            '<div class="panel" style="height:100%;">'
            '<div class="panel-header"><span>Implied rate · next meeting</span>'
            f'<span class="risk-badge" style="color:{lean_color};">{lean}</span></div>'
            '<div class="panel-body">'
            f'<div class="metric-big data-font" style="color:{PALETTE["accent"]};">{front_rate:.2f}<span class="metric-unit">%</span></div>'
            f'<div class="metric-sub">mean · {front_meeting.strftime("%b %Y")} · as of {snap.strftime("%d %b %Y")}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        hike_s = f"{hike:.0f}%" if pd.notna(hike) else "—"
        cut_s = f"{cut:.0f}%" if pd.notna(cut) else "—"
        st.markdown(
            '<div class="panel" style="height:100%;">'
            '<div class="panel-header"><span>Next-meeting odds</span></div>'
            '<div class="panel-body">'
            f'<div class="submodel-row"><span class="name" style="color:{PALETTE["risk_elevated"]};">Hike</span><span class="value">{hike_s}</span></div>'
            f'<div class="submodel-row"><span class="name" style="color:{PALETTE["risk_low"]};">Cut</span><span class="value">{cut_s}</span></div>'
            f'<div class="metric-sub">{front_meeting.strftime("%b %Y")} meeting</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            '<div class="panel" style="height:100%;">'
            '<div class="panel-header"><span>Implied rate · longest horizon</span></div>'
            '<div class="panel-body">'
            f'<div class="metric-big data-font" style="color:{PALETTE["text_primary"]};">{far_rate:.2f}<span class="metric-unit">%</span></div>'
            f'<div class="metric-sub">mean · {far_meeting.strftime("%b %Y")}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )


def _fan_chart(rp: pd.DataFrame) -> None:
    if rp.empty:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:24px;">Implied policy-rate path · latest snapshot, '
        '25th–75th percentile band</div>',
        unsafe_allow_html=True,
    )
    band = rp.dropna(subset=["p25", "p75"])
    fig = go.Figure()
    if not band.empty:
        fig.add_trace(
            go.Scatter(
                x=band.index, y=(band["p75"] / 100).values, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip", name="p75",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band.index, y=(band["p25"] / 100).values, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=_fade(PALETTE["accent"], 0.12),
                name="25th–75th pct", hoverinfo="skip",
            )
        )
    mean = rp.dropna(subset=["mean"])
    fig.add_trace(
        go.Scatter(
            x=mean.index, y=(mean["mean"] / 100).values, mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=1.8), marker=dict(size=5, color=PALETTE["accent"]),
            name="Mean",
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>mean</extra>",
        )
    )
    mode = rp.dropna(subset=["mode"])
    if not mode.empty:
        fig.add_trace(
            go.Scatter(
                x=mode.index, y=(mode["mode"] / 100).values, mode="lines",
                line=dict(color=PALETTE["text_muted"], width=1.0, dash="dot"),
                name="Mode",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>mode</extra>",
            )
        )
    fig.update_yaxes(title="Implied policy rate (%)")
    apply_template(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


def _path_comparison(df: pd.DataFrame, snap: pd.Timestamp) -> None:
    targets = [
        (snap, "Latest", PALETTE["accent"], 2.0),
        (snap - pd.Timedelta(days=30), "~1mo ago", PALETTE["submodel"]["labor"], 1.2),
        (snap - pd.Timedelta(days=90), "~3mo ago", PALETTE["submodel"]["sentiment"], 1.2),
        (snap - pd.Timedelta(days=180), "~6mo ago", PALETTE["text_muted"], 1.0),
    ]
    seen: set[pd.Timestamp] = set()
    traces = []
    for target, label, color, width in targets:
        s = nearest_snapshot(df, target)
        if s is None or s in seen:
            continue
        seen.add(s)
        rp = rate_path(df, s).dropna(subset=["mean"])
        if rp.empty:
            continue
        traces.append((s, label, color, width, rp))

    if len(traces) < 2:
        return

    st.markdown(
        '<div class="label-small" style="margin-top:24px;">How the implied path has shifted · '
        'mean rate by meeting, across recent snapshots</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    for s, label, color, width, rp in traces:
        fig.add_trace(
            go.Scatter(
                x=rp.index, y=(rp["mean"] / 100).values, mode="lines",
                line=dict(color=color, width=width),
                name=f"{label} ({s.strftime('%d %b %Y')})",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>" + label + "</extra>",
            )
        )
    fig.update_yaxes(title="Implied policy rate (%)")
    apply_template(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


def _heatmap(df: pd.DataFrame, snap: pd.Timestamp) -> None:
    mat = bucket_matrix(df, snap)
    if mat.empty:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:24px;">Probability by target range · '
        'latest snapshot (%)</div>',
        unsafe_allow_html=True,
    )
    x_labels = [d.strftime("%b %Y") for d in mat.columns]
    colorscale = [[0.0, "#0a0d12"], [0.5, "#7a5a3a"], [1.0, PALETTE["accent"]]]
    fig = go.Figure(
        go.Heatmap(
            z=mat.values,
            x=x_labels,
            y=mat.index.tolist(),
            colorscale=colorscale,
            hoverongaps=False,
            colorbar=dict(title="%", outlinewidth=0, tickfont=dict(color=PALETTE["text_muted"], size=9)),
            hovertemplate="%{x}<br>range %{y}<br>p = %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(title="Target range (%)", autorange="reversed")
    fig.update_xaxes(title="Meeting / contract")
    apply_template(fig, height=420, show_legend=False)
    st.plotly_chart(fig, use_container_width=True)


def _how_to_read(snap: pd.Timestamp) -> None:
    st.markdown(
        '<div class="panel"><div class="panel-header"><span>How to read this</span></div>'
        f'<div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        "<p>The Atlanta Fed estimates a full probability distribution for the policy rate after "
        "each upcoming quarterly SOFR contract, from CME options prices. The <b>fan chart</b> shows "
        "the mean expected path with its 25th-75th percentile band — the band widens with horizon "
        "because the market is less certain further out. The <b>shift chart</b> overlays the path "
        "from recent snapshots so you can see how expectations have re-priced. The <b>heatmap</b> "
        "shows the probability mass on each 25bp target range per meeting.</p>"
        f'<p style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:8px;">'
        f'Snapshot as of {snap.strftime("%d %b %Y")}. This is a bundled export, refreshed manually from the '
        f'<a href="https://www.atlantafed.org/cenfis/market-probability-tracker" style="color:{PALETTE["accent"]};">'
        "Atlanta Fed Market Probability Tracker</a>; source data derived from CME Group options on SOFR "
        "futures. Not investment advice.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
