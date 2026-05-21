"""Growth view: real-time GDP, nowcast, drivers, and a coincident factor.

A "where is the economy *now*" complement to the forward-looking recession
engine. Surfaces authoritative FRED series — official real-GDP growth, Atlanta
Fed GDPNow, real GDI, BEA contributions to growth, the Weekly Economic Index —
plus one light original model: a z-scored coincident growth factor built from
monthly hard data. Everything degrades gracefully when FRED is unreachable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.gdp import (
    CONTRIBUTION_SERIES,
    LABELS,
    coincident_factor,
    factor_gdp_frame,
    factor_prob_frame,
    fetch_gdp_bundle,
    latest,
    pearson,
)
from src.ui.components import (
    add_recession_shading,
    apply_template,
    line_chart,
    metric_card,
    percentile_rank,
    signed_bar,
    sparkline_svg,
)
from src.ui.theme import PALETTE


def _factor_band(z: float) -> tuple[str, str]:
    """Coincident factor → label + severity (higher = stronger growth)."""
    if not np.isfinite(z):
        return "—", "elevated"
    if z >= 0.5:
        return "ABOVE TREND", "low"
    if z >= -0.5:
        return "AT TREND", "elevated"
    if z >= -1.5:
        return "BELOW TREND", "high"
    return "CONTRACTING", "critical"


_SEV_COLOR = {
    "low": PALETTE["risk_low"],
    "elevated": PALETTE["risk_elevated"],
    "high": PALETTE["risk_high"],
    "critical": PALETTE["risk_critical"],
}


def _robust_range(values, pad: float = 0.10, lo_q: float = 0.02, hi_q: float = 0.98):
    """A y-axis range that ignores extreme outliers (e.g. the 2020–21 COVID
    swings) so normal history stays readable. Returns ``None`` if too sparse."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return None
    lo, hi = float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q))
    if hi <= lo:
        return None
    span = hi - lo
    return [lo - pad * span, hi + pad * span]


def _focus_axes(fig, series: pd.Series, *, y_robust: bool = True):
    """Clamp the x-axis to the data's span (recession shading otherwise stretches
    it back to 1960) and optionally apply a robust, outlier-resistant y-range."""
    s = series.dropna()
    if s.empty:
        return fig
    fig.update_xaxes(range=[s.index.min(), s.index.max()])
    if y_robust:
        r = _robust_range(s.values)
        if r is not None:
            fig.update_yaxes(range=r)
    return fig


def _recession_prob_history(probit: dict | None) -> pd.Series:
    """12-month recession-probability history from the probit ensemble, or empty."""
    if not probit or "error" in probit:
        return pd.Series(dtype=float)
    hist = probit.get("ensemble_history")
    if hist is None or len(hist) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(hist).dropna()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def render(nber: pd.Series, probit: dict | None = None) -> None:
    bundle = fetch_gdp_bundle()
    head = bundle.get("headline", {})
    contrib = bundle.get("contributions", {})
    highfreq = bundle.get("highfreq", {})
    factor = coincident_factor()
    prob = _recession_prob_history(probit)

    nothing = not head and not contrib and not highfreq and factor["composite"].empty
    if nothing:
        _unavailable(bundle.get("log", []) + factor.get("log", []))
        return

    _row_headline(head, nber)
    _row_contributions(contrib)
    _row_highfreq_and_factor(highfreq, factor, nber)
    _row_factor_validation(factor, head)
    _row_growth_vs_risk(factor, prob, nber)
    _row_revisions()
    _interpretation(head, factor, prob)


# --------------------------------------------------------------- row: headline


def _row_headline(head: dict, nber: pd.Series) -> None:
    st.markdown(
        '<div class="label-small" style="margin-top:8px;">'
        'Current-quarter growth · official print vs nowcast</div>',
        unsafe_allow_html=True,
    )
    gdp = head.get("A191RL1Q225SBEA")
    nowcast = head.get("GDPNOW")
    gdi = head.get("A261RL1Q225SBEA")

    gdp_now, gdp_dt = latest(gdp)
    now_now, now_dt = latest(nowcast)
    gdi_now, _ = latest(gdi)

    # Lead with GDPNow if available (it's the live read), else the official print.
    if np.isfinite(now_now):
        lead_val, lead_dt, lead_label = now_now, now_dt, "GDPNow nowcast"
    else:
        lead_val, lead_dt, lead_label = gdp_now, gdp_dt, "Real GDP (latest)"

    color = PALETTE["risk_low"] if lead_val >= 1.0 else (
        PALETTE["risk_elevated"] if lead_val >= 0 else PALETTE["risk_critical"]
    )

    left, right = st.columns([1, 3])
    with left:
        spark = sparkline_svg(gdp.dropna().tail(16).values, color=color) if gdp is not None else ""
        subline = f"as of {lead_dt.strftime('%b %Y')}" if lead_dt is not None else "unavailable"
        st.markdown(
            metric_card(
                label=lead_label,
                value=f"{lead_val:+.1f}" if np.isfinite(lead_val) else "—",
                unit="% SAAR",
                risk_color_hex=color,
                sparkline_html=spark,
                subline=subline,
            ),
            unsafe_allow_html=True,
        )
        rows = []
        if np.isfinite(gdp_now):
            rows.append(("Real GDP · last print", f"{gdp_now:+.1f}% SAAR"))
        if np.isfinite(now_now):
            rows.append(("GDPNow · current Q", f"{now_now:+.1f}% SAAR"))
        if np.isfinite(gdi_now):
            rows.append(("Real GDI · last print", f"{gdi_now:+.1f}% SAAR"))
        if np.isfinite(gdp_now) and np.isfinite(gdi_now):
            rows.append(("GDP − GDI gap", f"{gdp_now - gdi_now:+.1f} pp"))
        if rows:
            body = "".join(
                f'<div class="submodel-row"><span class="name">{lbl}</span>'
                f'<span class="value">{val}</span></div>'
                for lbl, val in rows
            )
            st.markdown(
                f'<div class="panel"><div class="panel-body">{body}</div></div>',
                unsafe_allow_html=True,
            )

    with right:
        if gdp is not None and not gdp.dropna().empty:
            fig = line_chart(
                gdp.rename("Real GDP (% SAAR)"),
                color=PALETTE["accent"],
                nber=nber,
                zero_line=True,
                height=300,
                yaxis_title="% change, SAAR",
            )
            _focus_axes(fig, gdp)
            gser = gdp.dropna()
            xmax = gser.index.max()
            if np.isfinite(now_now) and now_dt is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[now_dt], y=[now_now], mode="markers",
                        marker=dict(color=PALETTE["risk_low"], size=9, symbol="diamond"),
                        name="GDPNow",
                        hovertemplate="GDPNow %{y:+.1f}%<extra></extra>",
                    )
                )
                xmax = max(xmax, now_dt)  # keep the current-quarter nowcast in view
            fig.update_xaxes(range=[gser.index.min(), xmax])
            st.plotly_chart(fig, use_container_width=True)
        else:
            _note("Official real-GDP series unavailable.")


# ---------------------------------------------------------- row: contributions


def _row_contributions(contrib: dict) -> None:
    if not contrib:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'What is driving growth · BEA contributions, latest quarter</div>',
        unsafe_allow_html=True,
    )
    order = [sid for sid, _ in CONTRIBUTION_SERIES if sid in contrib]
    vals, dt = {}, None
    for sid in order:
        v, d = latest(contrib[sid])
        if np.isfinite(v):
            vals[LABELS.get(sid, sid)] = v
            dt = d
    if not vals:
        return
    series = pd.Series(vals)

    left, right = st.columns([3, 1])
    with right:
        total = float(series.sum())
        rows = [(name, f"{v:+.2f} pp") for name, v in series.items()]
        rows.append(("Sum of shown", f"{total:+.2f} pp"))
        body = "".join(
            f'<div class="submodel-row"><span class="name">{lbl}</span>'
            f'<span class="value">{val}</span></div>'
            for lbl, val in rows
        )
        asof = f" · {dt.strftime('%b %Y')}" if dt is not None else ""
        st.markdown(
            f'<div class="label-tiny" style="margin-bottom:6px;">Contribution (pp){asof}</div>'
            f'<div class="panel"><div class="panel-body">{body}</div></div>',
            unsafe_allow_html=True,
        )
    with left:
        fig = signed_bar(
            series,
            pos_color=PALETTE["risk_low"],
            neg_color=PALETTE["risk_high"],
            height=260,
            xaxis_title="contribution to real GDP growth (pp, SAAR)",
        )
        st.plotly_chart(fig, use_container_width=True)

    _contributions_history(contrib, order)


_CONTRIB_COLORS = {
    "Consumption": PALETTE["risk_low"],
    "Investment": PALETTE["accent"],
    "Government": PALETTE["submodel"]["housing"],
    "Net exports": PALETTE["risk_high"],
    "Inventories": PALETTE["submodel"]["sentiment"],
}


def _contributions_history(contrib: dict, order: list[str], quarters: int = 16) -> None:
    """Stacked history of component contributions — shows the rotation of drivers."""
    cols = {}
    for sid in order:
        s = contrib[sid].dropna()
        if not s.empty:
            cols[LABELS.get(sid, sid)] = s
    if len(cols) < 2:
        return
    df = pd.concat(cols, axis=1).sort_index().tail(quarters)
    if df.empty:
        return
    fig = go.Figure()
    for name in cols:
        fig.add_trace(
            go.Bar(
                x=df.index, y=df[name].values, name=name,
                marker=dict(color=_CONTRIB_COLORS.get(name, PALETTE["text_muted"]), line=dict(width=0)),
                hovertemplate="%{x|%b %Y}<br>" + name + " %{y:+.2f} pp<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="#3d4754", width=1))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="contribution (pp, SAAR)")
    apply_template(fig, height=260)
    st.markdown(
        '<div class="label-tiny" style="margin-top:6px;">Driver rotation · stacked contributions, '
        f'last {quarters} quarters (bars sum to ≈ real GDP growth)</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------- row: high-frequency + factor


def _row_highfreq_and_factor(highfreq: dict, factor: dict, nber: pd.Series) -> None:
    composite = factor.get("composite", pd.Series(dtype=float))
    wei = highfreq.get("WEI")
    if (wei is None or wei.dropna().empty) and composite.empty:
        return

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'High-frequency &amp; coincident momentum</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)

    with left:
        if wei is not None and not wei.dropna().empty:
            w_now, w_dt = latest(wei)
            st.markdown(
                f'<div class="label-tiny">Weekly Economic Index'
                f'{" · " + w_dt.strftime("%d %b %Y") if w_dt is not None else ""}'
                f' · {w_now:+.2f}%</div>',
                unsafe_allow_html=True,
            )
            fig = line_chart(
                wei.rename("WEI"), color=PALETTE["accent"], nber=nber,
                zero_line=True, height=260, yaxis_title="WEI (≈ YoY GDP %)",
            )
            _focus_axes(fig, wei)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _note("Weekly Economic Index unavailable.")

    with right:
        if not composite.empty:
            z_now, z_dt = latest(composite)
            label, sev = _factor_band(z_now)
            color = _SEV_COLOR[sev]
            pct = percentile_rank(composite, z_now)
            st.markdown(
                f'<div class="label-tiny">Coincident growth factor · '
                f'{label} · {pct:.0f}th pct'
                f'{" · " + z_dt.strftime("%b %Y") if z_dt is not None else ""}</div>',
                unsafe_allow_html=True,
            )
            fig = line_chart(
                composite.rename("Coincident factor (z)"), color=color, nber=nber,
                zero_line=True, height=260, yaxis_title="standardized (z)",
            )
            _focus_axes(fig, composite)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _note("Coincident factor unavailable (needs monthly activity series).")


# ----------------------------------------------------- row: factor validation


def _row_factor_validation(factor: dict, head: dict) -> None:
    """Does the coincident factor actually track GDP? Scatter + correlation."""
    composite = factor.get("composite", pd.Series(dtype=float))
    gdp = head.get("A191RL1Q225SBEA")
    frame = factor_gdp_frame(composite, gdp)
    if len(frame) < 8:
        return
    r = pearson(frame, "factor", "gdp")

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'Does the factor track GDP? · validation</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=frame["factor"].values, y=frame["gdp"].values, mode="markers",
                marker=dict(color=PALETTE["accent"], size=6, opacity=0.6),
                name="quarter",
                hovertemplate="factor %{x:+.2f}<br>GDP %{y:+.1f}%<extra></extra>",
            )
        )
        fig.update_xaxes(title="coincident factor (z, quarterly avg)")
        fig.update_yaxes(title="real GDP (% SAAR)", range=_robust_range(frame["gdp"].values))
        apply_template(fig, height=300, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        strength = ("strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak")
        rows = [
            ("Correlation (r)", f"{r:+.2f}"),
            ("Strength", strength),
            ("Quarters", f"{len(frame)}"),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{lbl}</span>'
            f'<span class="value">{val}</span></div>'
            for lbl, val in rows
        )
        st.markdown(
            f'<div class="panel"><div class="panel-body">{body}'
            f'<div style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
            "Each point is a quarter: factor (x) vs the GDP print (y). A positive "
            "slope means the factor is a fair real-time read on growth.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------- row: growth vs risk


def _row_growth_vs_risk(factor: dict, prob: pd.Series, nber: pd.Series) -> None:
    """Tie the coincident factor to the forward-looking recession probability."""
    composite = factor.get("composite", pd.Series(dtype=float))
    if composite.empty or prob is None or prob.empty:
        return
    frame = factor_prob_frame(composite, prob)
    if len(frame) < 8:
        return
    r = pearson(frame, "factor", "prob")

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'Growth momentum vs forward recession risk</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=composite.index, y=composite.values, mode="lines",
                line=dict(color=PALETTE["accent"], width=1.4), name="Coincident factor (z)",
                hovertemplate="%{x|%b %Y}<br>factor %{y:+.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prob.index, y=prob.values, mode="lines",
                line=dict(color=PALETTE["risk_critical"], width=1.4), name="Recession prob (12m, %)",
                yaxis="y2",
                hovertemplate="%{x|%b %Y}<br>prob %{y:.0f}%<extra></extra>",
            )
        )
        add_recession_shading(fig, nber)
        apply_template(fig, height=320)
        fig.update_yaxes(title="coincident factor (z)")
        fig.update_layout(
            yaxis2=dict(
                title="recession prob (%)", overlaying="y", side="right",
                range=[0, 100], gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color=PALETTE["risk_critical"]),
            )
        )
        _focus_axes(fig, composite, y_robust=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        z_now, _ = latest(composite)
        p_now, _ = latest(prob)
        rows = [
            ("Factor today", f"{z_now:+.2f} z"),
            ("Recession prob", f"{p_now:.0f}%" if np.isfinite(p_now) else "—"),
            ("Correlation (r)", f"{r:+.2f}"),
        ]
        body = "".join(
            f'<div class="submodel-row"><span class="name">{lbl}</span>'
            f'<span class="value">{val}</span></div>'
            for lbl, val in rows
        )
        st.markdown(
            f'<div class="panel"><div class="panel-body">{body}'
            f'<div style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
            "A negative r is expected: weak current growth coincides with elevated "
            "forward recession risk. Growth momentum is contemporaneous; the "
            "probability looks 12 months ahead.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------- row: revisions


def _row_revisions() -> None:
    from src.data.revisions import fetch_revision_pair, revision_summary

    df = fetch_revision_pair("GDPC1", start="1990-01-01")
    if df is None or df.empty:
        return
    # GDP *levels* are periodically rebased (reference-year changes), so level
    # diffs are dominated by rebasing, not real-time revision. Compare annualized
    # growth rates instead — the meaningful "how far does the first print move".
    first_g = ((df["first"] / df["first"].shift(1)) ** 4 - 1) * 100.0
    latest_g = ((df["latest"] / df["latest"].shift(1)) ** 4 - 1) * 100.0
    summ = revision_summary(first_g.dropna(), latest_g.dropna())
    if not summ or summ["n"] == 0:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'Real-time vs revised · why the first print misleads</div>',
        unsafe_allow_html=True,
    )
    rows = [
        ("Mean absolute revision", f"{summ['mean_abs_revision']:.2f} pp"),
        ("Median revision", f"{summ['median_revision']:+.2f} pp"),
        ("Share revised down", f"{summ['share_revised_down']:.0f}%"),
        ("Quarters", f"{summ['n']}"),
    ]
    body = "".join(
        f'<div class="submodel-row"><span class="name">{lbl}</span>'
        f'<span class="value">{val}</span></div>'
        for lbl, val in rows
    )
    st.markdown(
        f'<div class="panel"><div class="panel-body">{body}'
        f'<div style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
        "Real GDP first-print vs latest-revised <b>annualized growth</b> (not the "
        "rebased level), via ALFRED vintages. See Methodology &middot; Growth &amp; "
        "nowcasting.</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------- interpretation panel


def _interpretation(head: dict, factor: dict, prob: pd.Series | None = None) -> None:
    gdp_now, _ = latest(head.get("A191RL1Q225SBEA"))
    now_now, _ = latest(head.get("GDPNOW"))
    gdi_now, _ = latest(head.get("A261RL1Q225SBEA"))
    composite = factor.get("composite", pd.Series(dtype=float))
    z_now, _ = latest(composite)

    bits = [
        "<p><b>What this tab is.</b> The recession models look <i>forward</i> "
        "(probability of a downturn in the next 12 months). This tab looks at "
        "the <i>present</i>: how fast output is growing right now, before the "
        "official quarterly print lands and before it is revised.</p>",
        "<p><b>GDPNow vs the official print.</b> The Atlanta Fed's GDPNow "
        "mechanically aggregates the same monthly source data the BEA uses, so "
        "it tracks the current quarter in real time. We surface it rather than "
        "rebuild it — the institutions that run dynamic-factor and bridge models "
        "already publish their output to FRED.</p>",
    ]
    if np.isfinite(gdp_now) and np.isfinite(gdi_now):
        gap = gdp_now - gdi_now
        if abs(gap) >= 1.0:
            bits.append(
                f"<p><b>GDP–GDI divergence.</b> GDP ({gdp_now:+.1f}%) and GDI "
                f"({gdi_now:+.1f}%) disagree by {gap:+.1f} pp. GDI often catches "
                "turning points GDP misses, so a wide gap is a flag that the "
                "headline may be revised toward the weaker measure.</p>"
            )
    if np.isfinite(z_now):
        verb = "above its long-run trend" if z_now >= 0.5 else (
            "near trend" if z_now >= -0.5 else "below trend"
        )
        bits.append(
            f"<p><b>Coincident factor.</b> Our z-scored composite of payrolls, "
            f"industrial production, retail sales, and real consumption sits at "
            f"<b>{z_now:+.2f}</b> — {verb}. It is a transparent momentum gauge, "
            "not a GDP forecast; treat it as corroboration for the nowcast.</p>"
        )
    if prob is not None and not prob.empty and not composite.empty:
        frame = factor_prob_frame(composite, prob)
        r = pearson(frame, "factor", "prob")
        if np.isfinite(r):
            bits.append(
                f"<p><b>Link to recession risk.</b> Across history the factor and "
                f"the 12-month recession probability move inversely (r = {r:+.2f}): "
                "weak current momentum has coincided with elevated forward risk. "
                "This tab measures the present; the recession ensemble prices the "
                "next year.</p>"
            )
    bits.append(
        f'<p style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:8px;">'
        "Sources: BEA via FRED (real GDP/GDI, contributions), Atlanta Fed "
        "(GDPNow), Dallas Fed (Weekly Economic Index). Quarterly data is heavily "
        "revised — see the real-time-vs-revised panel above.</p>"
    )
    st.markdown(
        '<div class="panel"><div class="panel-header"><span>How to read this</span></div>'
        f'<div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        + "".join(bits)
        + "</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------- helpers


def _note(text: str) -> None:
    st.markdown(
        f'<div class="panel"><div class="panel-body" '
        f'style="font-size:12px;color:{PALETTE["text_muted"]};">{text}</div></div>',
        unsafe_allow_html=True,
    )


def _unavailable(log: list[str]) -> None:
    st.markdown(
        '<div class="label-small" style="margin-top:8px;">Growth · real-time GDP</div>',
        unsafe_allow_html=True,
    )
    log_html = ""
    if log:
        items = "".join(f"<li>{ln}</li>" for ln in log[:12])
        log_html = (
            f'<details style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
            "<summary>fetch attempts</summary>"
            f'<ul style="margin:6px 0 0 18px;padding:0;">{items}</ul></details>'
        )
    st.markdown(
        f'<div class="panel"><div class="panel-body" style="font-size:12px;color:{PALETTE["text_muted"]};">'
        "Growth data temporarily unavailable. This tab reads live FRED series "
        "(GDPNow, BEA GDP/GDI, the Weekly Economic Index); set "
        "<code>FRED_API_KEY</code> to enable it."
        f"{log_html}</div></div>",
        unsafe_allow_html=True,
    )
