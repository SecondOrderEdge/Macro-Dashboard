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
    fetch_gdp_bundle,
    latest,
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


def render(nber: pd.Series) -> None:
    bundle = fetch_gdp_bundle()
    head = bundle.get("headline", {})
    contrib = bundle.get("contributions", {})
    highfreq = bundle.get("highfreq", {})
    factor = coincident_factor()

    nothing = not head and not contrib and not highfreq and factor["composite"].empty
    if nothing:
        _unavailable(bundle.get("log", []) + factor.get("log", []))
        return

    _row_headline(head, nber)
    _row_contributions(contrib)
    _row_highfreq_and_factor(highfreq, factor, nber)
    _row_revisions()
    _interpretation(head, factor)


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
        spark = sparkline_svg(gdp.dropna().tail(40).values, color=color) if gdp is not None else ""
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
            if np.isfinite(now_now) and now_dt is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[now_dt], y=[now_now], mode="markers",
                        marker=dict(color=PALETTE["risk_low"], size=9, symbol="diamond"),
                        name="GDPNow",
                        hovertemplate="GDPNow %{y:+.1f}%<extra></extra>",
                    )
                )
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
            st.plotly_chart(fig, use_container_width=True)
        else:
            _note("Coincident factor unavailable (needs monthly activity series).")


# ------------------------------------------------------------- row: revisions


def _row_revisions() -> None:
    from src.data.revisions import fetch_revision_pair, revision_summary

    df = fetch_revision_pair("GDPC1", start="1990-01-01")
    if df is None or df.empty:
        return
    summ = revision_summary(df["first"], df["latest"])
    if not summ or summ["n"] == 0:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'Real-time vs revised · why the first print misleads</div>',
        unsafe_allow_html=True,
    )
    rows = [
        ("Mean absolute revision", f"{summ['mean_abs_revision']:.1f} bn$"),
        ("Median revision", f"{summ['median_revision']:+.1f} bn$"),
        ("Share revised down", f"{summ['share_revised_down']:.0f}%"),
        ("Observations", f"{summ['n']}"),
    ]
    body = "".join(
        f'<div class="submodel-row"><span class="name">{lbl}</span>'
        f'<span class="value">{val}</span></div>'
        for lbl, val in rows
    )
    st.markdown(
        f'<div class="panel"><div class="panel-body">{body}'
        f'<div style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
        "Real GDP (GDPC1) first release vs latest revision, via ALFRED vintages. "
        "See Methodology &middot; Growth &amp; nowcasting.</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------- interpretation panel


def _interpretation(head: dict, factor: dict) -> None:
    gdp_now, _ = latest(head.get("A191RL1Q225SBEA"))
    now_now, _ = latest(head.get("GDPNOW"))
    gdi_now, _ = latest(head.get("A261RL1Q225SBEA"))
    z_now, _ = latest(factor.get("composite", pd.Series(dtype=float)))

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
