"""Credit & funding-stress view (with a CLO-supply gauge).

A leading-credit-conditions lens: a standardized stress composite (HY/IG
spreads, NFCI, St. Louis stress index, SLOOS bank tightening), funding/liquidity
context, the Fed's quarterly CLO supply estimate, and the tie to forward
recession risk. Honestly scoped as systemic stress — not CLO tranche analytics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.credit import (
    CLO_SERIES,
    LIQUIDITY_SERIES,
    STRESS_SERIES,
    align_corr,
    credit_stress,
    fetch_clo,
    fetch_liquidity,
    latest,
    yoy,
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

_SEV_COLOR = {
    "low": PALETTE["risk_low"],
    "elevated": PALETTE["risk_elevated"],
    "high": PALETTE["risk_high"],
    "critical": PALETTE["risk_critical"],
}


def _stress_band(pct: float) -> tuple[str, str]:
    """Composite percentile → label + severity (higher percentile = more stress)."""
    if not np.isfinite(pct):
        return "—", "elevated"
    if pct >= 90:
        return "STRESSED", "critical"
    if pct >= 70:
        return "ELEVATED", "high"
    if pct >= 40:
        return "MODERATE", "elevated"
    return "CALM", "low"


def _clamp_x(fig, series: pd.Series):
    s = series.dropna()
    if not s.empty:
        fig.update_xaxes(range=[s.index.min(), s.index.max()])
    return fig


def _clip(nber: pd.Series, series: pd.Series) -> pd.Series:
    """Restrict recession shading to the plotted series' span — otherwise the
    vrects stretch the x-axis back to the 1960 start of the NBER series."""
    if nber is None or nber.empty or series is None or series.dropna().empty:
        return nber
    s = series.dropna()
    return nber.loc[(nber.index >= s.index.min()) & (nber.index <= s.index.max())]


def _recession_prob(probit: dict | None) -> pd.Series:
    if not probit or "error" in probit:
        return pd.Series(dtype=float)
    hist = probit.get("ensemble_history")
    if hist is None or len(hist) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(hist).dropna()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def render(nber: pd.Series, probit: dict | None = None) -> None:
    cs = credit_stress()
    composite = cs.get("composite", pd.Series(dtype=float))
    components = cs.get("components", pd.DataFrame())
    liq = fetch_liquidity()
    clo = fetch_clo()
    prob = _recession_prob(probit)

    if composite.empty and not liq and not clo:
        _unavailable(cs.get("log", []))
        return

    _row_headline(composite, components, nber)
    _row_drivers(components)
    _row_liquidity(liq)
    _row_clo(clo)
    _row_vs_risk(composite, prob, nber)
    _interpretation(composite, components, prob)


# --------------------------------------------------------------- row: headline


def _row_headline(composite: pd.Series, components: pd.DataFrame, nber: pd.Series) -> None:
    st.markdown(
        '<div class="label-small" style="margin-top:8px;">'
        'Credit &amp; funding stress · standardized composite</div>',
        unsafe_allow_html=True,
    )
    if composite.empty:
        _note("Credit-stress composite unavailable (needs live FRED spreads/conditions series).")
        return

    z_now, z_dt = latest(composite)
    pct = percentile_rank(composite, z_now)
    label, sev = _stress_band(pct)
    color = _SEV_COLOR[sev]

    left, right = st.columns([1, 3])
    with left:
        spark = sparkline_svg(composite.tail(120).values, color=color)
        st.markdown(
            metric_card(
                label="Credit-stress composite",
                value=f"{z_now:+.2f}",
                unit="σ",
                risk_color_hex=color,
                sparkline_html=spark,
                badge=label,
                subline=f"{pct:.0f}th pct since 1997 · {z_dt.strftime('%b %Y') if z_dt is not None else ''}",
            ),
            unsafe_allow_html=True,
        )
        rows = []
        for name in ("High-yield OAS", "Investment-grade OAS", "Banks tightening C&I"):
            if name in components.columns:
                v, _ = latest(components[name])
                if np.isfinite(v):
                    rows.append((name, f"{v:+.2f} σ"))
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
        fig = line_chart(
            composite.rename("Credit-stress composite (σ)"), color=color,
            nber=_clip(nber, composite), zero_line=True, height=300, yaxis_title="standardized (σ)",
        )
        _clamp_x(fig, composite)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- row: drivers


def _row_drivers(components: pd.DataFrame) -> None:
    if components is None or components.empty:
        return
    vals = {}
    for col in components.columns:
        v, _ = latest(components[col])
        if np.isfinite(v):
            vals[col] = v
    if not vals:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'What is driving stress · component z-scores, latest</div>',
        unsafe_allow_html=True,
    )
    series = pd.Series(vals)
    fig = signed_bar(
        series, pos_color=PALETTE["risk_high"], neg_color=PALETTE["risk_low"],
        height=240, xaxis_title="standardized deviation (σ) · positive = more stress",
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------- row: liquidity


def _row_liquidity(liq: dict) -> None:
    if not liq:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'Funding &amp; liquidity context</div>',
        unsafe_allow_html=True,
    )
    rows = []
    sofr, _ = latest(liq.get("SOFR"))
    if np.isfinite(sofr):
        rows.append(("SOFR (overnight)", f"{sofr:.2f}%"))
    wb_yoy = yoy(liq.get("WALCL"))
    if not wb_yoy.empty:
        rows.append(("Fed balance sheet · YoY", f"{wb_yoy.iloc[-1]:+.1f}%"))
    res = liq.get("WRESBAL")
    if res is not None and not res.dropna().empty:
        rows.append(("Bank reserves · level", f"${res.dropna().iloc[-1] / 1e6:.2f}tn"))  # WRESBAL is $mn
    rrp, _ = latest(liq.get("RRPONTSYD"))
    if np.isfinite(rrp):
        rows.append(("Reverse repo (ON RRP)", f"${rrp:.0f}bn"))
    m2_yoy = yoy(liq.get("M2SL"))
    if not m2_yoy.empty:
        rows.append(("M2 money stock · YoY", f"{m2_yoy.iloc[-1]:+.1f}%"))
    if not rows:
        return
    body = "".join(
        f'<div class="submodel-row"><span class="name">{lbl}</span>'
        f'<span class="value">{val}</span></div>'
        for lbl, val in rows
    )
    st.markdown(
        f'<div class="panel"><div class="panel-body">{body}'
        f'<div style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
        "Liquidity sign is regime-dependent (balance-sheet/M2 contraction tends to tighten "
        "conditions), so these sit beside the composite rather than inside it.</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------- row: CLO


def _row_clo(clo: dict) -> None:
    if not clo:
        return
    liab = clo.get("BOGZ1LM263163063Q")
    if liab is None or liab.dropna().empty:
        return

    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'CLO supply gauge · is the machine expanding or contracting?</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 1])
    with left:
        g = yoy(liab) if liab is not None else pd.Series(dtype=float)
        if not g.empty:
            fig = line_chart(
                g.rename("CLO liabilities · YoY %"), color=PALETTE["accent"],
                zero_line=True, height=240, yaxis_title="YoY %",
            )
            _clamp_x(fig, g)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _note("CLO supply series unavailable.")
    with right:
        rows = []
        lv, lv_dt = latest(liab)
        if np.isfinite(lv):
            rows.append(("CLO liabilities", f"${lv / 1000:,.0f}bn"))
        g = yoy(liab)
        if not g.empty:
            rows.append(("YoY growth", f"{g.iloc[-1]:+.1f}%"))
        body = "".join(
            f'<div class="submodel-row"><span class="name">{lbl}</span>'
            f'<span class="value">{val}</span></div>'
            for lbl, val in rows
        )
        asof = f" · {lv_dt.strftime('%b %Y')}" if lv_dt is not None else ""
        st.markdown(
            f'<div class="label-tiny" style="margin-bottom:6px;">Z.1 Financial Accounts{asof}</div>'
            f'<div class="panel"><div class="panel-body">{body}'
            f'<div style="margin-top:8px;color:{PALETTE["text_tiny"]};font-size:11px;">'
            "Quarterly, ~10-week lag; most US CLOs are offshore-domiciled. A coarse "
            "supply trend, not timely demand or default data.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- row: vs risk


def _row_vs_risk(composite: pd.Series, prob: pd.Series, nber: pd.Series) -> None:
    if composite.empty or prob is None or prob.empty:
        return
    frame, r = align_corr(composite, prob)
    if len(frame) < 8:
        return
    st.markdown(
        '<div class="label-small" style="margin-top:16px;">'
        'Credit stress vs forward recession risk</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=composite.index, y=composite.values, mode="lines",
            line=dict(color=PALETTE["accent"], width=1.4), name="Credit stress (σ)",
            hovertemplate="%{x|%b %Y}<br>stress %{y:+.2f}σ<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=prob.index, y=prob.values, mode="lines",
            line=dict(color=PALETTE["risk_critical"], width=1.4),
            name="Recession prob (12m, %)", yaxis="y2",
            hovertemplate="%{x|%b %Y}<br>prob %{y:.0f}%<extra></extra>",
        ))
        add_recession_shading(fig, _clip(nber, composite))
        apply_template(fig, height=320)
        fig.update_yaxes(title="credit stress (σ)")
        fig.update_layout(yaxis2=dict(
            title="recession prob (%)", overlaying="y", side="right",
            range=[0, 100], gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=PALETTE["risk_critical"]),
        ))
        _clamp_x(fig, composite)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        z_now, _ = latest(composite)
        p_now, _ = latest(prob)
        rows = [
            ("Stress today", f"{z_now:+.2f} σ"),
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
            "A positive r is expected: credit stress and forward recession risk rise together. "
            "Credit conditions typically lead the real economy.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------- interpretation panel


def _interpretation(composite: pd.Series, components: pd.DataFrame, prob: pd.Series | None) -> None:
    z_now, _ = latest(composite)
    bits = [
        "<p><b>What this tab is.</b> A leading <i>credit-conditions</i> lens. Credit tightens "
        "before the real economy turns, so a standardized stress composite — high-yield and "
        "investment-grade spreads, the Chicago Fed conditions index, the St. Louis stress index, "
        "and the SLOOS bank-tightening survey — is an upstream complement to the recession "
        "ensemble.</p>",
        "<p><b>Honest scope.</b> This is <i>systemic</i> credit &amp; funding stress, not "
        "CLO tranche analytics. The CLO panel is the Fed's quarterly Z.1 <i>supply</i> estimate "
        "(is issuance expanding or contracting) — coarse and lagged. Timely CLO demand, AAA "
        "spreads, and default/distress rates live behind paid vendors (JPM, Barclays, LCD, "
        "Moody's, S&amp;P) and are deliberately out of scope.</p>",
        "<p><b>Construction note.</b> The inputs overlap by design — NFCI already embeds credit "
        "spreads — so the composite is a robust summary of one underlying stress factor, not a "
        "set of independent signals. Each input is z-scored over the common 1997+ sample and "
        "averaged, oriented so positive = more stress.</p>",
    ]
    if np.isfinite(z_now):
        state = ("elevated" if z_now >= 1 else "above average" if z_now >= 0.25
                 else "benign" if z_now <= -0.5 else "around normal")
        bits.insert(1, f"<p>Composite stress is currently <b>{z_now:+.2f}σ</b> — {state} "
                       "relative to its post-1997 history.</p>")
    st.markdown(
        '<div class="panel"><div class="panel-header"><span>How to read this</span></div>'
        f'<div class="panel-body" style="font-size:13px;line-height:1.7;color:{PALETTE["text_primary"]};">'
        + "".join(bits)
        + f'<p style="color:{PALETTE["text_muted"]};font-size:11px;margin-top:8px;">'
        "Sources: ICE BofA OAS, Chicago Fed (NFCI), St. Louis Fed (STLFSI), Fed SLOOS, and the "
        "Z.1 Financial Accounts (CLOs) — all via FRED.</p>"
        "</div></div>",
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
        '<div class="label-small" style="margin-top:8px;">Credit &amp; funding stress</div>',
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
        "Credit data temporarily unavailable. This tab reads live FRED series (credit spreads, "
        "NFCI, SLOOS, liquidity, Z.1 CLOs); set <code>FRED_API_KEY</code> to enable it."
        f"{log_html}</div></div>",
        unsafe_allow_html=True,
    )
