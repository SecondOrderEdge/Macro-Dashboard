"""Visual theme: palette, fonts, and the global CSS we inject into Streamlit."""

from __future__ import annotations

PALETTE: dict = {
    "bg":            "#0a0d12",
    "panel_bg":      "#11151c",
    "panel_border":  "#1f2630",
    "text_primary":  "#d4d4d0",
    "text_muted":    "#6b7280",
    "text_tiny":     "#5a6470",
    "accent":        "#d4a574",
    "risk_low":      "#5ba3a3",
    "risk_elevated": "#e8b339",
    "risk_high":     "#c97c5d",
    "risk_critical": "#b54848",
    "submodel": {
        "yield_curve": "#e8b339",
        "labor":       "#5ba3a3",
        "credit":      "#c97c5d",
        "housing":     "#7a8b99",
        "sentiment":   "#9d7aa8",
    },
}


def risk_color(band: str) -> str:
    return {
        "LOW": PALETTE["risk_low"],
        "ELEVATED": PALETTE["risk_elevated"],
        "HIGH": PALETTE["risk_high"],
        "CRITICAL": PALETTE["risk_critical"],
    }.get(band.upper(), PALETTE["text_muted"])


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&display=swap');

html, body, [class*="css"]  {
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    background: #0a0d12;
    color: #d4d4d0;
    font-variant-numeric: tabular-nums;
}

.display-font { font-family: 'Fraunces', serif; font-weight: 400; }
.data-font { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
.label-tiny {
    font-size: 9px;
    letter-spacing: 0.25em;
    color: #5a6470;
    text-transform: uppercase;
}
.label-small {
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #6b7280;
    text-transform: uppercase;
}
.panel {
    background: #11151c;
    border: 1px solid #1f2630;
    border-radius: 2px;
    padding: 0;
    margin-bottom: 16px;
}
.panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid #1f2630;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #6b7280;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.panel-body { padding: 16px; }
.metric-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 48px;
    font-weight: 300;
    line-height: 1;
    letter-spacing: -0.04em;
}
.metric-unit {
    font-size: 14px;
    color: #6b7280;
    margin-left: 6px;
}
.metric-sub {
    color: #6b7280;
    font-size: 11px;
    margin-top: 8px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.risk-badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border: 1px solid currentColor;
    border-radius: 1px;
}
.submodel-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-top: 1px solid #1f2630;
    font-size: 11px;
}
.submodel-row:first-child { border-top: none; }
.submodel-row .name { color: #6b7280; letter-spacing: 0.1em; text-transform: uppercase; }
.submodel-row .value { color: #d4d4d0; font-feature-settings: 'tnum'; }

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 16px 0 24px 0;
    border-bottom: 1px solid #1f2630;
    margin-bottom: 16px;
}
.dashboard-title {
    font-family: 'Fraunces', serif;
    font-weight: 400;
    font-size: 24px;
    letter-spacing: 0.02em;
    color: #d4d4d0;
}
.dashboard-subtitle {
    color: #6b7280;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 4px;
}
.composite-readout {
    text-align: right;
}
.composite-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 36px;
    font-weight: 300;
    line-height: 1;
}

a.drill-link {
    color: #d4a574;
    text-decoration: none;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #0a0d12; }
.block-container { padding-top: 1rem; max-width: 1600px; }

/* Buttons */
div.stButton > button {
    background: transparent;
    color: #d4a574;
    border: 1px solid #1f2630;
    border-radius: 1px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 6px 14px;
}
div.stButton > button:hover {
    border-color: #d4a574;
    background: rgba(212, 165, 116, 0.04);
}

/* Tables */
.dataframe { font-family: 'JetBrains Mono', monospace !important; }
</style>
"""


def inject_theme() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
