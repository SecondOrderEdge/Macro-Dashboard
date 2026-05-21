"""Macro Dashboard — Streamlit entry point."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from src.data.fred_client import fetch_panel
from src.data.nber import load_recession_flags
from src.data.series_registry import fred_ids
from src.models.composite import composite_risk
from src.models.lame import LAME
from src.models.recession_probit import compute_probit_report
from src.models.yield_curve import YieldCurve
from src.ui.theme import PALETTE, inject_theme, risk_color
from src.ui.views import curve, dashboard, methodology, pulse, recession
from src.ui.views import lame as lame_view


st.set_page_config(
    page_title="U.S. Macro Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAV_OPTIONS = ["Macro Dashboard", "Recession", "Pulse", "Labor", "Yield Curve", "Methodology"]

inject_theme()


# --------------------------------------------------------------------- caching


@st.cache_data(ttl=21600, show_spinner=False)
def _load_panel(cache_version: str) -> pd.DataFrame:
    # cache_version participates in the key so adding a series to the registry
    # (e.g. CFNAIDIFF) forces a refetch instead of serving a stale panel.
    return fetch_panel(fred_ids(), start="1959-01-01")


@st.cache_data(ttl=21600, show_spinner=False)
def _load_nber() -> pd.Series:
    # Prefer FRED USREC (live, auto-updating); falls back to the bundled CSV.
    return load_recession_flags()


@st.cache_resource(show_spinner=False)
def _build_models(cache_version: str) -> dict:
    """Build the LAME, yield-curve, and four-model probit recession engine.

    ``cache_version`` participates in Streamlit's resource cache key — bump
    it whenever model or class code changes, otherwise the *old* instance
    keeps being served (cache_resource hashes function source, not its
    imported dependencies). The argument deliberately has no underscore
    prefix: Streamlit skips underscore-prefixed args when computing the
    cache key, which silently neutralises any value you pass.
    """
    panel = _load_panel(cache_version)
    nber = _load_nber()

    lame = LAME()
    lame.compute(panel)

    yc = YieldCurve(panel)

    # Four-model academic probit ensemble — the app-wide recession engine. It
    # carries its own in-sample + walk-forward calibration. Isolated in a
    # try/except so a probit-side failure can't take down the whole app.
    try:
        probit = compute_probit_report()
    except Exception as exc:  # noqa: BLE001
        probit = {"error": str(exc)}

    return {
        "lame": lame,
        "yield_curve": yc,
        "panel": panel,
        "nber": nber,
        "probit": probit,
    }


# ------------------------------------------------------------------------- run


def _probit_ensemble_now(models: dict) -> float:
    """Headline 12-month recession probability from the four-model probit ensemble."""
    probit = models.get("probit") or {}
    if not probit or "error" in probit:
        return float("nan")
    return float(probit.get("ensemble_probability", float("nan")))


def _recession_view(models: dict) -> tuple[dict, pd.DataFrame]:
    """Adapt the probit report into the ``current`` / ``history`` shape the
    dashboard cards expect (ensemble + per-model 'submodels' + history)."""
    probit = models.get("probit") or {}
    if not probit or "error" in probit:
        return {"ensemble": float("nan"), "submodels": {}, "drivers": {}}, pd.DataFrame()
    current = {
        "ensemble": probit["ensemble_probability"],
        "submodels": probit.get("model_probabilities", {}),
        "drivers": {},
    }
    history = pd.DataFrame({"ensemble": probit["ensemble_history"]})
    return current, history


def _composite_now(models: dict) -> dict:
    panel = models["panel"]
    ensemble_now = _probit_ensemble_now(models)
    lame_hist = models["lame"].history()
    lame_now = float(lame_hist.iloc[-1]) if not lame_hist.empty else float("nan")
    spreads = models["yield_curve"].spreads_history()
    curve_now = (
        float(spreads["spread_10y3m"].dropna().iloc[-1])
        if "spread_10y3m" in spreads.columns and not spreads["spread_10y3m"].dropna().empty
        else float("nan")
    )
    return composite_risk(ensemble_now, lame_now, curve_now)


def _header(models: dict | None) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d · %H:%M UTC")
    composite_html = ""
    if models is not None:
        try:
            comp = _composite_now(models)
            color = risk_color(comp["band"])
            composite_html = (
                f'<div class="composite-readout">'
                f'<div class="label-tiny">Composite Risk</div>'
                f'<div class="composite-number" style="color:{color};">{comp["composite"]}</div>'
                f'<div class="risk-badge" style="color:{color};margin-top:6px;">{comp["band"]}</div>'
                f"</div>"
            )
        except Exception:
            composite_html = ""

    st.markdown(
        (
            '<div class="dashboard-header">'
            '<div>'
            '<div class="dashboard-title">U.S. Macro Dashboard</div>'
            f'<div class="dashboard-subtitle">U.S. recession risk · {timestamp}</div>'
            '</div>'
            f'{composite_html}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _nav() -> str:
    """Top navigation. Drill-down buttons set ``pending_nav`` then rerun; we
    consume it here via ``manual_select`` so streamlit-option-menu actually
    moves its selection (its internal session state otherwise sticks)."""
    if "view" not in st.session_state:
        st.session_state.view = NAV_OPTIONS[0]

    manual_select = None
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        if target in NAV_OPTIONS:
            manual_select = NAV_OPTIONS.index(target)
            st.session_state.view = target

    default_index = NAV_OPTIONS.index(st.session_state.view) if st.session_state.view in NAV_OPTIONS else 0

    selected = option_menu(
        menu_title=None,
        options=NAV_OPTIONS,
        icons=["grid", "graph-down", "reception-4", "people", "activity", "book"],
        orientation="horizontal",
        default_index=default_index,
        manual_select=manual_select,
        key="nav",
        styles={
            "container": {"background-color": "#0a0d12", "padding": "0"},
            "nav-link": {
                "font-size": "11px",
                "letter-spacing": "0.15em",
                "text-transform": "uppercase",
                "color": "#6b7280",
                "background-color": "transparent",
                "padding": "10px 18px",
            },
            "nav-link-selected": {
                "color": "#d4a574",
                "background-color": "transparent",
                "border-bottom": "2px solid #d4a574",
            },
            "icon": {"display": "none"},
        },
    )
    st.session_state.view = selected
    return selected


def main() -> None:
    try:
        with st.spinner("Loading FRED data…"):
            # Bump this version string whenever model code changes — Streamlit's
            # cache_resource doesn't track imported modules, so a code edit to
            # e.g. src/models/lame.py won't otherwise invalidate the cached fit.
            models = _build_models("v15-market-implied")
    except Exception as exc:
        _header(None)
        _nav()
        st.error(
            f"Failed to initialise the dashboard: {exc}. "
            "Make sure FRED_API_KEY is set in `.env` or `.streamlit/secrets.toml`."
        )
        return

    _header(models)
    selected = _nav()

    if selected == "Macro Dashboard":
        rec_current, rec_history = _recession_view(models)
        dashboard.render(rec_current, rec_history, models["lame"], models["panel"], models["nber"])
    elif selected == "Recession":
        recession.render(models.get("probit"), models["nber"])
    elif selected == "Pulse":
        pulse.render(models["panel"], models["nber"], models["lame"])
    elif selected == "Labor":
        lame_view.render(models["panel"], models["nber"], models["lame"])
    elif selected == "Yield Curve":
        curve.render(models["panel"], models["nber"])
    elif selected == "Methodology":
        methodology.render(models.get("probit"))

    st.markdown(
        f'<div style="margin-top:48px;padding-top:16px;border-top:1px solid {PALETTE["panel_border"]};'
        f'color:{PALETTE["text_tiny"]};font-size:10px;letter-spacing:0.2em;text-transform:uppercase;">'
        "Data · FRED  ·  Recession dates · NBER  ·  This is research, not investment advice."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
