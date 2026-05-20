"""Macro Dashboard — Streamlit entry point."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from src.data.fred_client import fetch_panel
from src.data.nber import load_nber_recessions, recession_in_next_12m
from src.data.series_registry import fred_ids
from src.models.composite import composite_risk
from src.models.lame import LAME
from src.models.recession_ensemble import RecessionEnsemble
from src.models.yield_curve import YieldCurve
from src.ui.theme import PALETTE, inject_theme, risk_color
from src.ui.views import curve, dashboard, methodology, recession
from src.ui.views import lame as lame_view


st.set_page_config(
    page_title="Macro Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAV_OPTIONS = ["Macro Dashboard", "Recession", "Labor", "Yield Curve", "Methodology"]

inject_theme()


# --------------------------------------------------------------------- caching


@st.cache_data(ttl=21600, show_spinner=False)
def _load_panel() -> pd.DataFrame:
    return fetch_panel(fred_ids(), start="1959-01-01")


@st.cache_data(ttl=21600, show_spinner=False)
def _load_nber() -> pd.Series:
    return load_nber_recessions()


@st.cache_resource(show_spinner=False)
def _build_models(cache_version: str, exclude_pandemic: bool = True) -> dict:
    """Build (fit) the three models + walk-forward backtest.

    ``cache_version`` participates in Streamlit's resource cache key — bump
    it whenever model or class code changes, otherwise the *old* instance
    keeps being served (cache_resource hashes function source, not its
    imported dependencies). The argument deliberately has no underscore
    prefix: Streamlit skips underscore-prefixed args when computing the
    cache key, which silently neutralises any value you pass.

    ``exclude_pandemic`` drops 2020-02 through 2021-06 from training. The
    pandemic recession was caused by an exogenous shock and pulls some
    coefficients in directions that don't reflect typical business-cycle
    dynamics; excluding it gives a model that's more representative of
    pre-2020 cyclical behaviour.
    """
    panel = _load_panel()
    nber = _load_nber()
    fwd = recession_in_next_12m(nber)

    exclude = [("2020-02", "2021-06")] if exclude_pandemic else None
    ensemble = RecessionEnsemble(exclude_periods=exclude)
    ensemble.fit(panel, fwd)

    lame = LAME()
    lame.compute(panel)

    yc = YieldCurve(panel)

    # Walk-forward backtest — annual refits from 1985. Cached with the rest.
    oos_history = ensemble.walk_forward_predict(
        panel, fwd, oos_start="1985-01-01", refit_every_months=12,
    )
    oos_stats = ensemble.oos_calibration_stats(oos_history, fwd)

    return {
        "ensemble": ensemble,
        "lame": lame,
        "yield_curve": yc,
        "panel": panel,
        "nber": nber,
        "oos_history": oos_history,
        "oos_stats": oos_stats,
        "exclude_pandemic": exclude_pandemic,
    }


# ------------------------------------------------------------------------- run


def _composite_now(models: dict) -> dict:
    panel = models["panel"]
    ensemble_now = float(models["ensemble"].predict_current()["ensemble"])
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
            '<div class="dashboard-title">Macro Dashboard</div>'
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
        icons=["grid", "graph-down", "people", "activity", "book"],
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
            models = _build_models(
                "v4-no-hwiuratio",
                exclude_pandemic=st.session_state.get("exclude_pandemic", True),
            )
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
        dashboard.render(models["ensemble"], models["lame"], models["panel"], models["nber"])
    elif selected == "Recession":
        recession.render(models["ensemble"], models["nber"], models["panel"])
    elif selected == "Labor":
        lame_view.render(models["panel"], models["nber"], models["lame"])
    elif selected == "Yield Curve":
        curve.render(models["panel"], models["nber"])
    elif selected == "Methodology":
        methodology.render(
            models["ensemble"],
            oos_history=models.get("oos_history"),
            oos_stats=models.get("oos_stats"),
            exclude_pandemic=models.get("exclude_pandemic", True),
        )

    st.markdown(
        f'<div style="margin-top:48px;padding-top:16px;border-top:1px solid {PALETTE["panel_border"]};'
        f'color:{PALETTE["text_tiny"]};font-size:10px;letter-spacing:0.2em;text-transform:uppercase;">'
        "Data · FRED  ·  Recession dates · NBER  ·  This is research, not investment advice."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
