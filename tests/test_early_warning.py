"""Early-warning ladder: rung computation, lit logic, summary staging, render."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.early_warning import build_ladder, ladder_summary


def _panel(**overrides) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=40, freq="MS")
    base = {
        "T10Y3M": np.full(40, 1.5),       # steep (not inverted)
        "DRTSCILM": np.full(40, -5.0),    # banks easing
        "PERMIT": np.linspace(1400, 1500, 40),  # rising → YoY > 0
        "SAHMREALTIME": np.full(40, 0.1),  # calm
        "NFCI": np.full(40, -0.4),        # easy
        "VIXCLS": np.full(40, 15.0),      # calm
    }
    base.update(overrides)
    return pd.DataFrame(base, index=idx)


def test_calm_panel_all_clear():
    rungs = build_ladder(_panel(), probit={"ensemble_probability": 8.0}, lame=None)
    assert rungs, "expected rungs from a full panel"
    assert all(not r["lit"] for r in rungs)
    s = ladder_summary(rungs)
    assert s["stage"] == "ALL CLEAR" and s["n_lit"] == 0


def test_trouble_panel_lights_rungs_and_stages_acute():
    panel = _panel(
        T10Y3M=np.full(40, -0.5),                 # inverted (critical)
        DRTSCILM=np.full(40, 30.0),               # tightening (high)
        PERMIT=np.linspace(1600, 1200, 40),       # falling → YoY < 0
        SAHMREALTIME=np.full(40, 0.6),            # triggered
        NFCI=np.full(40, 0.4),                    # tightening
        VIXCLS=np.full(40, 16.0),                 # still calm (bottom rung unlit)
    )
    rungs = build_ladder(panel, probit={"ensemble_probability": 45.0}, lame=None)
    lit = {r["key"]: r["lit"] for r in rungs}
    assert lit["curve"] and lit["sloos"] and lit["housing"] and lit["sahm"]
    assert lit["ensemble"] and lit["nfci"]
    assert lit["vix"] is False
    # deepest lit (NFCI) is far down the ladder → ACUTE
    assert ladder_summary(rungs)["severity"] == "critical"
    # curve inversion at -0.5 is the most severe band
    curve = next(r for r in rungs if r["key"] == "curve")
    assert curve["severity"] == "critical"


def test_breadth_rung_from_lame():
    class _Lame:
        def current_breakdown(self):
            return pd.DataFrame({"z_score": [-1.0, -0.5, -0.2, 0.3]})  # 75% below trend

    rungs = build_ladder(_panel(), probit=None, lame=_Lame())
    breadth = next((r for r in rungs if r["key"] == "breadth"), None)
    assert breadth is not None and breadth["lit"] and breadth["severity"] == "critical"


def test_ensemble_rung_skipped_without_probit():
    rungs = build_ladder(_panel(), probit=None, lame=None)
    assert all(r["key"] != "ensemble" for r in rungs)


def test_empty_panel_no_rungs():
    rungs = build_ladder(pd.DataFrame(), probit=None, lame=None)
    assert rungs == []
    assert ladder_summary(rungs)["stage"] == "—"


def test_render_smoke():
    from src.ui.views import early_warning

    early_warning.render(_panel(), {"ensemble_probability": 45.0}, None)  # must not raise
    early_warning.render(pd.DataFrame(), None, None)  # unavailable path
