"""Credit-stress data layer + view: composite math, graceful fetch, render."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import credit
from src.data.credit import (
    STRESS_SERIES,
    align_corr,
    credit_stress,
    fetch_household,
    fetch_liquidity,
    latest,
    yoy,
)


def _clear(fn):
    if hasattr(fn, "clear"):
        fn.clear()


def _daily(n=2000, base=4.0, seed=0):
    idx = pd.date_range("2005-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.Series(base + np.cumsum(rng.normal(0, 0.02, n)), index=idx)


def test_latest_and_yoy():
    s = pd.Series([100.0, 110.0], index=pd.date_range("2020-01-01", periods=2, freq="YS"))
    assert latest(s)[0] == 110.0
    m = pd.Series(np.arange(1, 25, dtype=float), index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    g = yoy(m)
    assert not g.empty and g.iloc[-1] > 0  # rising series → positive YoY


def test_credit_stress_composite(monkeypatch):
    def fake(sid, start="1997-01-01"):
        return _daily(seed=hash(sid) % 1000)

    monkeypatch.setattr("src.data.fred_client.fetch_series", fake)
    _clear(credit_stress)
    out = credit_stress()
    comp = out["composite"]
    assert not comp.empty
    assert abs(float(comp.mean())) < 0.5  # z-scored components average near zero
    assert out["components"].shape[1] == len(STRESS_SERIES)


def test_credit_stress_empty_without_key(monkeypatch):
    monkeypatch.setattr("src.data.fred_client.fetch_series", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    _clear(credit_stress)
    out = credit_stress()
    assert out["composite"].empty
    assert len(out["log"]) >= 1


def test_fetch_liquidity_graceful(monkeypatch):
    monkeypatch.setattr("src.data.fred_client.fetch_series", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    _clear(fetch_liquidity)
    assert fetch_liquidity() == {}


def test_align_corr_sign():
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    a = pd.Series(np.sin(np.arange(120) / 5.0), index=idx)
    b = pd.Series(50 + 40 * np.sin(np.arange(120) / 5.0), index=idx)  # in phase → positive
    frame, r = align_corr(a, b)
    assert list(frame.columns) == ["a", "b"]
    assert r > 0.9


def test_stress_band_thresholds():
    from src.ui.views.credit import _stress_band

    assert _stress_band(95)[1] == "critical"
    assert _stress_band(80)[0] == "ELEVATED"
    assert _stress_band(10)[0] == "CALM"


def test_credit_render_smoke_empty(monkeypatch):
    from src.ui.views import credit as view

    monkeypatch.setattr(view, "credit_stress", lambda: {"composite": pd.Series(dtype=float), "components": pd.DataFrame(), "log": ["x"]})
    monkeypatch.setattr(view, "fetch_liquidity", lambda: {})
    monkeypatch.setattr(view, "fetch_clo", lambda: {})
    view.render(pd.Series(dtype=bool))  # unavailable path, must not raise


def test_credit_render_smoke_populated(monkeypatch):
    from src.ui.views import credit as view

    months = pd.date_range("2005-01-01", periods=200, freq="MS")
    composite = pd.Series(np.sin(np.arange(200) / 6.0), index=months)
    components = pd.DataFrame(
        {name: np.sin(np.arange(200) / 6.0 + i) for i, (_, name, _) in enumerate(STRESS_SERIES)},
        index=months,
    )
    liq = {
        "SOFR": pd.Series([5.3], index=[pd.Timestamp("2024-01-01")]),
        "WALCL": pd.Series(np.linspace(8e6, 7e6, 60), index=pd.date_range("2019-01-01", periods=60, freq="MS")),
        "M2SL": pd.Series(np.linspace(15000, 21000, 60), index=pd.date_range("2019-01-01", periods=60, freq="MS")),
        "RRPONTSYD": pd.Series(
            np.concatenate([np.linspace(0, 2500, 30), np.linspace(2500, 3, 30)]),
            index=pd.date_range("2019-01-01", periods=60, freq="MS"),
        ),  # buildup-then-drain → exercises the peak-annotation path
    }
    clo = {"BOGZ1LM263163063Q": pd.Series(np.linspace(5e5, 1e6, 40), index=pd.date_range("2014-01-01", periods=40, freq="QS"))}
    probit = {"ensemble_history": pd.Series(40 + 20 * np.sin(np.arange(200) / 6.0), index=months)}

    monkeypatch.setattr(view, "credit_stress", lambda: {"composite": composite, "components": components, "log": []})
    monkeypatch.setattr(view, "fetch_liquidity", lambda: liq)
    monkeypatch.setattr(view, "fetch_clo", lambda: clo)

    nber = pd.Series(False, index=pd.date_range("2005-01-01", periods=220, freq="MS"))
    view.render(nber, probit)  # full path, must not raise


def test_fetch_household_graceful(monkeypatch):
    monkeypatch.setattr("src.data.fred_client.fetch_series", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    _clear(fetch_household)
    assert fetch_household() == {}


def test_household_row_renders():
    from src.ui.views import credit as view

    w = pd.date_range("2018-01-01", periods=300, freq="W")
    m = pd.date_range("2018-01-01", periods=80, freq="MS")
    hh = {
        "MORTGAGE30US": pd.Series(np.linspace(4.0, 7.0, 300), index=w),
        "TERMCBCCALLNS": pd.Series(np.linspace(15.0, 22.0, 80), index=m),
        "TERMCBAUTO48NS": pd.Series(np.linspace(5.0, 8.0, 80), index=m),
        "GS10": pd.Series(np.linspace(2.5, 4.3, 80), index=m),
        "FEDFUNDS": pd.Series(np.linspace(0.1, 5.3, 80), index=m),
    }
    view._row_household(hh)   # populated path, must not raise
    view._row_household({})   # empty path, early return, must not raise
