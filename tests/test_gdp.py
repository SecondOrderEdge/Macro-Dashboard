"""Growth/GDP data layer: graceful fetch, coincident factor, view render."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import gdp
from src.data.gdp import (
    CONTRIBUTION_SERIES,
    HEADLINE_SERIES,
    coincident_factor,
    factor_gdp_frame,
    factor_prob_frame,
    fetch_gdp_bundle,
    latest,
    pearson,
)


def _monthly(n=240, base=100.0, slope=0.3, seed=0):
    idx = pd.date_range("2005-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    vals = base + slope * np.arange(n) + rng.normal(0, 1.0, n)
    return pd.Series(vals, index=idx)


def _clear(fn):
    # Streamlit-wrapped cache_data fns cache by args; clear so each test's
    # monkeypatched fetch_series is actually exercised (no cross-test leakage).
    if hasattr(fn, "clear"):
        fn.clear()


def test_latest_handles_empty_and_nan():
    assert latest(None) == (float("nan"), None) or np.isnan(latest(None)[0])
    v, d = latest(pd.Series([1.0, 2.0, np.nan], index=pd.date_range("2020-01-01", periods=3, freq="MS")))
    assert v == 2.0 and d == pd.Timestamp("2020-02-01")


def test_fetch_gdp_bundle_graceful_without_key(monkeypatch):
    # No FRED key / network: every fetch should be caught and skipped, never raise.
    def boom(*_a, **_k):
        raise RuntimeError("FRED_API_KEY not set.")

    monkeypatch.setattr("src.data.fred_client.fetch_series", boom)
    _clear(fetch_gdp_bundle)
    bundle = fetch_gdp_bundle()
    assert set(bundle) == {"headline", "contributions", "highfreq", "log"}
    assert bundle["headline"] == {} and bundle["contributions"] == {}
    assert len(bundle["log"]) >= 1


def test_coincident_factor_math(monkeypatch):
    def fake(sid, start="1960-01-01"):
        return _monthly(seed=hash(sid) % 1000)

    monkeypatch.setattr("src.data.fred_client.fetch_series", fake)
    _clear(coincident_factor)
    out = coincident_factor()
    comp = out["composite"]
    assert not comp.empty
    # Each input is z-scored before averaging → composite roughly centered.
    assert abs(float(comp.mean())) < 0.5
    assert out["components"].shape[1] == 4


def test_coincident_factor_empty_without_key(monkeypatch):
    monkeypatch.setattr("src.data.fred_client.fetch_series", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    _clear(coincident_factor)
    out = coincident_factor()
    assert out["composite"].empty


def test_factor_band_thresholds():
    from src.ui.views.growth import _factor_band

    assert _factor_band(1.0)[0] == "ABOVE TREND"
    assert _factor_band(0.0)[1] == "elevated"
    assert _factor_band(-1.0)[0] == "BELOW TREND"
    assert _factor_band(-2.0)[1] == "critical"


def _quarterly(vals):
    idx = pd.date_range("2018-03-01", periods=len(vals), freq="QS")
    return pd.Series(vals, index=idx)


def test_growth_render_smoke_populated(monkeypatch):
    from src.ui.views import growth

    head = {
        "A191RL1Q225SBEA": _quarterly([2.1, 3.0, -1.2, 2.5, 1.8, 2.9, 3.1, 2.2]),
        "GDPNOW": _quarterly([2.0, 2.6, 1.1, 2.3, 1.9, 2.7, 2.8, 2.4]),
        "A261RL1Q225SBEA": _quarterly([1.5, 2.2, -2.0, 1.9, 1.4, 2.1, 2.5, 1.0]),
    }
    contrib = {sid: _quarterly([0.5, 0.3, -0.2, 0.4, 0.1, 0.6, 0.7, 0.2]) for sid, _ in CONTRIBUTION_SERIES}
    highfreq = {"WEI": _monthly(n=60, base=2.0, slope=0.0, seed=7)}
    factor = {"composite": _monthly(n=120, base=0.0, slope=0.0, seed=9), "components": pd.DataFrame(), "log": []}

    monkeypatch.setattr(growth, "fetch_gdp_bundle", lambda: {"headline": head, "contributions": contrib, "highfreq": highfreq, "log": []})
    monkeypatch.setattr(growth, "coincident_factor", lambda: factor)

    nber = pd.Series(False, index=pd.date_range("2018-01-01", periods=80, freq="MS"))
    growth.render(nber)  # must not raise


def test_growth_render_smoke_empty(monkeypatch):
    from src.ui.views import growth

    empty_factor = {"composite": pd.Series(dtype=float), "components": pd.DataFrame(), "log": ["x"]}
    monkeypatch.setattr(growth, "fetch_gdp_bundle", lambda: {"headline": {}, "contributions": {}, "highfreq": {}, "log": ["x"]})
    monkeypatch.setattr(growth, "coincident_factor", lambda: empty_factor)

    growth.render(pd.Series(dtype=bool))  # must not raise (unavailable path)


def test_headline_ids_present():
    ids = [sid for sid, _ in HEADLINE_SERIES]
    assert "GDPNOW" in ids and "A191RL1Q225SBEA" in ids


def test_factor_gdp_frame_aligns_and_correlates():
    months = pd.date_range("2010-01-01", periods=120, freq="MS")
    composite = pd.Series(np.linspace(-2, 2, 120), index=months)
    # GDP roughly tracks the factor, sampled quarterly.
    q = pd.date_range("2010-01-01", periods=40, freq="QS")
    gdp = pd.Series(np.linspace(-1, 3, 40) + 1.0, index=q)
    frame = factor_gdp_frame(composite, gdp)
    assert list(frame.columns) == ["factor", "gdp"] and len(frame) >= 30
    assert pearson(frame, "factor", "gdp") > 0.9  # both monotonic up


def test_factor_gdp_frame_empty_inputs():
    assert factor_gdp_frame(pd.Series(dtype=float), pd.Series(dtype=float)).empty


def test_factor_prob_frame_and_pearson_sign():
    months = pd.date_range("2000-01-01", periods=200, freq="MS")
    composite = pd.Series(np.sin(np.arange(200) / 6.0), index=months)
    prob = pd.Series(50 - 40 * np.sin(np.arange(200) / 6.0), index=months)  # inverse, 0-100
    frame = factor_prob_frame(composite, prob)
    assert list(frame.columns) == ["factor", "prob"]
    assert pearson(frame, "factor", "prob") < -0.9


def test_pearson_too_few_points():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
    assert np.isnan(pearson(df, "a", "b"))


def test_robust_range_ignores_outliers():
    from src.ui.views.growth import _robust_range

    vals = list(np.random.default_rng(0).normal(2, 1, 200)) + [-30.0, 35.0]  # COVID-like
    lo, hi = _robust_range(vals)
    assert lo > -10 and hi < 10  # outliers excluded from the view range


def test_recession_prob_history_extraction():
    from src.ui.views.growth import _recession_prob_history

    assert _recession_prob_history(None).empty
    assert _recession_prob_history({"error": "x"}).empty
    s = pd.Series([10.0, 20.0], index=pd.date_range("2020-01-01", periods=2, freq="MS"))
    out = _recession_prob_history({"ensemble_history": s})
    assert len(out) == 2 and out.iloc[-1] == 20.0


def test_growth_render_smoke_with_probit(monkeypatch):
    from src.ui.views import growth

    head = {
        "A191RL1Q225SBEA": pd.Series(np.linspace(1, 3, 40), index=pd.date_range("2014-01-01", periods=40, freq="QS")),
        "GDPNOW": pd.Series([2.4], index=[pd.Timestamp("2024-01-01")]),
    }
    contrib = {sid: pd.Series(np.linspace(0, 0.6, 40), index=pd.date_range("2014-01-01", periods=40, freq="QS")) for sid, _ in CONTRIBUTION_SERIES}
    factor = {"composite": pd.Series(np.sin(np.arange(150) / 5.0), index=pd.date_range("2012-01-01", periods=150, freq="MS")), "components": pd.DataFrame(), "log": []}
    probit = {"ensemble_history": pd.Series(40 - 20 * np.sin(np.arange(150) / 5.0), index=pd.date_range("2012-01-01", periods=150, freq="MS"))}

    monkeypatch.setattr(growth, "fetch_gdp_bundle", lambda: {"headline": head, "contributions": contrib, "highfreq": {}, "log": []})
    monkeypatch.setattr(growth, "coincident_factor", lambda: factor)

    nber = pd.Series(False, index=pd.date_range("2012-01-01", periods=170, freq="MS"))
    growth.render(nber, probit)  # exercises factor-validation + growth-vs-risk rows
