"""Start-dated recession target + "in recession now" nowcast panel wiring.

Target: y_t = 1 if an NBER peak P falls in t+1..t+12; months P..T (peak month
through trough) are excluded (NaN); months whose 12-month window runs past the
last USREC observation are unlabelled (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import recession_probit as rp
from src.ui import nowcast as nc

MS = pd.DateOffset(months=1)


def _usrec(start: str, end: str, recessions: list[tuple[str, str]]) -> pd.Series:
    """USREC = 1 on P+1..T for each (peak, trough) pair (NBER convention)."""
    idx = pd.date_range(start, end, freq="MS")
    u = pd.Series(0.0, index=idx)
    for peak, trough in recessions:
        u[(idx > pd.Timestamp(peak)) & (idx <= pd.Timestamp(trough))] = 1.0
    return u


# ------------------------------------------------------------------ target


def test_turning_points_from_usrec():
    u = _usrec("2000-01-01", "2012-12-01", [("2001-03-01", "2001-11-01"), ("2007-12-01", "2009-06-01")])
    assert rp.nber_turning_points(u) == [
        (pd.Timestamp("2001-03-01"), pd.Timestamp("2001-11-01")),
        (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01")),
    ]


def test_peak_window_is_the_twelve_months_before_the_peak():
    u = _usrec("2000-01-01", "2012-12-01", [("2007-12-01", "2009-06-01")])
    y = rp.start_target(u)
    peak = pd.Timestamp("2007-12-01")
    for k in range(1, 13):  # t = P-1 .. P-12 → the peak lies in t+1..t+12
        assert y.loc[peak - pd.DateOffset(months=k)] == 1.0
    assert y.loc[peak - pd.DateOffset(months=13)] == 0.0
    assert (y.loc[: peak - pd.DateOffset(months=13)] == 0.0).all()
    assert y.loc[peak - pd.DateOffset(months=12): peak - MS].sum() == 12


def test_in_recession_months_and_peak_month_are_excluded():
    u = _usrec("2000-01-01", "2012-12-01", [("2007-12-01", "2009-06-01")])
    y = rp.start_target(u)
    excluded = y.loc["2007-12-01":"2009-06-01"]
    assert excluded.isna().all()          # peak month + every USREC=1 month
    assert len(excluded) == 19
    assert y.loc["2009-07-01"] == 0.0     # first month after the trough is labelled again
    assert y.loc["2007-11-01"] == 1.0     # month before the peak is a positive


def test_back_to_back_recessions_keep_exclusion_over_the_next_peak_window():
    # 1980 recession (peak 1980-01, trough 1980-07) sits inside the 12 months
    # before the 1981-07 peak: its months stay excluded, not relabelled 1.
    u = _usrec("1975-06-01", "1990-12-01", [("1980-01-01", "1980-07-01"), ("1981-07-01", "1982-11-01")])
    y = rp.start_target(u)
    assert y.loc["1980-01-01":"1980-07-01"].isna().all()
    assert (y.loc["1980-08-01":"1981-06-01"] == 1.0).all()
    assert y.loc["1981-07-01":"1982-11-01"].isna().all()
    # 1979-01..1979-12 precede the 1980-01 peak.
    assert (y.loc["1979-01-01":"1979-12-01"] == 1.0).all()
    assert y.loc["1978-12-01"] == 0.0


def test_sample_end_leaves_last_twelve_months_unlabelled():
    u = _usrec("2010-01-01", "2020-12-01", [])
    y = rp.start_target(u)
    last = u.index[-1]
    assert y.loc[last - pd.DateOffset(months=12)] == 0.0  # window fully observed
    assert y.loc[last - pd.DateOffset(months=11):].isna().all()
    assert y.dropna().index.max() == last - pd.DateOffset(months=12)


def test_known_peak_near_sample_end_still_unlabelled_until_window_observed():
    # Peak dated 2020-02 but data end 2020-06: months 2019-07..2020-01 are
    # positives in principle, but their windows aren't fully observed, so the
    # single 12-month label lag applies and they stay NaN.
    u = _usrec("2010-01-01", "2020-06-01", [("2020-02-01", "2020-04-01")])
    y = rp.start_target(u)
    assert y.loc["2019-06-01"] == 1.0
    assert y.loc["2019-07-01":].isna().all()


def test_recession_ongoing_at_sample_end_is_excluded():
    u = _usrec("2000-01-01", "2012-06-01", [("2011-12-01", "2012-06-01")])  # still open
    pts = rp.nber_turning_points(u)
    assert pts[-1] == (pd.Timestamp("2011-12-01"), pd.Timestamp("2012-06-01"))
    y = rp.start_target(u)
    assert y.loc["2011-12-01":].isna().all()
    assert y.loc["2010-12-01"] == 1.0  # peak within t+1..t+12 and window observed


def test_sample_starting_inside_a_recession_has_no_peak_and_is_excluded():
    idx = pd.date_range("1970-01-01", "1980-12-01", freq="MS")
    u = pd.Series(0.0, index=idx)
    u[(idx >= "1970-01-01") & (idx <= "1970-11-01")] = 1.0
    pts = rp.nber_turning_points(u)
    assert pts[0] == (None, pd.Timestamp("1970-11-01"))
    y = rp.start_target(u)
    assert y.loc["1970-01-01":"1970-11-01"].isna().all()
    assert y.loc["1970-12-01"] == 0.0


def test_target_matches_published_nber_chronology():
    # NBER table (nber.org, last updated 2023-03-14), USREC = 1 on P+1..T.
    peaks = ["1969-12", "1973-11", "1980-01", "1981-07", "1990-07", "2001-03", "2007-12", "2020-02"]
    troughs = ["1970-11", "1975-03", "1980-07", "1982-11", "1991-03", "2001-11", "2009-06", "2020-04"]
    u = _usrec("1967-03-01", "2026-08-01", [(p + "-01", t + "-01") for p, t in zip(peaks, troughs)])
    y = rp.start_target(u)
    assert [p.strftime("%Y-%m") for p, _ in rp.nber_turning_points(u)] == peaks
    assert y.index.max() == pd.Timestamp("2026-08-01")
    assert y.dropna().index.max() == pd.Timestamp("2025-08-01")
    assert int(y.loc["1985-01-01":"2025-08-01"].notna().sum()) == 448
    assert int(y.loc["1985-01-01":"2025-08-01"].sum()) == 48


def test_engineer_features_uses_start_target_by_default():
    assert rp.TARGET_DEFINITION == "start"
    u = _usrec("1990-01-01", "2012-12-01", [("2001-03-01", "2001-11-01"), ("2007-12-01", "2009-06-01")])
    raw = pd.DataFrame({"USREC": u, "GS10": 4.0, "TB3MS": 2.0})
    data, _, _ = rp.engineer_features(raw)
    pd.testing.assert_series_equal(data["TARGET"], rp.start_target(u), check_names=False)
    assert rp.target_series(raw).isna().sum() == 0
    assert "2008-06-01" not in rp.target_series(raw).index.strftime("%Y-%m-%d")


def test_expanding_base_rate_ignores_excluded_months():
    u = _usrec("2000-01-01", "2012-12-01", [("2007-12-01", "2009-06-01")])
    y = rp.start_target(u)
    t = pd.Timestamp("2011-01-01")
    br = rp.expanding_base_rate(y, pd.DatetimeIndex([t]))
    known = y.loc[: t - pd.DateOffset(months=12)].dropna()
    assert br.iloc[0] == pytest.approx(known.mean())


def test_calibration_drops_in_recession_months():
    u = _usrec("2000-01-01", "2012-12-01", [("2007-12-01", "2009-06-01")])
    y = rp.start_target(u)
    pred = pd.Series(10.0, index=u.index)
    st = rp.calibration_stats(pred, y)
    assert st["n_obs"] == int(y.notna().sum())


# ---------------------------------------------------------- nowcast panel


def test_recession_state_rules():
    assert rp.recession_state(1.0, 0.0, 0.0) == "nber_recession"
    assert rp.recession_state(0.0, 50.0, 0.0) == "nowcast_flag"   # CP threshold inclusive
    assert rp.recession_state(0.0, 49.9, 0.49) == "expansion"
    assert rp.recession_state(0.0, 1.0, 0.50) == "nowcast_flag"   # Sahm threshold inclusive
    assert rp.recession_state(None, None, None) == "expansion"
    assert rp.CP_SIGNAL_THRESHOLD == 50.0 and rp.SAHM_SIGNAL_THRESHOLD == 0.50


def test_nowcast_panel_reads_latest_unshifted_values():
    idx = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    raw = pd.DataFrame({
        "USREC": 0.0,
        "RECPROUSM156N": np.linspace(1, 12, 12),
        "SAHMREALTIME": np.r_[np.linspace(0, 0.6, 11), np.nan],
    }, index=idx)
    panel = rp.nowcast_panel(raw)
    cp, sahm = panel["indicators"]["Chauvet-Piger"], panel["indicators"]["Sahm rule"]
    assert cp["value"] == pytest.approx(12.0) and cp["as_of"] == "2020-12" and cp["signal"] is False
    assert sahm["value"] == pytest.approx(0.6) and sahm["as_of"] == "2020-11" and sahm["signal"] is True
    assert panel["state"] == "nowcast_flag" and panel["headline_applicable"] is True


@pytest.fixture(scope="module")
def raw_with_nowcast():
    rng = np.random.default_rng(3)
    idx = pd.date_range("1967-01-01", "2025-12-01", freq="MS")
    n = len(idx)
    spread = pd.Series(1.5 + np.cumsum(rng.normal(0, 0.06, n)), index=idx).clip(-2.5, 4.0)
    tb3ms = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.04, n)), index=idx).clip(0.05, None)
    # Short recessions a year after the spread's low tail, so the start target
    # has enough positives and training rows.
    usrec = pd.Series(0.0, index=idx)
    peaks = pd.date_range("1972-06-01", "2020-06-01", freq="72MS")
    for p in peaks:
        usrec[(idx > p) & (idx <= p + pd.DateOffset(months=6))] = 1.0
        spread[(idx >= p - pd.DateOffset(months=14)) & (idx <= p - pd.DateOffset(months=2))] -= 2.0
    raw = pd.DataFrame(index=idx)
    raw["GS10"] = tb3ms + spread
    raw["TB3MS"] = tb3ms
    raw["FEDFUNDS"] = tb3ms - 0.1
    raw["UNRATE"] = (5 - 0.3 * spread + rng.normal(0, 0.1, n)).clip(2, 14)
    raw["UMCSENT"] = (90 + 6 * spread + rng.normal(0, 3, n)).clip(50, 110)
    raw["INDPRO"] = (100 + np.cumsum(rng.normal(0.1, 0.3, n))).clip(40, None)
    raw["USREC"] = usrec
    raw["RECPROUSM156N"] = (usrec * 85 + rng.normal(2, 1, n)).clip(0, 100)
    raw["SAHMREALTIME"] = (usrec * 0.8 + rng.normal(0, 0.05, n))
    return raw


@pytest.fixture(scope="module")
def start_report(raw_with_nowcast):
    return rp.build_report(raw_with_nowcast, bootstrap=5)


def test_report_carries_nowcast_panel_outside_the_ensemble(start_report):
    rep = start_report
    assert rep["model_metadata"]["target_definition"] == "start"
    assert set(rep["nowcast"]["indicators"]) == {"Chauvet-Piger", "Sahm rule"}
    assert rep["recession_state"] == rep["nowcast"]["state"] == "expansion"
    assert rep["headline_applicable"] is True
    probs = rep["model_probabilities"]
    assert "Chauvet-Piger" not in probs and "Sahm rule" not in probs
    assert rep["ensemble_probability"] == pytest.approx(np.mean(list(probs.values())), abs=0.01)


def test_report_state_follows_latest_usrec(raw_with_nowcast):
    raw = raw_with_nowcast.copy()
    raw.loc[raw.index[-3]:, "USREC"] = 1.0  # a recession dated at the sample end
    rep = rp.build_report(raw, bootstrap=0)
    assert rep["recession_state"] == "nber_recession"
    assert rep["headline_applicable"] is False
    assert np.isfinite(rep["ensemble_probability"])  # still computed, just not headlined


def test_ui_headline_per_state():
    base = {"ensemble_probability": 12.4, "nowcast": {"indicators": {}}}
    expansion = {**base, "recession_state": "expansion"}
    flag = {**base, "recession_state": "nowcast_flag"}
    rec = {**base, "recession_state": "nber_recession"}
    assert nc.headline_value_text(expansion) == "12" and nc.state_note(expansion) == ""
    assert not nc.panel_is_prominent(expansion)
    assert nc.headline_value_text(flag) == "12" and "may already be under way" in nc.state_note(flag)
    assert nc.panel_is_prominent(flag)
    assert nc.headline_value_text(rec) == "—" and "withheld" in nc.state_note(rec)
    assert nc.panel_is_prominent(rec) and not nc.headline_applicable(rec)
    assert nc.headline_value_text({"error": "x"}) == "—"


def test_ui_panel_html_lists_both_indicators(start_report):
    html = nc.nowcast_panel_html(start_report)
    assert "Chauvet-Piger" in html and "Sahm rule" in html
    assert "threshold 50%" in html and "threshold 0.50 pp" in html
    assert "not part of the ensemble" in html
    assert "\n" not in html  # single line: Streamlit markdown would treat indents as code
    rows = nc.nowcast_rows(start_report)
    assert [r["name"] for r in rows] == ["Chauvet-Piger", "Sahm rule"]


def test_ui_panel_highlighted_in_recession():
    rep = {"ensemble_probability": 40.0, "recession_state": "nber_recession",
           "nowcast": {"indicators": {"Sahm rule": {"value": 0.9, "unit": "pp", "threshold": 0.5,
                                                    "signal": True, "as_of": "2030-01", "source": "x"}},
                       "usrec_latest": 1.0, "usrec_as_of": "2030-01", "state": "nber_recession"}}
    html = nc.nowcast_panel_html(rep)
    assert "HIGHLIGHTED" in html and "over threshold" in html and "withheld" in html
