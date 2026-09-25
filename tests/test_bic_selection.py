"""Pre-registered BIC-member selection: SPREAD forced, stationary sign-restricted
pool, cap of 4 features, reselected inside every walk-forward fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import recession_probit as rp

LEVELS = {"FEDFUNDS", "GS10", "TB3MS", "UNRATE", "TCU", "DRALACBS", "T10Y2Y", "T10Y3M",
          "USSLIND", "PCECC96_YOY", "CPIAUCSL_YOY", "CPILFESL_YOY", "PCEPI_YOY",
          "PCEPILFE_YOY", "PPIACO_YOY"}


def test_pool_excludes_levels_inflation_and_duplicate_spreads():
    assert rp.MAX_FEATURES_BIC == 4
    assert rp.BIC_FORCED_FEATURE == "SPREAD"
    assert not (set(rp.BIC_POOL_SIGNS) & LEVELS)
    assert set(rp.BIC_POOL_SIGNS.values()) <= {"negative", "positive"}
    assert rp.BIC_SIGNS["SPREAD"] == "negative"
    assert rp.BIC_POOL_SIGNS["UNRATE_CHG3"] == "positive"
    assert rp.BIC_POOL_SIGNS["PERMIT_YOY"] == "negative"


def _panel(n_extra_good: int = 5, seed: int = 0) -> pd.DataFrame:
    """Labelled data where several pool features carry signal (with the right
    sign), a rate level carries strong signal, and one pool feature carries
    signal with the WRONG sign."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1970-01-01", "2020-12-01", freq="MS")
    n = len(idx)
    z = rng.normal(size=n)
    data = pd.DataFrame(index=idx)
    data["SPREAD"] = -0.8 * z + rng.normal(0, 1, n)
    good = ["PERMIT_YOY", "HOUST_YOY", "IPMAN_YOY", "PAYEMS_YOY", "RSAFS_YOY"][:n_extra_good]
    for g in good:
        data[g] = -0.8 * z + rng.normal(0, 1, n)          # negative sign expected: OK
    data["ICSA_YOY"] = -0.8 * z + rng.normal(0, 1, n)       # positive expected: wrong sign
    data["FEDFUNDS"] = 2.0 * z + rng.normal(0, 0.3, n)      # strong, but a level: not a candidate
    data["CPILFESL_YOY"] = 2.0 * z + rng.normal(0, 0.3, n)  # inflation: not a candidate
    latent = 1.2 * z + rng.normal(0, 0.7, n) - 1.3
    data["TARGET"] = (latent > 0).astype(float)
    data.loc[idx[-12:], "TARGET"] = np.nan
    return data


def test_selection_forces_spread_caps_and_uses_only_the_pool():
    data = _panel()
    sel = rp.select_bic_features(data)
    assert sel[0] == "SPREAD"
    assert 1 <= len(sel) <= 4
    assert set(sel) <= {"SPREAD", *rp.BIC_POOL_SIGNS}
    assert "FEDFUNDS" not in sel and "CPILFESL_YOY" not in sel
    assert "ICSA_YOY" not in sel  # wrong sign is rejected


def test_selected_model_respects_every_sign():
    data = _panel()
    sel = rp.select_bic_features(data)
    rows = rp.complete_rows(data, sel)
    res = rp._fit_probit(rows["TARGET"], rows[sel])
    assert rp.check_sign_constraints(res, sel, rp.BIC_SIGNS)


def test_selection_respects_cutoff_and_small_samples():
    data = _panel()
    # Fewer than MIN_WINDOW labelled rows up to the cutoff → spread alone.
    assert rp.select_bic_features(data, cutoff=data.index[60]) == ["SPREAD"]
    # Future rows cannot change a fold's selection.
    cut = data.index[300]
    a = rp.select_bic_features(data, cutoff=cut)
    tampered = data.copy()
    tampered.loc[tampered.index > cut, "TARGET"] = 1.0 - tampered.loc[tampered.index > cut, "TARGET"]
    assert rp.select_bic_features(tampered, cutoff=cut) == a


def test_low_coverage_pool_feature_is_ineligible():
    data = _panel()
    data["JTSJOL_YOY"] = np.where(np.arange(len(data)) > len(data) * 0.5, -data["SPREAD"] * 3, np.nan)
    assert "JTSJOL_YOY" not in rp.select_bic_features(data)


@pytest.fixture(scope="module")
def raw_start():
    rng = np.random.default_rng(3)
    idx = pd.date_range("1967-01-01", "2025-12-01", freq="MS")
    n = len(idx)
    spread = pd.Series(1.5 + np.cumsum(rng.normal(0, 0.06, n)), index=idx).clip(-2.5, 4.0)
    tb3ms = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.04, n)), index=idx).clip(0.05, None)
    usrec = pd.Series(0.0, index=idx)
    for p in pd.date_range("1972-06-01", "2020-06-01", freq="72MS"):
        usrec[(idx > p) & (idx <= p + pd.DateOffset(months=6))] = 1.0
        spread[(idx >= p - pd.DateOffset(months=14)) & (idx <= p - pd.DateOffset(months=2))] -= 2.0
    raw = pd.DataFrame(index=idx)
    raw["GS10"] = tb3ms + spread
    raw["TB3MS"] = tb3ms
    raw["FEDFUNDS"] = tb3ms - 0.1
    raw["UNRATE"] = (5 - 0.3 * spread + rng.normal(0, 0.1, n)).clip(2, 14)
    raw["UMCSENT"] = (90 + 6 * spread + rng.normal(0, 3, n)).clip(50, 110)
    raw["PERMIT"] = (1500 + 90 * spread + rng.normal(0, 40, n)).clip(400, None)
    raw["INDPRO"] = (100 + np.cumsum(rng.normal(0.1, 0.3, n))).clip(40, None)
    raw["CPILFESL"] = 100 * np.exp(np.cumsum(rng.normal(0.003, 0.002, n)))
    raw["USREC"] = usrec
    return raw


def test_live_report_uses_the_pre_registered_rule(raw_start):
    rep = rp.build_report(raw_start, bootstrap=0)
    feats = rep["bic_selected_features"]
    assert feats[0] == "SPREAD" and len(feats) <= 4
    assert set(feats) <= {"SPREAD", *rp.BIC_POOL_SIGNS}


def test_walk_forward_reselects_inside_each_fold(raw_start, monkeypatch):
    calls = []
    real = rp.select_bic_features

    def spy(data, cutoff=None):
        calls.append(cutoff)
        return real(data, cutoff=cutoff)

    monkeypatch.setattr(rp, "select_bic_features", spy)
    oos = rp.walk_forward(raw_start, oos_start="1995-01-01", refit_every_months=24)
    sel = oos.attrs["bic_selections"]
    refits = pd.date_range("1995-01-01", periods=len(sel), freq="24MS")
    assert list(sel) == [r.strftime("%Y-%m") for r in refits]
    fold_cutoffs = [c for c in calls if c is not None]
    assert fold_cutoffs == [r - pd.DateOffset(months=12) for r in refits]
    for feats in sel.values():
        assert feats[0] == "SPREAD" and len(feats) <= 4
