"""Recession ensemble: probability bounds, aggregation, sign constraints, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.recession_ensemble import RecessionEnsemble, _select_submodel, _YIELD_CURVE


@pytest.fixture(scope="module")
def fitted_ensemble():
    """A module-scoped fitted ensemble — fitting once is slow."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("1970-01-01", "2025-12-01", freq="MS")
    n = len(idx)

    def trended(level, scale):
        return level + np.cumsum(rng.normal(0, scale, size=n))

    panel = pd.DataFrame(index=idx)
    panel["DGS10"] = (4 + trended(0, 0.05)).clip(0.2, None)
    panel["DGS2"] = (panel["DGS10"] - 0.8).clip(0.1, None)
    panel["DGS3MO"] = (panel["DGS2"] - 0.4).clip(0.05, None)
    panel["DGS5"] = (panel["DGS10"] - 0.4).clip(0.1, None)
    panel["T10Y3M"] = panel["DGS10"] - panel["DGS3MO"]
    panel["T10Y2Y"] = panel["DGS10"] - panel["DGS2"]
    panel["UNRATE"] = (5 + trended(0, 0.04)).clip(2.0, None)
    panel["ICSA"] = 350_000 + rng.normal(0, 5_000, n)
    panel["JTSJOL"] = (7_000 + trended(0, 50)).clip(2_000, None)
    panel["JTSQUR"] = (2.5 + rng.normal(0, 0.1, n)).clip(1.0, None)
    panel["BAA10Y"] = (2 + rng.normal(0, 0.3, n)).clip(0.5, None)
    panel["DRTSCILM"] = rng.normal(0, 10, n)
    panel["BAMLH0A0HYM2"] = (5 + rng.normal(0, 0.5, n)).clip(2.0, None)
    panel["PERMIT"] = (1_500 + trended(0, 5)).clip(500, None)
    panel["HOUST"] = (1_400 + trended(0, 5)).clip(400, None)
    panel["PCEC96"] = 10_000 + np.arange(n) * 5
    panel["VIXCLS"] = (18 + np.abs(rng.normal(0, 5, n))).clip(8, None)
    panel["USSLIND"] = 1.5 + rng.normal(0, 0.3, n)
    panel["SP500"] = (1_000 + np.arange(n) * 3).clip(300, None)

    from src.data.nber import load_nber_recessions, recession_in_next_12m

    nber = load_nber_recessions(start="1970-01-01", end="2025-12-31")
    fwd = recession_in_next_12m(nber)

    ensemble = RecessionEnsemble()
    ensemble.fit(panel, fwd)
    return ensemble, panel, fwd


def test_submodels_return_probabilities_in_unit_interval(fitted_ensemble):
    ensemble, _, _ = fitted_ensemble
    current = ensemble.predict_current()
    for name, prob in current["submodels"].items():
        assert 0.0 <= prob <= 100.0, f"{name} probability out of range: {prob}"


def test_ensemble_is_mean_of_submodels(fitted_ensemble):
    ensemble, _, _ = fitted_ensemble
    current = ensemble.predict_current()
    expected = float(np.mean(list(current["submodels"].values())))
    assert abs(current["ensemble"] - expected) < 1e-9


def test_history_has_expected_columns(fitted_ensemble):
    ensemble, _, _ = fitted_ensemble
    hist = ensemble.predict_history()
    assert "ensemble" in hist.columns
    assert (hist["ensemble"] >= 0).all() and (hist["ensemble"] <= 100).all()


def test_brier_and_auc_computable(fitted_ensemble):
    ensemble, _, _ = fitted_ensemble
    stats = ensemble.calibration_stats()
    assert np.isfinite(stats["brier"])
    assert 0.0 <= stats["brier"] <= 1.0
    if np.isfinite(stats["auc"]):
        assert 0.0 <= stats["auc"] <= 1.0


def test_sign_constraints_drop_wrong_sign_variables():
    """If we feed a feature whose data argues for the opposite sign, it should be dropped."""
    # Build a tiny dataset where t10y3m is positively correlated with y (wrong sign).
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("1980-01-01", periods=n, freq="MS")
    y = (rng.uniform(size=n) > 0.7).astype(int)
    X = pd.DataFrame(
        {
            "t10y3m": y + rng.normal(0, 0.1, n),  # POSITIVELY correlated with y — wrong sign
            "t10y2y": -y + rng.normal(0, 0.1, n),  # correct sign (negative coef)
            "term_premium_proxy": -y + rng.normal(0, 0.1, n),
        },
        index=idx,
    )
    fitted = _select_submodel(_YIELD_CURVE, X, pd.Series(y, index=idx))
    assert fitted is not None
    # t10y3m must be dropped because its sign disagrees with the prior.
    assert "t10y3m" not in fitted.feature_names
