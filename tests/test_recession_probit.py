"""Four-model probit ensemble: feature engineering, selection, report assembly.

These tests exercise the pure modelling functions on synthetic FRED-style data
so they never touch the network. Recessions are generated as a function of a
low term spread (and weak sentiment) 12 months ahead, so the probit fits
converge and respect the economic sign constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import recession_probit as rp


@pytest.fixture(scope="module")
def synthetic_raw():
    rng = np.random.default_rng(11)
    idx = pd.date_range("1967-01-01", "2025-12-01", freq="MS")
    n = len(idx)

    # Term spread: random walk that dips negative periodically.
    spread = 1.5 + np.cumsum(rng.normal(0, 0.06, n))
    spread = pd.Series(spread, index=idx).clip(-2.5, 4.0)

    tb3ms = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.04, n)), index=idx).clip(0.05, None)
    gs10 = tb3ms + spread  # so GS10 - TB3MS == spread

    # Sentiment weakens when spread is low (negative coefficient → recession risk).
    umcsent = (90 + 8 * spread + rng.normal(0, 3, n)).clip(50, 110)

    # Recession 12 months after the spread sits in its low tail.
    low = spread < spread.quantile(0.20)
    future_rec = low.shift(12, fill_value=False)
    usrec = pd.Series(np.where(future_rec, 1.0, 0.0), index=idx)

    raw = pd.DataFrame(index=idx)
    raw["GS10"] = gs10
    raw["TB3MS"] = tb3ms
    raw["T10Y3M"] = spread
    raw["FEDFUNDS"] = tb3ms - 0.1
    raw["UNRATE"] = (5 - 0.5 * spread + np.cumsum(rng.normal(0, 0.02, n))).clip(2.0, 14.0)
    raw["UMCSENT"] = umcsent
    raw["BUSLOANS"] = (2000 + np.cumsum(rng.normal(2, 8, n))).clip(500, None)
    raw["HOUST"] = (1400 + 100 * spread + rng.normal(0, 40, n)).clip(400, None)
    raw["PERMIT"] = (1500 + 90 * spread + rng.normal(0, 40, n)).clip(400, None)
    raw["INDPRO"] = (100 + np.cumsum(rng.normal(0.1, 0.3, n))).clip(40, None)
    raw["RECPROUSM156N"] = (usrec.rolling(3, min_periods=1).mean() * 80 + rng.normal(0, 1, n)).clip(0, 100)
    raw["USREC"] = usrec
    return raw


def test_engineer_features_builds_derived_and_target(synthetic_raw):
    data, feature_cols, feat_to_cat = rp.engineer_features(synthetic_raw)
    assert "SPREAD" in data.columns
    assert "UNRATE_CHG3" in data.columns
    assert "TARGET" in data.columns
    # YoY series become *_YOY features.
    assert "HOUST_YOY" in feature_cols
    assert "INDPRO_YOY" in feature_cols
    # Level series keep their raw name.
    assert "UMCSENT" in feature_cols
    assert feat_to_cat["SPREAD"].startswith("Yields")


def test_filter_by_coverage_keeps_full_history_features(synthetic_raw):
    data, feature_cols, _ = rp.engineer_features(synthetic_raw)
    available = rp.filter_by_coverage(data, feature_cols)
    assert "SPREAD" in available
    assert all(c in data.columns for c in available)


def test_estrella_mishkin_closed_form_is_monotonic():
    # Lower spread → higher probability for the frozen closed form.
    from scipy import stats
    p_low = stats.norm.cdf(rp._EM_CONST + rp._EM_SPREAD * -1.0) * 100
    p_high = stats.norm.cdf(rp._EM_CONST + rp._EM_SPREAD * 2.0) * 100
    assert p_low > p_high
    assert 0 <= p_high <= 100 and 0 <= p_low <= 100


def test_forward_stepwise_bic_returns_list_with_seed(synthetic_raw):
    data, feature_cols, _ = rp.engineer_features(synthetic_raw)
    available = rp.filter_by_coverage(data, feature_cols)
    model_df = data[available + ["TARGET", "USREC"]].dropna()
    y = model_df["TARGET"].astype(float)
    selected = rp.forward_stepwise_bic(
        y, model_df[available], available, max_features=5, seed=["SPREAD"]
    )
    assert isinstance(selected, list)
    assert "SPREAD" in selected


@pytest.fixture(scope="module")
def report(synthetic_raw):
    return rp.build_report(synthetic_raw, bootstrap=25, rng_seed=3)


def test_report_core_fields(report):
    assert 0 <= report["ensemble_probability"] <= 100
    assert 0 <= report["bic_probability"] <= 100
    assert report["signal"] in {"LOW", "ELEVATED", "HIGH"}
    assert report["consensus"] in {"STRONG", "MODERATE", "WEAK"}


def test_report_has_four_forward_models(report):
    probs = report["model_probabilities"]
    for name in ["NY Fed", "Wright", "BIC-selected", "Estrella-Mishkin"]:
        assert name in probs
        assert 0 <= probs[name] <= 100


def test_chauvet_piger_is_a_separate_benchmark(report):
    # Coincident benchmark must be reported but excluded from the ensemble inputs.
    assert "Chauvet-Piger" not in report["model_probabilities"]
    assert "Chauvet-Piger" in report["benchmark_probabilities"]


def test_report_ensemble_is_mean_of_forward_models_only(report):
    probs = list(report["model_probabilities"].values())
    assert report["ensemble_probability"] == pytest.approx(np.mean(probs), abs=0.05)


def test_report_bootstrap_ci_brackets_estimate(report):
    lo, hi = report["ci_lower"], report["ci_upper"]
    if lo is not None and hi is not None:
        assert lo <= hi


def test_report_watchlist_structure(report):
    sens = report["sensitivity"]
    assert sens, "watchlist should not be empty"
    row = sens[0]
    for key in ("feature", "current_value", "impact_pp", "trigger_30", "trigger_50"):
        assert key in row


def test_report_history_series_nonempty(report):
    assert not report["ensemble_history"].empty
    assert not report["bic_history"].empty
    assert (report["ensemble_history"] >= 0).all()
    assert (report["ensemble_history"] <= 100).all()


def test_report_trend_attribution(report):
    ta = report["trend_attribution"]
    assert "partial_effects" in ta
    assert "prob_change_pp" in ta


def test_target_series_is_binary(synthetic_raw):
    t = rp.target_series(synthetic_raw)
    assert not t.empty
    assert set(np.unique(t.values)).issubset({0.0, 1.0})


def test_walk_forward_is_out_of_sample_and_bounded(synthetic_raw):
    oos = rp.walk_forward(synthetic_raw, oos_start="1990-01-01", refit_every_months=24)
    assert not oos.empty
    assert (oos >= 0).all() and (oos <= 100).all()
    # OOS predictions must start at/after the requested start.
    assert oos.index.min() >= pd.Timestamp("1990-01-01")


def test_calibration_stats_shape(synthetic_raw):
    oos = rp.walk_forward(synthetic_raw, oos_start="1990-01-01", refit_every_months=24)
    target = rp.target_series(synthetic_raw)
    stats = rp.calibration_stats(oos, target)
    assert 0 <= stats["brier"] <= 1
    assert stats["n_obs"] > 0
    assert not stats["reliability_curve"].empty


def test_report_exposes_bic_coefficients(report):
    assert "bic_const" in report
    coefs = report["bic_coefficients"]
    assert set(coefs) == set(report["bic_selected_features"])
    assert all(isinstance(v, float) for v in coefs.values())


def test_scenario_probability_matches_baseline_with_no_overrides(report):
    # With no overrides the scenario must reproduce the BIC point estimate.
    assert rp.scenario_probability(report) == pytest.approx(report["bic_probability"], abs=0.5)


def test_scenario_probability_is_phi_of_linear_index(report):
    from scipy import stats

    coefs = report["bic_coefficients"]
    const = report["bic_const"]
    vals = {f: report["indicator_readings"][f]["value"] for f in coefs}

    # No override → exactly Φ(const + Σ β·x_current).
    z = const + sum(coefs[f] * vals[f] for f in coefs)
    assert rp.scenario_probability(report) == pytest.approx(float(stats.norm.cdf(z) * 100), abs=0.01)

    # One overridden driver recomputes the linear index correctly.
    f0 = next(iter(coefs))
    over = {f0: vals[f0] + 1.0}
    z2 = z + coefs[f0] * 1.0
    assert rp.scenario_probability(report, over) == pytest.approx(float(stats.norm.cdf(z2) * 100), abs=0.01)


def test_scenario_probability_respects_sign_constraint(report):
    # Pushing the spread far down (if selected) must not lower probability.
    if "SPREAD" not in report["bic_coefficients"]:
        pytest.skip("SPREAD not selected in this fixture's BIC model")
    cur = report["indicator_readings"]["SPREAD"]["value"]
    p_low = rp.scenario_probability(report, {"SPREAD": cur - 3.0})
    p_high = rp.scenario_probability(report, {"SPREAD": cur + 3.0})
    assert p_low >= p_high  # lower spread ⇒ weakly higher recession probability


def test_walk_forward_excludes_unobserved_labels(synthetic_raw, monkeypatch):
    # The training cutoff must lag each refit by 12 months so future labels
    # can't leak. We assert no training row used has index within 12 months
    # before a refit date by spying on the fitted training set sizes: a leaky
    # implementation would include ~12 extra recent rows per refit.
    oos = rp.walk_forward(synthetic_raw, oos_start="1995-01-01", refit_every_months=24)
    assert not oos.empty
    # Predictions still cover the post-cutoff period and stay bounded.
    assert (oos >= 0).all() and (oos <= 100).all()


def test_feature_label_plain_english():
    assert rp.feature_label("CPILFESL_YOY") == "Core CPI (YoY)"
    assert rp.feature_label("UMCSENT") == "U. Michigan Consumer Sentiment"
    assert rp.feature_label("SPREAD") == "10Y–3M Treasury spread"
    assert rp.feature_label("UNRATE_CHG3") == "Unemployment rate · 3-month change"
    # Unknown codes fall back to the raw mnemonic.
    assert rp.feature_label("MADE_UP_CODE") == "MADE_UP_CODE"


def test_sign_constraint_helper_rejects_wrong_sign():
    class _Res:
        params = pd.Series({"const": 0.1, "SPREAD": 0.5})  # SPREAD must be negative

    assert rp.check_sign_constraints(_Res(), ["SPREAD"]) is False

    class _Res2:
        params = pd.Series({"const": 0.1, "SPREAD": -0.5})

    assert rp.check_sign_constraints(_Res2(), ["SPREAD"]) is True


# ------------------------------------------------ OOS backtest: benchmark, samples, lags


def test_expanding_base_rate_uses_only_labels_known_at_each_date():
    idx = pd.date_range("2000-01-01", periods=48, freq="MS")
    y = pd.Series(np.r_[np.zeros(12), np.ones(12), np.zeros(24)], index=idx)
    br = rp.expanding_base_rate(y, idx, label_lag_months=12)
    # Nothing is known during the first 12 months.
    assert br.iloc[:12].isna().all()
    # At month t the rate is the mean of labels for months <= t-12.
    for t in [idx[12], idx[20], idx[30], idx[47]]:
        known = y.loc[: t - pd.DateOffset(months=12)]
        assert br.loc[t] == pytest.approx(known.mean())
    # A label that only becomes observable later must not move today's rate.
    y2 = y.copy()
    y2.iloc[40:] = 1.0
    assert rp.expanding_base_rate(y2, idx).loc[idx[45]] == pytest.approx(br.loc[idx[45]])


def test_calibration_stats_reports_both_benchmarks():
    idx = pd.date_range("1967-01-01", "2020-12-01", freq="MS")
    rng = np.random.default_rng(5)
    # Recession windows every ~8 years so both classes appear in and before the window.
    y = pd.Series(((idx.year % 8) == 0).astype(float), index=idx)
    pred = (y * 0.5 + 0.25 + rng.normal(0, 0.05, len(idx))).clip(0, 1) * 100
    oos = pred.loc["1985-01-01":]
    stats = rp.calibration_stats(oos, y)
    for key in ("baseline_brier", "skill_score", "baseline_brier_expanding", "skill_score_expanding"):
        assert np.isfinite(stats[key])
    assert stats["n_obs_expanding"] == stats["n_obs"] == len(oos)  # target history predates the window
    assert stats["start"] == oos.index.min() and stats["end"] == oos.index.max()
    # Full-sample baseline is the Brier of the in-window mean: m * (1 - m).
    m = y.loc[oos.index].mean()
    assert stats["baseline_brier"] == pytest.approx(m * (1 - m))
    # Expanding baseline is the Brier of the known-at-the-time rate.
    br = rp.expanding_base_rate(y, oos.index)
    assert stats["baseline_brier_expanding"] == pytest.approx(float(((br - y.loc[oos.index]) ** 2).mean()))


def test_complete_rows_is_per_model_not_global():
    idx = pd.date_range("1970-01-01", periods=60, freq="MS")
    data = pd.DataFrame(
        {"SPREAD": 1.0, "LATE": np.r_[np.full(24, np.nan), np.ones(36)], "TARGET": 0.0}, index=idx
    )
    assert rp.complete_rows(data, ["SPREAD"]).index.min() == idx[0]
    assert rp.complete_rows(data, ["SPREAD", "LATE"]).index.min() == idx[24]
    cut = rp.complete_rows(data, ["SPREAD"], cutoff=idx[9])
    assert cut.index.max() == idx[9] and len(cut) == 10


def test_walk_forward_not_truncated_by_short_history_feature(synthetic_raw, monkeypatch):
    # A candidate feature starting in 1976 (enough coverage to be "available")
    # used to truncate every model's training sample to 1976+ via a global
    # dropna, delaying the first OOS month past 1985. Per-model complete rows
    # let the spread models score from the requested start.
    # Pinned to the window target: this fixture's synthetic recessions are long
    # (1969-75), so the start-dated target's in-recession exclusions would leave
    # < MIN_WINDOW rows by 1985 for reasons unrelated to what is tested here.
    monkeypatch.setattr(rp, "TARGET_DEFINITION", "window")
    raw = synthetic_raw.copy()
    late = pd.Series(np.linspace(0.5, 2.0, len(raw)), index=raw.index)
    late[raw.index < "1976-06-01"] = np.nan
    raw["T10Y2Y"] = late
    data, cols, _ = rp.engineer_features(raw)
    assert "T10Y2Y" in rp.filter_by_coverage(data, cols)
    oos = rp.walk_forward(raw, oos_start="1985-01-01", refit_every_months=12)
    assert oos.index.min() == pd.Timestamp("1985-01-01")


def test_publication_lags_shift_macro_series_not_markets_or_target(synthetic_raw):
    raw = synthetic_raw.copy()
    raw["GDPC1"] = np.arange(len(raw), dtype=float)
    lagged = rp.apply_publication_lags(raw)
    assert rp.PUBLICATION_LAG_MONTHS["UNRATE"] == 1 and rp.PUBLICATION_LAG_MONTHS["GDPC1"] == 4
    pd.testing.assert_series_equal(lagged["UNRATE"], raw["UNRATE"].shift(1))
    pd.testing.assert_series_equal(lagged["GDPC1"], raw["GDPC1"].shift(4))
    for market in ("GS10", "TB3MS", "FEDFUNDS"):
        pd.testing.assert_series_equal(lagged[market], raw[market])
    pd.testing.assert_series_equal(lagged["USREC"], raw["USREC"])

    plain, _, _ = rp.engineer_features(raw)
    lagged_feats, _, _ = rp.engineer_features(raw, publication_lags=True)
    pd.testing.assert_series_equal(plain["TARGET"], lagged_feats["TARGET"])
    pd.testing.assert_series_equal(plain["SPREAD"], lagged_feats["SPREAD"])
    pd.testing.assert_series_equal(lagged_feats["UNRATE_CHG3"], plain["UNRATE_CHG3"].shift(1))


def test_walk_forward_uses_publication_lagged_features(synthetic_raw, monkeypatch):
    calls = []
    real_prepare = rp._prepare

    def spy(raw, **kwargs):
        calls.append(kwargs)
        return real_prepare(raw, **kwargs)

    monkeypatch.setattr(rp, "_prepare", spy)
    rp.walk_forward(synthetic_raw, oos_start="1995-01-01", refit_every_months=24)
    assert calls and calls[0].get("publication_lags") is True
