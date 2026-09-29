"""CAPE loader: source merge, bundled CSV, summary."""

from __future__ import annotations

import pandas as pd

import numpy as np

from src.data.cape import (
    _load_bundled,
    _merge,
    _parse_shiller_full,
    cape_band,
    cape_extras_summary,
    cape_summary,
    ecy_band,
    fetch_cape_extras,
    fit_cape_expected_return,
    implied_10y_real_return,
    implied_return_band,
    load_cape_return_history,
    subsequent_10y_real_return,
)


def _series(pairs, source):
    idx = pd.to_datetime([d for d, _ in pairs])
    s = pd.Series([v for _, v in pairs], index=idx, name="cape")
    s.attrs["source"] = source
    return s


def test_merge_later_sources_win_per_month():
    base = _series([("2023-01-01", 28.0), ("2023-02-01", 28.5)], "github")
    fresh = _series([("2023-02-01", 30.0), ("2023-03-01", 31.0)], "bundled")
    merged = _merge([base, fresh])
    assert merged.loc["2023-01-01"] == 28.0          # only in base
    assert merged.loc["2023-02-01"] == 30.0          # fresh overrides base
    assert merged.loc["2023-03-01"] == 31.0          # only in fresh
    assert merged.index.is_monotonic_increasing
    assert "github" in merged.attrs["source"] and "bundled" in merged.attrs["source"]


def test_merge_skips_empty():
    only = _series([("2024-01-01", 33.0)], "bundled")
    merged = _merge([pd.Series(dtype=float), only, pd.Series(dtype=float)])
    assert len(merged) == 1
    assert merged.attrs["source"] == "bundled"


def test_load_bundled_reads_date_cape(tmp_path, monkeypatch):
    csv = tmp_path / "cape.csv"
    csv.write_text("date,cape\n2025-11-01,38.1\n2025-12-01,38.6\n")
    monkeypatch.setattr("src.data.cape._BUNDLED_PATH", csv)
    s = _load_bundled([])
    assert len(s) == 2
    assert s.loc["2025-12-01"] == 38.6
    assert s.attrs["source"] == "bundled:data/cape.csv"


def test_load_bundled_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.cape._BUNDLED_PATH", tmp_path / "nope.csv")
    assert _load_bundled([]).empty


def test_cape_summary_and_band():
    idx = pd.date_range("1990-01-01", periods=420, freq="MS")
    s = pd.Series([15 + (i % 25) for i in range(len(idx))], index=idx, name="cape")
    summ = cape_summary(s)
    assert summ["as_of"] == idx[-1]
    assert 0 <= summ["modern_percentile"] <= 100
    assert cape_band(10)[0] == "CHEAP"
    assert cape_band(95)[0] == "EXTREME"


def test_bundled_file_loads():
    # The committed data/cape.csv must parse via the real path.
    from src.data.cape import _BUNDLED_PATH

    assert _BUNDLED_PATH.exists()
    s = _load_bundled([])
    assert not s.empty
    assert s.index.is_monotonic_increasing


def test_ecy_band_inverted_vs_cape():
    # High ECY (high percentile) is attractive; low ECY is rich.
    assert ecy_band(90)[0] == "ATTRACTIVE"
    assert ecy_band(5)[1] == "critical"


def test_cape_extras_summary():
    idx = pd.date_range("2000-01-01", periods=320, freq="MS")
    ex = pd.DataFrame(
        {"tr_cape": [20 + (i % 30) for i in range(len(idx))],
         "ecy": [0.02 + 0.0001 * (i % 50) for i in range(len(idx))]},
        index=idx,
    )
    summ = cape_extras_summary(ex)
    assert summ["tr_cape"]["today"] == ex["tr_cape"].iloc[-1]
    assert 0 <= summ["ecy"]["modern_percentile"] <= 100


def test_parse_shiller_full_extracts_three_series():
    df = pd.DataFrame({
        "Date": [2026.03, 2026.04, 2026.05],
        "CAPE": [36.94, 38.14, 39.58],
        "TR CAPE": [39.47, 40.70, 42.22],
        "Yield": [0.0178, 0.0163, 0.0139],
    })
    out = _parse_shiller_full(df)
    assert list(out.columns) == ["date", "cape", "tr_cape", "ecy", "real_tr_price"]
    assert out["real_tr_price"].isna().all()
    assert out["date"].iloc[-1] == pd.Timestamp("2026-05-01")
    assert out["cape"].iloc[-1] == 39.58
    assert out["tr_cape"].iloc[-1] == 42.22
    assert out["ecy"].iloc[-1] == 0.0139


def test_parse_shiller_full_rejects_nonfraction_yield():
    # If a 'Yield' column isn't the ECY fraction (values >> 1), it's dropped.
    df = pd.DataFrame({
        "Date": [2026.04, 2026.05],
        "CAPE": [38.14, 39.58],
        "Yield": [4.2, 4.1],  # looks like a bond yield in %, not ECY fraction
    })
    out = _parse_shiller_full(df)
    assert out["ecy"].isna().all()


def test_fetch_cape_extras_has_columns():
    ex = fetch_cape_extras()
    assert list(ex.columns) == ["tr_cape", "ecy"]
    assert not ex.empty  # the committed cape.csv now carries the extras


def test_parse_shiller_full_real_tr_price_is_second_price_column():
    # skiprows=7 collapses both price indexes to "Price"; pandas names the
    # real total-return index Price.1 (Data-sheet column 9).
    df = pd.DataFrame(
        [[1881.01, 90.0, 100.0, 15.0], [1881.02, 91.0, 110.0, 16.0]],
        columns=["Date", "Price", "Price.1", "CAPE"],
    )
    out = _parse_shiller_full(df)
    assert out["real_tr_price"].tolist() == [100.0, 110.0]


def test_subsequent_10y_real_return_compounds_a_doubling():
    idx = pd.date_range("2000-01-01", periods=121, freq="MS")
    price = pd.Series(100.0, index=idx)
    price.iloc[-1] = 200.0
    realized = subsequent_10y_real_return(price)
    assert abs(realized.iloc[0] - (2.0 ** 0.1 - 1.0)) < 1e-12
    assert realized.iloc[1:].isna().all()


def test_subsequent_return_rejects_a_calendar_gap():
    idx = pd.date_range("2000-01-01", periods=130, freq="MS")
    price = pd.Series(np.linspace(100.0, 250.0, len(idx)), index=idx)
    price = price.drop(idx[10])
    realized = subsequent_10y_real_return(price)
    # Position 0's partner 120 rows later is no longer 120 calendar months later.
    assert pd.isna(realized.iloc[0])


def test_fit_recovers_inverse_cape_line():
    idx = pd.date_range("1900-01-01", periods=400, freq="MS")
    cape = pd.Series(np.linspace(8.0, 40.0, len(idx)), index=idx)
    true_a, true_b = 0.01, 0.9
    fwd = true_a + true_b / cape
    fit = fit_cape_expected_return(cape, fwd)
    assert fit["predictor"] == "inverse_cape"
    assert fit["n"] == 400
    assert abs(fit["alpha"] - true_a) < 1e-10
    assert abs(fit["beta"] - true_b) < 1e-8
    assert fit["r2"] > 0.999
    got = implied_10y_real_return(20.0, fit)
    assert abs(got - (true_a + true_b / 20.0)) < 1e-10
    # Higher CAPE maps to a lower implied real return.
    assert implied_10y_real_return(40.0, fit) < implied_10y_real_return(15.0, fit)


def test_fit_empty_on_short_sample():
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    cape = pd.Series(20.0, index=idx)
    fwd = pd.Series(0.05, index=idx)
    assert fit_cape_expected_return(cape, fwd) == {}
    assert np.isnan(implied_10y_real_return(20.0, {}))


def test_implied_return_band():
    assert implied_return_band(0.08) == ("ABOVE AVG", "low")
    assert implied_return_band(0.05)[0] == "BELOW AVG"
    assert implied_return_band(0.028) == ("LOW", "high")
    assert implied_return_band(0.0)[1] == "critical"


def test_load_return_history_without_price_column(tmp_path, monkeypatch):
    csv = tmp_path / "cape.csv"
    csv.write_text("date,cape\n2000-01-01,20\n2000-02-01,21\n")
    monkeypatch.setattr("src.data.cape._BUNDLED_PATH", csv)
    hist = load_cape_return_history()
    assert list(hist.columns) == ["cape", "real_tr_price", "fwd_real_return_10y"]
    assert hist["fwd_real_return_10y"].isna().all()
    assert fit_cape_expected_return(hist["cape"], hist["fwd_real_return_10y"]) == {}


def test_bundled_cape_return_history_matches_shiller_checkpoints():
    # Published "10 Year Annualized Stock Real Return" from Shiller's workbook.
    # These deep-history windows are invariant to a CPI rebase (the price index
    # scales, the 10-year ratio does not).
    expected = {
        "1881-01-01": 0.04535327605849804,
        "1929-09-01": -0.014156423600160895,
        "1982-08-01": 0.1431275985462923,
        "2000-01-01": -0.030322393887708188,
        "2009-03-01": 0.14270204314620294,
    }
    hist = load_cape_return_history()
    assert hist["real_tr_price"].dropna().gt(0).all()
    for date, value in expected.items():
        got = hist.loc[pd.Timestamp(date), "fwd_real_return_10y"]
        assert abs(got - value) < 1e-9, date
    # The last ten years have not elapsed.
    assert hist["fwd_real_return_10y"].iloc[-120:].isna().all()
    assert np.isfinite(hist["fwd_real_return_10y"].iloc[-121])

    fit = fit_cape_expected_return(hist["cape"], hist["fwd_real_return_10y"])
    assert fit["predictor"] == "inverse_cape"
    assert fit["beta"] > 0
    assert fit["n"] > 1000
    assert 0.15 < fit["r2"] < 0.6
    assert fit["sample_start"] == pd.Timestamp("1881-01-01")
    today = float(hist["cape"].dropna().iloc[-1])
    implied = implied_10y_real_return(today, fit)
    # Current CAPE is expensive; the historical fit should be well below the
    # long-run average subsequent real return (~7%).
    assert -0.02 < implied < 0.05
    assert implied_10y_real_return(40.0, fit) < implied_10y_real_return(15.0, fit)
