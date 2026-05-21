"""CAPE loader: source merge, bundled CSV, summary."""

from __future__ import annotations

import pandas as pd

from src.data.cape import (
    _load_bundled,
    _merge,
    _parse_shiller_full,
    cape_band,
    cape_extras_summary,
    cape_summary,
    ecy_band,
    fetch_cape_extras,
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
    assert list(out.columns) == ["date", "cape", "tr_cape", "ecy"]
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
