"""Atlanta Fed Market Probability Tracker parsing."""

from __future__ import annotations

import pandas as pd

from src.data.market_probability import (
    directional_probs,
    latest_snapshot,
    load_market_probabilities,
    parse_market_probability,
    probability_buckets,
    rate_path,
)


SAMPLE = (
    "date,reference_start_date,target_range,field,value\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Rate: 25th percentile,471.21\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Rate: mean,481.98\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Rate: mode,498.33\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Rate: 75th percentile,518.52\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Prob: cut,27.98\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Prob: hike,45.18\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Prob: 450bps - 475bps,13.89\n"
    "2023-03-29,2023-06-21,475bps - 500bps,Prob: 475bps - 500bps,28.22\n"
)


def test_parses_long_frame():
    df = parse_market_probability(SAMPLE)
    assert list(df.columns) == ["snapshot_date", "meeting_date", "target_range", "field", "value"]
    assert len(df) == 8
    assert df["snapshot_date"].iloc[0] == pd.Timestamp("2023-03-29")
    assert df["meeting_date"].iloc[0] == pd.Timestamp("2023-06-21")


def test_probability_buckets_extracts_ranges_and_excludes_hike_cut():
    buckets = probability_buckets(parse_market_probability(SAMPLE))
    assert len(buckets) == 2  # only the two "Prob: <range>" rows
    row = buckets[buckets["low_bps"] == 475].iloc[0]
    assert row["high_bps"] == 500
    assert row["probability"] == 28.22
    # hike/cut must not leak into the bucket view.
    assert set(buckets["low_bps"]) == {450, 475}


def test_directional_probs():
    d = directional_probs(parse_market_probability(SAMPLE))
    assert len(d) == 1
    assert d.iloc[0]["prob_hike"] == 45.18
    assert d.iloc[0]["prob_cut"] == 27.98


def test_rate_path_pivots_published_stats():
    rp = rate_path(parse_market_probability(SAMPLE))
    assert list(rp.columns) == ["mean", "mode", "p25", "p75"]
    row = rp.loc[pd.Timestamp("2023-06-21")]
    assert row["mean"] == 481.98
    assert row["mode"] == 498.33
    assert row["p25"] == 471.21
    assert row["p75"] == 518.52


def test_latest_snapshot_picks_max_date():
    text = SAMPLE + "2023-04-05,2023-06-21,475bps - 500bps,Rate: mean,485.0\n"
    df = parse_market_probability(text)
    latest = latest_snapshot(df)
    assert (latest["snapshot_date"] == pd.Timestamp("2023-04-05")).all()
    assert len(latest) == 1


def test_empty_input_returns_empty_frame():
    df = parse_market_probability("date,reference_start_date,target_range,field,value\n")
    assert df.empty
    assert list(df.columns) == ["snapshot_date", "meeting_date", "target_range", "field", "value"]


def test_bundled_file_loads_and_is_sane():
    df = load_market_probabilities()
    assert not df.empty
    # Real export spans multiple years and many meetings.
    assert df["snapshot_date"].nunique() > 100
    # Bucket probabilities for a single (snapshot, meeting) group sum to ~100.
    buckets = probability_buckets(df)
    grp = buckets.groupby(["snapshot_date", "meeting_date"])["probability"].sum()
    near_100 = grp[(grp > 95) & (grp < 105)]
    assert len(near_100) > 0
    # Rate path on the latest snapshot is populated and ordered.
    rp = rate_path(df)
    assert not rp.empty
    assert rp.index.is_monotonic_increasing
