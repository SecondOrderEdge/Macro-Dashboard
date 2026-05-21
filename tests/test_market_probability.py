"""Atlanta Fed Market Probability Tracker parsing."""

from __future__ import annotations

import pandas as pd

from src.data.market_probability import (
    expected_rate_path,
    latest_snapshot,
    load_market_probabilities,
    parse_market_probability,
)


SAMPLE = (
    "2023-04-17\t2023-06-21\t475bps - 500bps\tProb: 400bps - 425bps\t0.74\n"
    "2023-04-17\t2023-06-21\t475bps - 500bps\tProb: 425bps - 450bps\t0.49\n"
    "2023-04-17\t2023-06-21\t475bps - 500bps\tProb: 450bps - 475bps\t1.69\n"
)


def test_parses_rows_and_ranges():
    df = parse_market_probability(SAMPLE)
    assert len(df) == 3
    row = df.iloc[0]
    assert row["snapshot_date"] == pd.Timestamp("2023-04-17")
    assert row["meeting_date"] == pd.Timestamp("2023-06-21")
    assert row["current_low_bps"] == 475
    assert row["current_high_bps"] == 500
    # Strips the "Prob:" label and the "bps" suffix from the target bucket.
    assert row["target_low_bps"] == 400
    assert row["target_high_bps"] == 425
    assert row["probability"] == 0.74


def test_skips_header_and_blank_lines():
    text = "snapshot\tmeeting\tcurrent\ttarget\tprob\n\n" + SAMPLE
    df = parse_market_probability(text)
    assert len(df) == 3


def test_tolerates_whitespace():
    text = "2023-04-17 \t 2023-06-21\t 475bps - 500bps \t Prob: 450bps - 475bps \t 1.69 \n"
    df = parse_market_probability(text)
    assert len(df) == 1
    assert df.iloc[0]["target_low_bps"] == 450
    assert df.iloc[0]["probability"] == 1.69


def test_empty_input_returns_empty_frame():
    df = parse_market_probability("")
    assert df.empty
    assert list(df.columns) == [
        "snapshot_date",
        "meeting_date",
        "current_low_bps",
        "current_high_bps",
        "target_low_bps",
        "target_high_bps",
        "probability",
    ]


def test_latest_snapshot_picks_max_date():
    text = SAMPLE + "2023-05-01\t2023-06-21\t500bps - 525bps\tProb: 475bps - 500bps\t90.0\n"
    df = parse_market_probability(text)
    latest = latest_snapshot(df)
    assert (latest["snapshot_date"] == pd.Timestamp("2023-05-01")).all()
    assert len(latest) == 1


def test_expected_rate_path_is_probability_weighted_midpoint():
    # Two buckets: 400-425 (mid 412.5) at 25%, 450-475 (mid 462.5) at 75%.
    text = (
        "2024-01-01\t2024-03-20\t450bps - 475bps\tProb: 400bps - 425bps\t25\n"
        "2024-01-01\t2024-03-20\t450bps - 475bps\tProb: 450bps - 475bps\t75\n"
    )
    df = parse_market_probability(text)
    path = expected_rate_path(df)
    expected = (412.5 * 25 + 462.5 * 75) / 100.0
    assert path.loc[pd.Timestamp("2024-03-20")] == expected


def test_bundled_file_loads():
    df = load_market_probabilities()
    assert not df.empty
    assert {"snapshot_date", "meeting_date", "probability"}.issubset(df.columns)
