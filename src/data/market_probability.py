"""Atlanta Fed Market Probability Tracker loader.

The Atlanta Fed's Market Probability Tracker estimates, from CME options on
SOFR futures, the market-implied probability that the FOMC's target range will
sit in each 25bp bucket after a given meeting. See
https://www.atlantafed.org/cenfis/market-probability-tracker.

The tracker is published behind a bot-blocking front end (and many deployment
environments restrict outbound access to it), so this module reads a bundled
tab-separated export rather than fetching live. Refresh it by replacing
``data/market_probability_tracker.tsv`` with a newer export in the same shape:

    snapshot_date <tab> meeting_date <tab> current_range <tab> target_range <tab> probability
    2023-04-17    2023-06-21    475bps - 500bps    Prob: 400bps - 425bps    0.74

``probability`` is a percentage (0.74 means 0.74%).
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "market_probability_tracker.tsv"

COLUMNS = [
    "snapshot_date",
    "meeting_date",
    "current_low_bps",
    "current_high_bps",
    "target_low_bps",
    "target_high_bps",
    "probability",
]

# Two integers separated by any run of non-digits. Tolerates the "bps" suffix
# and the leading "Prob:" label, e.g. "475bps - 500bps" and "Prob: 400bps - 425bps".
_RANGE_RE = r"(\d+)\D+(\d+)"


def parse_market_probability(text: str) -> pd.DataFrame:
    """Parse the tab-separated Market Probability Tracker export into a tidy frame.

    Returns a DataFrame with :data:`COLUMNS`. Malformed lines (headers, blanks,
    rows missing a parseable date/range/probability) are dropped rather than
    raising, so a stray header row in an export won't break the load.
    """
    return _read(StringIO(text))


def load_market_probabilities(path: Path | str | None = None) -> pd.DataFrame:
    """Load the bundled (or supplied) Market Probability Tracker export."""
    path = Path(path) if path else _DEFAULT_PATH
    return _read(path)


def _read(buffer) -> pd.DataFrame:
    raw = pd.read_csv(
        buffer,
        sep="\t",
        header=None,
        dtype=str,
        names=["snapshot_date", "meeting_date", "current_range", "target_range", "probability"],
        engine="python",
        skip_blank_lines=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    for col in raw.columns:
        raw[col] = raw[col].str.strip()

    snapshot = pd.to_datetime(raw["snapshot_date"], errors="coerce")
    meeting = pd.to_datetime(raw["meeting_date"], errors="coerce")
    current = raw["current_range"].str.extract(_RANGE_RE)
    target = raw["target_range"].str.extract(_RANGE_RE)
    probability = pd.to_numeric(raw["probability"], errors="coerce")

    out = pd.DataFrame(
        {
            "snapshot_date": snapshot,
            "meeting_date": meeting,
            "current_low_bps": pd.to_numeric(current[0], errors="coerce"),
            "current_high_bps": pd.to_numeric(current[1], errors="coerce"),
            "target_low_bps": pd.to_numeric(target[0], errors="coerce"),
            "target_high_bps": pd.to_numeric(target[1], errors="coerce"),
            "probability": probability,
        }
    )

    out = out.dropna(subset=["snapshot_date", "meeting_date", "target_low_bps", "probability"])
    int_cols = ["current_low_bps", "current_high_bps", "target_low_bps", "target_high_bps"]
    out[int_cols] = out[int_cols].astype("Int64")
    out = out.sort_values(["snapshot_date", "meeting_date", "target_low_bps"]).reset_index(drop=True)
    return out[COLUMNS]


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for the most recent ``snapshot_date`` only."""
    if df.empty:
        return df
    latest = df["snapshot_date"].max()
    return df[df["snapshot_date"] == latest].copy()


def expected_rate_path(df: pd.DataFrame, snapshot: str | pd.Timestamp | None = None) -> pd.Series:
    """Probability-weighted expected target-range midpoint (bps) per meeting.

    Uses the latest snapshot unless ``snapshot`` is given. Each meeting's value
    is ``sum(prob * midpoint) / sum(prob)`` over its buckets, so it is robust to
    a snapshot's probabilities not summing to exactly 100.
    """
    if df.empty:
        return pd.Series(dtype=float, name="expected_bps")
    sub = latest_snapshot(df) if snapshot is None else df[df["snapshot_date"] == pd.Timestamp(snapshot)]
    if sub.empty:
        return pd.Series(dtype=float, name="expected_bps")
    sub = sub.copy()
    mid = (sub["target_low_bps"].astype(float) + sub["target_high_bps"].astype(float)) / 2.0
    weighted = (mid * sub["probability"]).groupby(sub["meeting_date"]).sum()
    weights = sub["probability"].groupby(sub["meeting_date"]).sum()
    return (weighted / weights).rename("expected_bps")
