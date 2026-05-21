"""Atlanta Fed Market Probability Tracker loader.

The Atlanta Fed's Market Probability Tracker estimates, from CME options on
SOFR futures, the market-implied distribution of the FOMC's policy rate after
each of the next several quarterly contract expiries. See
https://www.atlantafed.org/cenfis/market-probability-tracker.

The tracker is published behind a bot-blocking front end (and many deployment
environments restrict outbound access to it), so this module reads a bundled
CSV export rather than fetching live. Refresh it by replacing
``data/market_probability_tracker.csv`` with a newer export in the same shape:

    date,reference_start_date,target_range,field,value
    2023-03-29,2023-06-21,475bps - 500bps,Rate: mean,481.98
    2023-03-29,2023-06-21,475bps - 500bps,Prob: 475bps - 500bps,28.22
    2023-03-29,2023-06-21,475bps - 500bps,Prob: cut,27.98

Columns:
    date                 snapshot date (when the market implied this)
    reference_start_date meeting / reference-quarter start being priced
    target_range         current target range at the snapshot (raw string)
    field                one of:
                           "Rate: mean" / "Rate: mode" /
                           "Rate: 25th percentile" / "Rate: 75th percentile"
                             -> value is a rate in basis points (481.98 = 4.82%)
                           "Prob: <low>bps - <high>bps"
                             -> value is the probability (%) of that 25bp bucket
                           "Prob: hike" / "Prob: cut"
                             -> value is the probability (%) of a hike / cut
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "market_probability_tracker.csv"

# Long-form columns returned by the loader/parser.
COLUMNS = ["snapshot_date", "meeting_date", "target_range", "field", "value"]

# Two integers separated by any run of non-digits — tolerates the "bps" suffix
# and the "Prob:" label, e.g. "475bps - 500bps" / "Prob: 400bps - 425bps".
_RANGE_RE = r"(\d+)\D+(\d+)"

# Map the published "Rate:" field labels to short column names.
_RATE_FIELDS = {
    "Rate: mean": "mean",
    "Rate: mode": "mode",
    "Rate: 25th percentile": "p25",
    "Rate: 75th percentile": "p75",
}


def parse_market_probability(text: str) -> pd.DataFrame:
    """Parse the Market Probability Tracker CSV text into a tidy long frame."""
    return _read(StringIO(text))


def load_market_probabilities(path: Path | str | None = None) -> pd.DataFrame:
    """Load the bundled (or supplied) Market Probability Tracker CSV export.

    Returns a long frame with :data:`COLUMNS`. Use :func:`probability_buckets`,
    :func:`directional_probs`, and :func:`rate_path` to pull typed views.
    """
    path = Path(path) if path else _DEFAULT_PATH
    return _read(path)


def _read(buffer) -> pd.DataFrame:
    raw = pd.read_csv(
        buffer,
        header=0,
        dtype=str,
        names=["snapshot_date", "meeting_date", "target_range", "field", "value"],
        skip_blank_lines=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    for col in ("target_range", "field"):
        raw[col] = raw[col].str.strip()

    out = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(raw["snapshot_date"], errors="coerce"),
            "meeting_date": pd.to_datetime(raw["meeting_date"], errors="coerce"),
            "target_range": raw["target_range"],
            "field": raw["field"],
            "value": pd.to_numeric(raw["value"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["snapshot_date", "meeting_date", "field", "value"])
    out = out.sort_values(["snapshot_date", "meeting_date"]).reset_index(drop=True)
    return out[COLUMNS]


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for the most recent ``snapshot_date`` only."""
    if df.empty:
        return df
    return df[df["snapshot_date"] == df["snapshot_date"].max()].copy()


def probability_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bucket probabilities: snapshot_date, meeting_date, low_bps, high_bps, probability.

    Excludes the directional ``Prob: hike`` / ``Prob: cut`` rows (no bps range).
    """
    cols = ["snapshot_date", "meeting_date", "low_bps", "high_bps", "probability"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    sub = df[df["field"].str.startswith("Prob:")].copy()
    ranges = sub["field"].str.extract(_RANGE_RE)
    sub["low_bps"] = pd.to_numeric(ranges[0], errors="coerce")
    sub["high_bps"] = pd.to_numeric(ranges[1], errors="coerce")
    sub = sub.dropna(subset=["low_bps", "high_bps"])
    sub["low_bps"] = sub["low_bps"].astype(int)
    sub["high_bps"] = sub["high_bps"].astype(int)
    sub = sub.rename(columns={"value": "probability"})
    return sub[cols].reset_index(drop=True)


def directional_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Hike / cut probabilities: snapshot_date, meeting_date, prob_hike, prob_cut."""
    cols = ["snapshot_date", "meeting_date", "prob_hike", "prob_cut"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    sub = df[df["field"].isin(["Prob: hike", "Prob: cut"])]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    wide = sub.pivot_table(
        index=["snapshot_date", "meeting_date"], columns="field", values="value", aggfunc="last"
    ).reset_index()
    wide = wide.rename(columns={"Prob: hike": "prob_hike", "Prob: cut": "prob_cut"})
    for c in ("prob_hike", "prob_cut"):
        if c not in wide.columns:
            wide[c] = pd.NA
    return wide[cols]


def rate_path(df: pd.DataFrame, snapshot: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Implied rate distribution per meeting (bps), indexed by ``meeting_date``.

    Columns ``mean``, ``mode``, ``p25``, ``p75`` come straight from the
    published ``Rate:`` fields. Uses the latest snapshot unless ``snapshot`` is
    given.
    """
    cols = ["mean", "mode", "p25", "p75"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    sub = latest_snapshot(df) if snapshot is None else df[df["snapshot_date"] == pd.Timestamp(snapshot)]
    sub = sub[sub["field"].isin(_RATE_FIELDS)].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub["stat"] = sub["field"].map(_RATE_FIELDS)
    wide = sub.pivot_table(index="meeting_date", columns="stat", values="value", aggfunc="last")
    wide = wide.reindex(columns=cols)
    wide.index.name = "meeting_date"
    return wide.sort_index()
