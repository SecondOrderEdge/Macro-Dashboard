"""Refresh the bundled Atlanta Fed Market Probability Tracker export.

Downloads the tracker data from the URL in the ``MPT_DATA_URL`` environment
variable, validates that it parses into the expected shape via the project's own
parser, and writes it to ``data/market_probability_tracker.csv``. Designed to be
run by ``.github/workflows/refresh-market-probability.yml`` on a schedule; the
workflow commits the file only if it actually changed.

Exit codes:
    0  success — file written, or URL not configured (skip), so a missing
       variable never fails the scheduled job
    1  the URL *was* configured but the download or validation failed
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _ROOT / "data" / "market_probability_tracker.csv"
_URL = os.environ.get("MPT_DATA_URL", "").strip()

# The published "MPT Historical Data" download is an .xlsx; the in-repo file is
# the long CSV the parser expects. These are the columns we serialise to.
_EXPECTED_COLUMNS = ["date", "reference_start_date", "target_range", "field", "value"]

# A real browser UA — the Atlanta Fed front end rejects obvious bots.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _to_csv_text(raw: bytes) -> str | None:
    """Coerce a download into long-CSV text, handling the .xlsx export.

    The Atlanta Fed publishes "MPT Historical Data" as an .xlsx; we read the
    sheet whose columns match the long schema and serialise it to CSV. A plain
    CSV/TSV download is decoded as-is. Returns None (with a diagnostic) if the
    payload is neither — e.g. an HTML block page.
    """
    if raw[:4] == b"PK\x03\x04":  # ZIP magic — .xlsx is a zip container
        import io

        import pandas as pd

        try:
            xls = pd.ExcelFile(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: could not open .xlsx download: {exc}")
            return None
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            lower = {str(c).strip().lower(): c for c in df.columns}
            if all(col in lower for col in _EXPECTED_COLUMNS):
                df = df[[lower[col] for col in _EXPECTED_COLUMNS]]
                df.columns = _EXPECTED_COLUMNS
                return df.to_csv(index=False)
        print(f"ERROR: no .xlsx sheet has the expected columns {_EXPECTED_COLUMNS}.")
        for sheet in xls.sheet_names:
            cols = list(xls.parse(sheet, nrows=0).columns)
            print(f"  sheet {sheet!r} columns: {cols}")
        return None

    return raw.decode("utf-8-sig", errors="replace")


def main() -> int:
    if not _URL:
        print(
            "MPT_DATA_URL not set — skipping refresh. Set the repository Actions "
            "variable MPT_DATA_URL to the tracker's data-download URL to enable."
        )
        return 0

    try:
        req = urllib.request.Request(_URL, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001 - any network/HTTP failure
        print(f"ERROR: download from MPT_DATA_URL failed: {exc}")
        return 1

    text = _to_csv_text(raw)
    if text is None:
        return 1
    # Normalise newlines so identical data doesn't churn the file on CRLF diffs.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"

    # Validate via the project's own parser — single source of truth for "valid".
    sys.path.insert(0, str(_ROOT))
    from src.data.market_probability import parse_market_probability, probability_buckets

    df = parse_market_probability(text)
    n_snaps = int(df["snapshot_date"].nunique()) if not df.empty else 0
    if df.empty or n_snaps < 10:
        head = text[:200].replace("\n", " | ")
        print(
            "ERROR: download did not parse into the expected Market Probability "
            f"Tracker shape (rows={len(df)}, snapshots={n_snaps}). The URL may be "
            "returning HTML/a block page or a different format. First bytes: " + head
        )
        return 1

    # Sanity: at least one (snapshot, meeting) group's buckets should sum near 100.
    buckets = probability_buckets(df)
    if not buckets.empty:
        sums = buckets.groupby(["snapshot_date", "meeting_date"])["probability"].sum()
        if not ((sums > 95) & (sums < 105)).any():
            print("ERROR: parsed data has no bucket group summing ~100% — likely malformed.")
            return 1

    _TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {_TARGET.relative_to(_ROOT)} — {len(df):,} rows, {n_snaps} snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
