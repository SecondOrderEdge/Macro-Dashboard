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

import html
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _ROOT / "data" / "market_probability_tracker.csv"

# The published "MPT Historical Data" download (an .xlsx). Override via the
# MPT_DATA_URL env / Actions variable if the Atlanta Fed ever relocates it.
# (Moved from .../Documents/cenfis/... in Sep 2026; the old path now returns an
# HTTP 200 HTML "404 Page - Not Found" page.)
_DEFAULT_URL = (
    "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/"
    "research-and-data/data/market-probability-tracker/mpt_histdata.xlsx"
)
_URL = os.environ.get("MPT_DATA_URL", "").strip() or _DEFAULT_URL

# Tracker landing page — scraped for the current .xlsx link if the URL above
# stops serving a spreadsheet (the Atlanta Fed has relocated it before).
_PAGE_URL = "https://www.atlantafed.org/research-and-data/data/market-probability-tracker"
_LINK_RE = re.compile(r"""href=["']([^"']*mpt_histdata\.xlsx[^"']*)["']""", re.IGNORECASE)

# The published "MPT Historical Data" download is an .xlsx; the in-repo file is
# the long CSV the parser expects. These are the columns we serialise to.
_EXPECTED_COLUMNS = ["date", "reference_start_date", "target_range", "field", "value"]

# Browser-like headers — the Atlanta Fed front end rejects obvious bots.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
        "application/octet-stream,*/*"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.atlantafed.org/research-and-data/data/market-probability-tracker",
}


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

        n = len(_EXPECTED_COLUMNS)
        # The export has LICENSE / DICTIONARY / DATA sheets; pick the data sheet
        # (named 'data' if present, else the widest non-empty one). The downstream
        # parser reads columns positionally, so a differently-labelled DATA sheet
        # still works as long as the column order matches — but if the labels do
        # match we reorder by name to be safe. The parser's validation (snapshot
        # count, bucket sums) is the backstop against a wrong-order sheet.
        for sheet in sorted(xls.sheet_names, key=lambda s: str(s).strip().lower() != "data"):
            df = xls.parse(sheet).dropna(axis=1, how="all").dropna(axis=0, how="all")
            if df.shape[0] == 0 or df.shape[1] < n:
                continue
            lower = {str(c).strip().lower(): c for c in df.columns}
            if all(col in lower for col in _EXPECTED_COLUMNS):
                df = df[[lower[col] for col in _EXPECTED_COLUMNS]]
            else:
                df = df.iloc[:, :n]  # assume canonical order; parser validates
            df.columns = _EXPECTED_COLUMNS
            print(f"Using sheet {sheet!r} ({len(df):,} rows).")
            return df.to_csv(index=False)

        print(f"ERROR: no usable data sheet in {xls.sheet_names}.")
        for sheet in xls.sheet_names:
            try:
                cols = list(xls.parse(sheet, nrows=0).columns)
            except Exception:  # noqa: BLE001
                cols = ["<unreadable>"]
            print(f"  sheet {sheet!r} columns: {cols}")
        return None

    text = raw.decode("utf-8-sig", errors="replace")
    if _looks_like_html(text):
        print(
            "ERROR: download is an HTML page, not the .xlsx/CSV export (the file may "
            "have moved, or a bot-block page was served)."
        )
        return None
    return text


def _is_xlsx(raw: bytes) -> bool:
    return raw[:4] == b"PK\x03\x04"


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:512].lower()
    return head.startswith("<") or "<html" in head or "<!doctype" in head


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _discover_url(page_html: str) -> str | None:
    """Return the absolute mpt_histdata.xlsx link from the tracker page, if any."""
    match = _LINK_RE.search(page_html)
    if not match:
        return None
    return urllib.parse.urljoin(_PAGE_URL, html.unescape(match.group(1)))


def main() -> int:
    if not _URL:  # only if both the variable and the baked-in default are blank
        print("No download URL configured — skipping refresh.")
        return 0

    print(f"Downloading Market Probability Tracker data from {_URL}")
    try:
        raw = _fetch(_URL)
    except Exception as exc:  # noqa: BLE001 - any network/HTTP failure
        print(f"ERROR: download from {_URL} failed: {exc}")
        return 1

    if not _is_xlsx(raw) and _looks_like_html(raw[:2048].decode("utf-8", errors="replace")):
        # The URL served an HTML page (e.g. a soft 404 after a relocation).
        # Look up the current download link on the tracker page and retry once.
        print(f"URL returned HTML, not a spreadsheet; looking up the link on {_PAGE_URL}")
        try:
            found = _discover_url(_fetch(_PAGE_URL).decode("utf-8", errors="replace"))
            if found and found != _URL:
                print(f"Downloading Market Probability Tracker data from {found}")
                raw = _fetch(found)
            else:
                print("No different mpt_histdata.xlsx link found on the tracker page.")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: link discovery failed: {exc}")

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
