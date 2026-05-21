"""Refresh data/cape.csv from Robert Shiller's published spreadsheet.

Downloads Shiller's Excel (the canonical, monthly-updated source), parses CAPE,
TR CAPE, and the Excess CAPE Yield with the same parser the app uses, validates
it, and writes ``data/cape.csv`` (``date,cape,tr_cape,ecy``). Run by
``.github/workflows/refresh-cape.yml`` monthly; the workflow commits only when
the data changed. The in-app live fetch remains as a fallback.

Sources: the current ie_data.xls link is scraped from shillerdata.com (its
WordPress upload path rotates), with econ.yale.edu and a GitHub-Pages mirror as
fallbacks. Override/add a URL via ``CAPE_DATA_URL``. Picks the source with the
most recent month and never regresses to an older latest month than the
committed file.

Exit codes:
    0  success — file written or already current
    1  no source returned a usable, fresh-enough CAPE series
"""

from __future__ import annotations

import io
import os
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _ROOT / "data" / "cape.csv"

_URLS = [
    os.environ.get("CAPE_DATA_URL", "").strip(),
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
    "https://posix4e.github.io/shiller_wrapper_data/ie_data.xls",
    "https://shillerdata.com/wp-content/uploads/ie_data.xls",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/vnd.ms-excel,application/octet-stream,*/*",
}


def _shillerdata_url() -> str | None:
    """Scrape shillerdata.com for the current ie_data.xls link (path rotates)."""
    try:
        req = urllib.request.Request("https://shillerdata.com/", headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  shillerdata.com scrape: {type(exc).__name__}: {exc}")
        return None
    m = re.search(r'href="([^"]*ie_data\.xls[^"]*)"', html)
    if not m:
        print("  shillerdata.com: ie_data.xls link not found on homepage")
        return None
    href = m.group(1)
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://shillerdata.com" + href
    return href


def _existing_latest():
    if not _TARGET.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_csv(_TARGET, parse_dates=["date"])
        return df["date"].max() if not df.empty else None
    except Exception:  # noqa: BLE001
        return None


def _shillerdata_url() -> str | None:
    """Scrape shillerdata.com for the current ie_data.xls link.

    The file lives under a rotating WordPress upload path, so a hardcoded URL
    goes stale (404s). The homepage always links the latest file; find it.
    """
    import re

    try:
        req = urllib.request.Request("https://shillerdata.com/", headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  shillerdata.com scrape: {type(exc).__name__}: {exc}")
        return None
    m = re.search(r'href="([^"]*ie_data\.xls[^"]*)"', html)
    if not m:
        print("  shillerdata.com: ie_data.xls link not found on homepage")
        return None
    href = m.group(1)
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://shillerdata.com" + href
    return href


def main() -> int:
    sys.path.insert(0, str(_ROOT))
    import pandas as pd

    from src.data.cape import _parse_shiller_full

    scraped = _shillerdata_url()
    urls: list[str] = []
    for u in ([scraped] if scraped else []) + _URLS:
        if u and u not in urls:
            urls.append(u)

    # Try the freshly-scraped shillerdata.com link first, then the static list.
    scraped = _shillerdata_url()
    urls: list[str] = []
    for u in ([scraped] if scraped else []) + _URLS:
        if u and u not in urls:
            urls.append(u)

    best = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except Exception as exc:  # noqa: BLE001
            print(f"  {url}: {type(exc).__name__}: {exc}")
            continue
        for engine in ("openpyxl", "xlrd"):
            try:
                df = pd.read_excel(io.BytesIO(raw), sheet_name="Data", skiprows=7, engine=engine)
            except Exception as exc:  # noqa: BLE001
                print(f"  {url} via {engine}: {type(exc).__name__}: {exc}")
                continue
            full = _parse_shiller_full(df)
            if not full.empty:
                latest = full["date"].max()
                print(f"  {url} via {engine}: {len(full)} months, latest {latest.date()} = {full['cape'].iloc[-1]:.2f}")
                if best is None or latest > best["date"].max():
                    best = full
                break

    if best is None or best.empty:
        print("ERROR: no Shiller source returned a parseable CAPE series.")
        return 1

    latest_date = best["date"].max()
    latest_val = float(best["cape"].iloc[-1])
    if not (3.0 < latest_val < 80.0):
        print(f"ERROR: latest CAPE {latest_val} is outside the sane range (3–80).")
        return 1

    prior = _existing_latest()
    if prior is not None and latest_date < prior:
        print(f"Source latest {latest_date.date()} is older than committed {prior.date()} — keeping current file.")
        return 0

    best.to_csv(_TARGET, index=False)
    print(f"Wrote {_TARGET.relative_to(_ROOT)} — {len(best)} months, latest {latest_date.date()} = {latest_val:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
