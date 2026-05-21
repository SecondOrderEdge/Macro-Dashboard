"""Refresh data/cape.csv from Robert Shiller's published spreadsheet.

Downloads Shiller's Excel (the canonical, monthly-updated CAPE source), parses
the CAPE column with the same parser the app uses, validates it, and writes
``data/cape.csv`` (``date,cape``). Run by
``.github/workflows/refresh-cape.yml`` monthly; the workflow commits only when
the data changed. The in-app live fetch remains as a fallback.

Tries several known Shiller URLs (override/add one via the ``CAPE_DATA_URL``
env / Actions variable). Picks the source with the most recent month and never
regresses to an older latest month than the committed file.

Exit codes:
    0  success — file written or already current
    1  no source returned a usable, fresh-enough CAPE series
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _ROOT / "data" / "cape.csv"

# Shiller moved off econ.yale.edu to shillerdata.com, but that WordPress site
# bot-blocks CI runners. posix4e/shiller_wrapper_data republishes the *same*
# ie_data.xls to GitHub Pages weekly, which runners can reach — so it's the
# practical primary in CI. An override can be supplied via CAPE_DATA_URL.
_URLS = [
    os.environ.get("CAPE_DATA_URL", "").strip(),
    "https://posix4e.github.io/shiller_wrapper_data/ie_data.xls",
    "https://shillerdata.com/wp-content/uploads/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xlsx",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/vnd.ms-excel,application/octet-stream,*/*",
}


def _existing_latest():
    if not _TARGET.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_csv(_TARGET, parse_dates=["date"])
        return df["date"].max() if not df.empty else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    sys.path.insert(0, str(_ROOT))
    import pandas as pd

    from src.data.cape import _parse_shiller_excel

    best = None
    for url in [u for u in _URLS if u]:
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
            s = _parse_shiller_excel(df)
            if not s.empty:
                print(f"  {url} via {engine}: {len(s)} months, latest {s.index[-1].date()} = {float(s.iloc[-1]):.2f}")
                if best is None or s.index[-1] > best.index[-1]:
                    best = s
                break

    if best is None or best.empty:
        print("ERROR: no Shiller source returned a parseable CAPE series.")
        return 1

    latest_val = float(best.iloc[-1])
    if not (3.0 < latest_val < 80.0):
        print(f"ERROR: latest CAPE {latest_val} is outside the sane range (3–80).")
        return 1

    prior = _existing_latest()
    if prior is not None and best.index[-1] < prior:
        print(f"Source latest {best.index[-1].date()} is older than committed {prior.date()} — keeping current file.")
        return 0

    out = best.rename("cape")
    out.index.name = "date"
    out.to_csv(_TARGET)
    print(f"Wrote {_TARGET.relative_to(_ROOT)} — {len(out)} months, latest {best.index[-1].date()} = {latest_val:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
