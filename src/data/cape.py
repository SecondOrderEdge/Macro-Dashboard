"""CAPE ratio (Shiller cyclically-adjusted P/E) from Shiller's published dataset.

CAPE is intentionally NOT an input to the recession ensemble — empirically
it doesn't help short-horizon recession prediction (it ran hot for most of
2014-2024 without a recession arriving). It's exposed here as a *valuation*
context indicator: at extreme readings, the market downside conditional on
a recession arriving is materially larger than at average readings.

Data path:

1. **Primary**: GitHub-hosted CSV mirror at
   ``raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv``.
   This is the ``datahub.io/core/s-and-p-500`` repo, which curates Shiller's
   historical series as a flat CSV. The ``PE10`` column is the CAPE ratio.
   We use it as the primary source because GitHub Raw is reliable and
   doesn't require any binary spreadsheet parser.

2. **Fallback**: Robert Shiller's original Excel at Yale. Requires
   ``openpyxl`` (for .xlsx) or ``xlrd<2.0`` (for .xls); served at
   ``econ.yale.edu/~shiller/data/`` and at ``shillerdata.com``.

Either source returns the same monthly CAPE series. Any failure returns an
empty Series so the dashboard degrades gracefully rather than crashing.

The same bundled file carries Shiller's real total-return price index. From
that index we build the subsequent 10-year annualized real total return and
an OLS map from the CAPE earnings yield (1/CAPE) to that return. The map is
display-only valuation context: it is not an input to the recession ensemble,
the composite, or LAME.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_BUNDLED_PATH = Path(__file__).resolve().parents[2] / "data" / "cape.csv"


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

_GITHUB_CSV = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv"
)

_YALE_URLS = (
    "https://shillerdata.com/wp-content/uploads/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xlsx",
)


def _cache_data(*args: Any, **kwargs: Any):
    try:
        import streamlit as st

        return st.cache_data(*args, **kwargs)
    except Exception:
        def _passthrough(fn):
            return fn

        return _passthrough


@_cache_data(ttl=604800, show_spinner=False)  # 7 days
def fetch_cape_history() -> pd.Series:
    """Return a monthly-indexed CAPE Series, or an empty one on total failure.

    Sources are overlaid in increasing order of freshness so the most recent
    available value wins for each month:

    1. **datahub GitHub CSV** — long history, but the mirror is unmaintained
       (it has lagged by years), so it only fills old gaps.
    2. **Bundled ``data/cape.csv``** — refreshed monthly by the refresh-cape
       GitHub Action; this is the reliable, current base in deployment.
    3. **Shiller's Excel (live)** — freshest if reachable, but the source blocks
       bots in many environments, so it's best-effort on top.

    Errors during fetching/parsing are recorded on the returned Series via
    ``attrs['fetch_log']`` so the UI can surface them.
    """
    log: list[str] = []
    github_series = _try_fetch_github(log)
    bundled_series = _load_bundled(log)
    yale_series = _try_fetch_yale(log)

    merged = _merge([github_series, bundled_series, yale_series])
    if merged.empty:
        return _empty(log)
    merged.attrs["fetch_log"] = log
    return merged


@_cache_data(ttl=604800, show_spinner=False)  # 7 days
def fetch_cape_extras() -> pd.DataFrame:
    """Total-return CAPE and Excess CAPE Yield, from the bundled ``cape.csv``.

    These two series come only from Shiller's workbook (no live mirror provides
    them), so they're read straight from ``data/cape.csv``. ``ecy`` is the
    Excess CAPE Yield as a fraction (e.g. 0.0139 = 1.39%). Returns an empty
    frame if those columns are absent (older CSV without the enrichment).
    """
    cols = ["tr_cape", "ecy"]
    try:
        if not _BUNDLED_PATH.exists():
            return pd.DataFrame(columns=cols)
        df = pd.read_csv(_BUNDLED_PATH)
        if "date" not in df.columns:
            return pd.DataFrame(columns=cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        out = pd.DataFrame(index=df.index)
        for c in cols:
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
        return out
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)


def _merge(series_list: list[pd.Series]) -> pd.Series:
    """Overlay CAPE series in order; later (fresher) sources win per month."""
    out = pd.Series(dtype=float, name="cape")
    sources: list[str] = []
    for s in series_list:
        if s is None or s.empty:
            continue
        for ts, val in s.items():
            out.loc[ts] = val
        sources.append(s.attrs.get("source", "?"))
    out = out.sort_index()
    out.attrs["source"] = " + ".join(sources) if sources else "none"
    return out


def _load_bundled(log: list[str]) -> pd.Series:
    """Read the committed ``data/cape.csv`` (date,cape), refreshed by the Action."""
    try:
        if not _BUNDLED_PATH.exists():
            return pd.Series(dtype=float, name="cape")
        df = pd.read_csv(_BUNDLED_PATH)
        date_col = "date" if "date" in df.columns else df.columns[0]
        val_col = "cape" if "cape" in df.columns else df.columns[-1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col, val_col])
        s = pd.Series(df[val_col].astype(float).values, index=df[date_col], name="cape").sort_index()
        s.attrs["source"] = "bundled:data/cape.csv"
        return s
    except Exception as exc:  # noqa: BLE001
        log.append(f"bundled cape.csv: {type(exc).__name__}: {exc}")
        return pd.Series(dtype=float, name="cape")


def _try_fetch_yale(log: list[str]) -> pd.Series:
    try:
        import requests
    except ImportError:
        log.append("requests not installed")
        return pd.Series(dtype=float, name="cape")

    for url in _YALE_URLS:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30)
        except Exception as exc:
            log.append(f"yale {url}: {type(exc).__name__}: {exc}")
            continue
        if r.status_code != 200:
            log.append(f"yale {url}: HTTP {r.status_code}")
            continue
        for engine in ("openpyxl", "xlrd"):
            try:
                df = pd.read_excel(
                    io.BytesIO(r.content),
                    sheet_name="Data",
                    skiprows=7,
                    engine=engine,
                )
            except Exception as exc:
                log.append(f"yale {url} via {engine}: {type(exc).__name__}: {exc}")
                continue
            s = _parse_shiller_excel(df)
            if not s.empty:
                s.attrs["source"] = f"yale-excel ({engine})"
                return s
            log.append(f"yale {url} via {engine}: parsed but empty")
    return pd.Series(dtype=float, name="cape")


def _try_fetch_github(log: list[str]) -> pd.Series:
    try:
        return _fetch_github_csv(log)
    except Exception as exc:
        log.append(f"github csv: {type(exc).__name__}: {exc}")
        return pd.Series(dtype=float, name="cape")


def _empty(log: list[str]) -> pd.Series:
    s = pd.Series(dtype=float, name="cape")
    s.attrs["fetch_log"] = log
    return s


# ---------------------------------------------------------------- GitHub CSV


def _fetch_github_csv(log: list[str]) -> pd.Series:
    import requests

    r = requests.get(_GITHUB_CSV, headers=_HEADERS, timeout=30)
    if r.status_code != 200:
        log.append(f"{_GITHUB_CSV}: HTTP {r.status_code}")
        return pd.Series(dtype=float, name="cape")

    df = pd.read_csv(io.BytesIO(r.content))
    if "Date" not in df.columns or "PE10" not in df.columns:
        log.append(f"github csv: missing columns; got {list(df.columns)[:6]}")
        return pd.Series(dtype=float, name="cape")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "PE10"])
    df["PE10"] = pd.to_numeric(df["PE10"], errors="coerce")
    df = df.dropna(subset=["PE10"])
    # PE10 == 0 is the missing-data placeholder used for the first 10 years
    # of the sample (before there's enough history to compute it) and
    # sometimes for the most-recent months that haven't accumulated yet.
    df = df[df["PE10"] > 0]
    s = pd.Series(df["PE10"].astype(float).values, index=df["Date"], name="cape")
    s = s.sort_index()
    s.attrs["source"] = "github:datasets/s-and-p-500"
    return s


# ---------------------------------------------------------------- Yale Excel


def _parse_shiller_excel(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, name="cape")
    date_col = df.columns[0]
    cape_col = None
    for c in df.columns:
        cu = str(c).upper().replace(" ", "")
        if "CAPE" in cu or "P/E10" in cu or "PE10" in cu or "CYCLICALLYADJUSTED" in cu:
            cape_col = c
            break
    if cape_col is None:
        return pd.Series(dtype=float, name="cape")

    df = df[[date_col, cape_col]].copy()
    df = df.dropna(subset=[date_col, cape_col])
    df["date"] = df[date_col].apply(_yale_date_to_timestamp)
    df = df.dropna(subset=["date"])
    df[cape_col] = pd.to_numeric(df[cape_col], errors="coerce")
    df = df.dropna(subset=[cape_col])
    s = pd.Series(df[cape_col].astype(float).values, index=df["date"], name="cape")
    return s.sort_index()


def _yale_date_to_timestamp(v) -> pd.Timestamp | None:
    """Yale's date column: float like ``2024.04`` (Apr 2024) or ``2024.1``
    (Oct 2024 — digits after the decimal are positional, not numeric)."""
    if pd.isna(v):
        return None
    try:
        s = f"{float(v):.2f}"
    except Exception:
        return None
    if "." not in s:
        return None
    year_str, frac = s.split(".")
    try:
        year = int(year_str)
        month = int(frac.ljust(2, "0")[:2])
    except Exception:
        return None
    if month < 1 or month > 12 or year < 1871 or year > 2100:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


# ---------------------------------------------------------------- summary


def cape_summary(cape: pd.Series, modern_start: str = "1950-01-01") -> dict:
    """Current value, percentile rank, median, recent trend, classic peaks."""
    if cape is None or cape.empty:
        return {}
    cape = cape.dropna().sort_index()
    today = float(cape.iloc[-1])
    as_of = cape.index[-1]

    modern = cape.loc[cape.index >= pd.Timestamp(modern_start)]
    if modern.empty:
        modern = cape
    pct = float((modern <= today).mean() * 100.0)
    median = float(modern.median())

    one_yr_ago_idx = as_of - pd.DateOffset(months=12)
    s_to_year_ago = cape.loc[cape.index <= one_yr_ago_idx]
    yr_ago = float(s_to_year_ago.iloc[-1]) if not s_to_year_ago.empty else float("nan")

    def _peak(start: str, end: str) -> float:
        window = cape.loc[start:end]
        return float(window.max()) if not window.empty else float("nan")

    peaks = {
        "1929 peak": _peak("1929-01-01", "1929-12-31"),
        "2000 dot-com peak": _peak("1999-01-01", "2000-12-31"),
        "2007 peak": _peak("2007-01-01", "2007-12-31"),
    }

    return {
        "as_of": as_of,
        "today": today,
        "modern_percentile": pct,
        "modern_median": median,
        "one_year_ago": yr_ago,
        "modern_start": pd.Timestamp(modern_start),
        "peaks": peaks,
        "source": cape.attrs.get("source", "unknown"),
    }


def cape_band(percentile: float) -> tuple[str, str]:
    """Map a percentile rank to a label + severity bucket."""
    if percentile < 25:
        return "CHEAP", "low"
    if percentile < 60:
        return "AVERAGE", "elevated"
    if percentile < 85:
        return "EXPENSIVE", "high"
    return "EXTREME", "critical"


def ecy_band(percentile: float) -> tuple[str, str]:
    """Band for the Excess CAPE Yield. Inverted vs CAPE: a *high* ECY (high
    percentile) is attractive (equities offer a wide cushion over real bond
    yields); a *low* ECY is rich."""
    if percentile >= 75:
        return "ATTRACTIVE", "low"
    if percentile >= 40:
        return "NEUTRAL", "elevated"
    if percentile >= 15:
        return "RICH", "high"
    return "VERY RICH", "critical"


def cape_extras_summary(extras: pd.DataFrame, modern_start: str = "1950-01-01") -> dict:
    """Latest value, percentile, and median for tr_cape and ecy (post-1950)."""
    if extras is None or extras.empty:
        return {}
    out: dict[str, dict] = {}
    for col in ("tr_cape", "ecy"):
        s = extras[col].dropna() if col in extras.columns else pd.Series(dtype=float)
        if s.empty:
            out[col] = {}
            continue
        today = float(s.iloc[-1])
        modern = s.loc[s.index >= pd.Timestamp(modern_start)]
        if modern.empty:
            modern = s
        out[col] = {
            "today": today,
            "as_of": s.index[-1],
            "modern_percentile": float((modern <= today).mean() * 100.0),
            "modern_median": float(modern.median()),
        }
    return out


def _parse_shiller_full(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the Shiller 'Data' sheet into date, cape, tr_cape, ecy, real_tr_price.

    Column labels in the workbook: ``CAPE``, ``TR CAPE``, ``Yield`` (the
    Excess CAPE Yield, stored as a fraction), and the real total-return price
    index. The first column is the Yale fractional date. Missing columns come
    back as NaN.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "cape", "tr_cape", "ecy", "real_tr_price"])
    date_col = df.columns[0]
    out = pd.DataFrame()
    out["date"] = df[date_col].apply(_yale_date_to_timestamp)

    def _num(name: str) -> pd.Series:
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series([float("nan")] * len(df))

    out["cape"] = _num("CAPE")
    out["tr_cape"] = _num("TR CAPE")
    ecy = _num("Yield")
    # Guard: ECY is a small fraction; if 'Yield' isn't that column, drop it.
    if ecy.notna().any() and ecy.abs().median() > 0.5:
        ecy = pd.Series([float("nan")] * len(df))
    out["ecy"] = ecy
    # Positional: the helper returns one value per source row, same order.
    out["real_tr_price"] = _real_total_return_price(df).to_numpy()

    out = out.dropna(subset=["date", "cape"]).sort_values("date").reset_index(drop=True)
    return out


def _real_total_return_price(df: pd.DataFrame) -> pd.Series:
    """Shiller's real total-return price index.

    The workbook header spells this "Real Total Return Price", but the single
    header row the refresh keeps (``skiprows=7``) labels both the real price
    and the real total-return price as ``Price``. pandas disambiguates them
    as ``Price`` and ``Price.1``; the second is the total-return index
    (column 9 on the Data sheet).
    """
    nan = pd.Series([float("nan")] * len(df), index=df.index)
    for c in df.columns:
        key = re.sub(r"[^A-Z0-9]", "", str(c).upper())
        if "TOTALRETURN" in key and "PRICE" in key:
            return pd.to_numeric(df[c], errors="coerce")
    price_cols = [
        c for c in df.columns
        if re.sub(r"\.\d+$", "", str(c)).strip().upper() == "PRICE"
    ]
    if len(price_cols) >= 2:
        return pd.to_numeric(df[price_cols[1]], errors="coerce")
    return nan


# ------------------------------------------------- 10y real return from CAPE


# Ten years of monthly observations. The annualized return is
# (P_{t+120} / P_t) ** (1/10) - 1, i.e. exponent 12/120.
_RETURN_HORIZON_MONTHS = 120
# A line needs a real cloud, not a handful of overlapping months.
_MIN_FIT_OBS = 60


def subsequent_10y_real_return(
    real_tr_price: pd.Series,
    horizon_months: int = _RETURN_HORIZON_MONTHS,
) -> pd.Series:
    """Annualized real total return over the next ``horizon_months`` months.

    ``(P_{t+h} / P_t) ** (12/h) - 1``. For the 10-year window (h = 120) this
    is the compound annual growth of Shiller's real total-return price index,
    and it reproduces his published "10 Year Annualized Stock Real Return"
    column. A month is left blank when there is no price exactly ``h`` months
    later (the window has not elapsed, or the monthly calendar has a gap).
    """
    price = pd.to_numeric(real_tr_price, errors="coerce").sort_index()
    price = price[~price.index.duplicated(keep="last")]
    out = pd.Series(np.nan, index=price.index, dtype=float, name="fwd_real_return_10y")
    if horizon_months < 1 or len(price) <= horizon_months:
        return out

    future = price.shift(-horizon_months)
    future_dates = pd.Series(price.index, index=price.index).shift(-horizon_months)
    expected = price.index + pd.DateOffset(months=int(horizon_months))
    delta = pd.to_datetime(future_dates.to_numpy()) - pd.to_datetime(expected.to_numpy())
    on_calendar = pd.Series(np.abs(delta) <= pd.Timedelta(days=5), index=price.index).fillna(False)

    ratio = future / price
    ann = ratio ** (12.0 / float(horizon_months)) - 1.0
    ok = on_calendar & (price > 0) & (future > 0) & np.isfinite(ann)
    mask = ok.fillna(False).to_numpy(dtype=bool)
    out.iloc[mask] = ann.to_numpy(dtype=float)[mask]
    return out


def load_cape_return_history(path: Path | None = None) -> pd.DataFrame:
    """CAPE, real total-return price, and the subsequent 10-year real return.

    Reads the bundled ``data/cape.csv`` (or ``path``). The forward return is
    computed here from ``real_tr_price``; it is not a stored lookup table.
    Returns an empty frame if the price column is absent, so an older CSV
    still leaves the rest of the CAPE panel usable.
    """
    cols = ["cape", "real_tr_price", "fwd_real_return_10y"]
    path = path or _BUNDLED_PATH
    try:
        if not path.exists():
            return pd.DataFrame(columns=cols)
        df = pd.read_csv(path)
        if "date" not in df.columns:
            return pd.DataFrame(columns=cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
        out = pd.DataFrame(index=df.index)
        out["cape"] = (
            pd.to_numeric(df["cape"], errors="coerce") if "cape" in df.columns else np.nan
        )
        if "real_tr_price" in df.columns:
            out["real_tr_price"] = pd.to_numeric(df["real_tr_price"], errors="coerce")
        else:
            out["real_tr_price"] = np.nan
        out["fwd_real_return_10y"] = subsequent_10y_real_return(out["real_tr_price"])
        return out
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)


def fit_cape_expected_return(
    cape: pd.Series,
    fwd_real_return: pd.Series,
    min_obs: int = _MIN_FIT_OBS,
) -> dict:
    """OLS of the subsequent 10-year real total return on the CAPE earnings yield.

    Specification::

        r_{t → t+10} = α + β · (1 / CAPE_t) + ε_t

    Why 1/CAPE rather than CAPE. Campbell and Shiller's present-value identity
    (Campbell & Shiller 1988; Campbell & Shiller 1998, "Valuation Ratios and
    the Long-Run Stock Market Outlook") writes a valuation *yield* as the
    discounted sum of expected future returns and earnings growth. Holding
    expected real earnings growth roughly constant, long-horizon real returns
    are approximately linear in the earnings yield E10/P = 1/CAPE, not in the
    multiple. That is also the yield inside Shiller's Excess CAPE Yield
    (1/CAPE minus the real bond yield). Their published regressions often use
    log(P/E), which traces nearly the same curve; a straight line in CAPE
    itself is the worse-documented alternative and runs off toward large
    negative returns as the multiple rises. On the full Shiller history the
    earnings-yield fit matches the cloud at least as well as a line in CAPE.

    Overlapping ten-year windows are not independent observations, so R²
    overstates how precise the relationship is. The result is a historical
    statistical description for the valuation panel, not a forecast, not
    investment advice, and not an input to the recession models.

    Returns ``{}`` when fewer than ``min_obs`` paired months are available.
    """
    paired = pd.DataFrame({"cape": cape, "r": fwd_real_return}).dropna()
    paired = paired[np.isfinite(paired["cape"]) & (paired["cape"] > 0)]
    paired = paired[np.isfinite(paired["r"])]
    if len(paired) < min_obs:
        return {}

    earnings_yield = 1.0 / paired["cape"].to_numpy(dtype=float)
    y = paired["r"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(y)), earnings_yield])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    fitted = design @ np.asarray(coef, dtype=float)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha": alpha,
        "beta": beta,
        "r2": float(r2),
        "n": int(len(paired)),
        "predictor": "inverse_cape",
        "sample_start": paired.index.min(),
        "sample_end": paired.index.max(),
        "realized_median": float(np.median(y)),
    }


def implied_10y_real_return(cape_today: float, fit: dict) -> float:
    """Plug a CAPE reading into a fit from :func:`fit_cape_expected_return`."""
    if not fit or cape_today is None or not np.isfinite(cape_today) or float(cape_today) <= 0:
        return float("nan")
    return float(fit["alpha"] + fit["beta"] / float(cape_today))


def implied_return_band(ann_return: float) -> tuple[str, str]:
    """Display band for an implied 10-year annualized real return.

    Anchored to round levels around the long-run average subsequent real
    return in the Shiller sample (about 7% annualized). A low implied return
    is the expensive-valuation reading. Labels are for the chart, not a
    recommendation.
    """
    if ann_return is None or not np.isfinite(ann_return):
        return "—", "elevated"
    if ann_return >= 0.07:
        return "ABOVE AVG", "low"
    if ann_return >= 0.04:
        return "BELOW AVG", "elevated"
    if ann_return >= 0.02:
        return "LOW", "high"
    return "VERY LOW", "critical"
