"""Offline tests for scripts/refresh_market_probability.py payload handling."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refresh_market_probability.py"
_spec = importlib.util.spec_from_file_location("refresh_market_probability", _SCRIPT)
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)

_ROWS = [
    ["2023-03-29", "2023-06-21", "475bps - 500bps", "Rate: mean", 481.98],
    ["2023-03-29", "2023-06-21", "475bps - 500bps", "Prob: 475bps - 500bps", 28.22],
]


def test_html_soft_404_is_rejected_not_parsed_as_csv():
    page = b"\n\n<!DOCTYPE html>\n<html lang='en'><head><title>404 Page - Not Found</title>"
    assert refresh._to_csv_text(page) is None


def test_plain_csv_passes_through():
    csv = "date,reference_start_date,target_range,field,value\n2023-03-29,2023-06-21,x,Rate: mean,1\n"
    assert refresh._to_csv_text(csv.encode()) == csv


def test_xlsx_data_sheet_is_converted_to_long_csv():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"note": ["license text"]}).to_excel(xw, sheet_name="LICENSE", index=False)
        pd.DataFrame(_ROWS, columns=refresh._EXPECTED_COLUMNS).to_excel(xw, sheet_name="DATA", index=False)
    raw = buf.getvalue()
    assert refresh._is_xlsx(raw)
    text = refresh._to_csv_text(raw)
    assert text.splitlines()[0] == ",".join(refresh._EXPECTED_COLUMNS)
    assert "Prob: 475bps - 500bps" in text


def test_discover_url_from_tracker_page():
    page = (
        '<a class="btn" href="/-/media/Project/Atlanta/FRBA/Documents/research-and-data/'
        'data/market-probability-tracker/mpt_histdata.xlsx">MPT Historical Data</a>'
    )
    assert refresh._discover_url(page) == (
        "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/research-and-data/"
        "data/market-probability-tracker/mpt_histdata.xlsx"
    )
    assert refresh._discover_url("<html>no link</html>") is None
