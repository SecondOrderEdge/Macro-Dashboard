"""The printable-HTML export must capture the complete methodology, in light."""

from __future__ import annotations

import streamlit as st

from src.ui.views import methodology


def test_print_html_is_complete_and_light():
    html = methodology._build_print_html(None)

    # A self-contained, light document.
    assert html.startswith("<!doctype html>")
    assert "</html>" in html.strip()[-20:]
    assert "background:#fff" in html

    # Every numbered section is present (1..18) — i.e. nothing got clipped.
    for n in range(1, 19):
        assert f"{n}." in html, f"section {n} missing from export"

    # Spot-check content from across the whole page, including the newest bits.
    for marker in ("Philosophy", "Beveridge", "Excess CAPE Yield", "Breadth", "Reproducibility"):
        assert marker in html, f"expected '{marker}' in printable export"


def test_print_capture_restores_streamlit_functions():
    saved = (st.markdown, st.plotly_chart, st.dataframe, st.info, st.columns)
    methodology._build_print_html(None)
    assert (st.markdown, st.plotly_chart, st.dataframe, st.info, st.columns) == saved
