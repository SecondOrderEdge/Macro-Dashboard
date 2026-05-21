"""ALFRED revision summary: first-release vs final-revised comparison."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.revisions import revision_summary


def _series(values, start="2020-01-01"):
    idx = pd.date_range(start=start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float)


def test_revision_summary_computes_revision():
    first = _series([100.0, 100.0, 100.0, 100.0])
    latest = _series([101.0, 99.0, 100.0, 97.0])  # +1, -1, 0, -3
    out = revision_summary(first, latest)
    assert out["n"] == 4
    assert out["median_revision"] == -0.5
    assert out["mean_abs_revision"] == pytest.approx((1 + 1 + 0 + 3) / 4)
    assert out["share_revised_down"] == 50.0  # 2 of 4 revised down
    assert out["last_first"] == 100.0
    assert out["last_latest"] == 97.0


def test_revision_summary_aligns_on_overlap():
    first = _series([1.0, 2.0, 3.0])
    latest = _series([1.5, 2.5, 3.5, 4.5])  # one extra month with no first release
    out = revision_summary(first, latest)
    assert out["n"] == 3  # only the overlapping dates
    assert not out["revision"].empty


def test_revision_summary_handles_empty():
    out = revision_summary(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out["n"] == 0
    assert out["aligned"].empty


def test_revision_summary_handles_none():
    out = revision_summary(None, None)
    assert out["n"] == 0
