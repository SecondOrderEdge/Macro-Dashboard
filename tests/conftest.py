"""Shared pytest fixtures and synthetic data helpers.

We avoid hitting the FRED API in tests by generating panels with the same
column names (FRED IDs) the model code expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def monthly_index():
    return pd.date_range("1970-01-01", "2025-12-01", freq="MS")


@pytest.fixture
def synthetic_panel(monthly_index, rng):
    """Build a synthetic FRED-style panel with all series used by the models."""
    n = len(monthly_index)

    def trended(level, scale, drift=0.0):
        return level + np.cumsum(rng.normal(drift, scale, size=n))

    panel = pd.DataFrame(index=monthly_index)
    # Curve
    panel["DGS10"] = (4 + trended(0, 0.05)).clip(0.2, None)
    panel["DGS2"] = (panel["DGS10"] - 0.8 + rng.normal(0, 0.15, n)).clip(0.1, None)
    panel["DGS3MO"] = (panel["DGS2"] - 0.4 + rng.normal(0, 0.1, n)).clip(0.05, None)
    panel["DGS5"] = (panel["DGS10"] - 0.4 + rng.normal(0, 0.1, n)).clip(0.1, None)
    panel["DGS1MO"] = panel["DGS3MO"] - 0.05
    panel["DGS6MO"] = panel["DGS3MO"] + 0.05
    panel["DGS1"] = panel["DGS2"] - 0.2
    panel["DGS3"] = panel["DGS5"] - 0.1
    panel["DGS7"] = panel["DGS10"] - 0.2
    panel["DGS20"] = panel["DGS10"] + 0.1
    panel["DGS30"] = panel["DGS10"] + 0.2
    panel["T10Y3M"] = panel["DGS10"] - panel["DGS3MO"]
    panel["T10Y2Y"] = panel["DGS10"] - panel["DGS2"]
    # Labor
    panel["UNRATE"] = (5 + trended(0, 0.05)).clip(2.0, 15.0)
    panel["ICSA"] = 350_000 + rng.normal(0, 5_000, n)
    panel["CCSA"] = 2_500_000 + rng.normal(0, 20_000, n)
    panel["JTSJOL"] = (7_000 + trended(0, 50)).clip(2_000, None)
    panel["JTSQUR"] = (2.5 + rng.normal(0, 0.1, n)).clip(1.0, None)
    panel["AWHAETP"] = 34.5 + rng.normal(0, 0.1, n)
    panel["TEMPHELPS"] = (2_500 + trended(0, 5)).clip(1_000, None)
    panel["PAYEMS"] = (150_000 + trended(0, 100, drift=50)).clip(100_000, None)
    panel["U6RATE"] = 9 + trended(0, 0.05)
    panel["CIVPART"] = 63 + rng.normal(0, 0.1, n)
    # Credit
    panel["BAA10Y"] = (2 + rng.normal(0, 0.3, n)).clip(0.5, None)
    panel["DRTSCILM"] = rng.normal(0, 10, n)
    panel["BAMLH0A0HYM2"] = (5 + rng.normal(0, 0.5, n)).clip(2.0, None)
    # Housing
    panel["PERMIT"] = (1_500 + trended(0, 5)).clip(500, None)
    panel["HOUST"] = (1_400 + trended(0, 5)).clip(400, None)
    panel["PCEC96"] = 10_000 + np.arange(n) * 5
    # Sentiment
    panel["VIXCLS"] = (18 + np.abs(rng.normal(0, 5, n))).clip(8, None)
    panel["USSLIND"] = 1.5 + rng.normal(0, 0.3, n)
    panel["SP500"] = (1_000 + np.arange(n) * 3).clip(300, None)

    panel.index = pd.DatetimeIndex(panel.index)
    return panel


@pytest.fixture
def nber_series():
    from src.data.nber import load_nber_recessions

    return load_nber_recessions(start="1970-01-01", end="2025-12-31")


@pytest.fixture
def fwd_target(nber_series):
    from src.data.nber import recession_in_next_12m

    return recession_in_next_12m(nber_series)
