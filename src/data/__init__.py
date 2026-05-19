"""Data access layer: FRED client, series registry, NBER helpers."""

from .fred_client import fetch_panel, fetch_series, forward_fill_limited
from .series_registry import SERIES_REGISTRY, transform_series
from .nber import load_nber_recessions, recession_in_next_12m

__all__ = [
    "fetch_panel",
    "fetch_series",
    "forward_fill_limited",
    "SERIES_REGISTRY",
    "transform_series",
    "load_nber_recessions",
    "recession_in_next_12m",
]
