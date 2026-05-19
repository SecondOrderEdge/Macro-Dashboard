"""Streamlit-aware caching helpers.

Outside Streamlit (e.g. inside pytest) these decorators degrade to plain
function calls so the same code works in both contexts.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable


def cache_resource(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Cache an expensive, non-data resource (models, fitted estimators)."""
    try:
        import streamlit as st

        return st.cache_resource(show_spinner=False)(fn)
    except Exception:

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper


def cache_data(ttl: int = 3600) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache pure data with a TTL."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        try:
            import streamlit as st

            return st.cache_data(ttl=ttl, show_spinner=False)(fn)
        except Exception:

            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

    return _decorator
