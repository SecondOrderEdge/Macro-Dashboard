"""Early-warning ladder: how close is trouble?

Trouble — slow macro or fast financial — tends to travel a recognizable chain,
from the earliest/slowest signal (yield-curve inversion, ~12–18mo lead) down to
the fastest (vol spikes, days–weeks). This reduces inputs the dashboard already
computes into a lead-time-ordered ladder of binary "has this warning fired?"
rungs; the depth of the lit rungs answers "how close is trouble?".

Descriptive, not a forecast: lead times are stylized averages, thresholds are
judgmental (surfaced on each rung), and the sequence is NOT deterministic — 2020
lit the bottom rungs with no curve-led runway. Pure functions over the panel +
the probit report + the LAME model, so they're unit-testable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _last(panel: pd.DataFrame, col: str) -> tuple[float, Any]:
    if panel is None or col not in panel.columns:
        return float("nan"), None
    s = panel[col].dropna()
    return (float(s.iloc[-1]), s.index[-1]) if not s.empty else (float("nan"), None)


def _yoy(panel: pd.DataFrame, col: str, periods: int = 12) -> tuple[float, Any]:
    if panel is None or col not in panel.columns:
        return float("nan"), None
    s = panel[col].dropna()
    if s.empty:
        return float("nan"), None
    g = (s.resample("ME").last().pct_change(periods) * 100.0).dropna()
    return (float(g.iloc[-1]), g.index[-1]) if not g.empty else (float("nan"), None)


def _sahm(panel: pd.DataFrame) -> tuple[float, Any]:
    try:
        from src.models.external import sahm_rule

        s = sahm_rule(panel).dropna()
        return (float(s.iloc[-1]), s.index[-1]) if not s.empty else (float("nan"), None)
    except Exception:  # noqa: BLE001
        return float("nan"), None


def _ensemble(probit: dict | None) -> float:
    if not probit or "error" in probit:
        return float("nan")
    v = probit.get("ensemble_probability")
    return float(v) if v is not None else float("nan")


def _below_trend(lame) -> float:
    if lame is None:
        return float("nan")
    try:
        z = lame.current_breakdown()["z_score"].dropna()
        return float((z < 0).mean() * 100.0) if not z.empty else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def build_ladder(panel: pd.DataFrame, probit: dict | None = None, lame=None) -> list[dict]:
    """Lead-time-ordered rungs (slowest/earliest first). Each rung that can be
    computed becomes a dict; rungs whose inputs are unavailable are omitted."""
    rungs: list[dict] = []

    def add(key, label, track, lead, value, value_str, lit, severity, detail, as_of):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return
        rungs.append({
            "key": key, "label": label, "track": track, "lead": lead,
            "value": float(value), "value_str": value_str, "lit": bool(lit),
            "severity": severity, "detail": detail, "as_of": as_of,
        })

    # 1) Yield-curve inversion — earliest, slowest macro signal.
    v, dt = _last(panel, "T10Y3M")
    add("curve", "Yield-curve inversion (10y–3m)", "macro", "≈ 12–18 mo",
        v, f"{v:+.2f} pp" if np.isfinite(v) else "—", np.isfinite(v) and v <= 0,
        "critical" if v <= -0.5 else "high" if v <= 0 else "elevated" if v <= 0.5 else "low",
        "Inverts (≤ 0) roughly 12–18 months before recessions.", dt)

    # 2) Bank lending standards (SLOOS).
    v, dt = _last(panel, "DRTSCILM")
    add("sloos", "Bank lending standards (SLOOS)", "macro", "≈ 6–12 mo",
        v, f"{v:+.0f}% net" if np.isfinite(v) else "—", np.isfinite(v) and v > 0,
        "critical" if v >= 40 else "high" if v >= 20 else "elevated" if v > 0 else "low",
        "Net % of banks tightening C&I standards; positive = tightening.", dt)

    # 3) Housing rolling over (permits). Dead-band: a trivially-negative YoY
    # (noise around zero) shouldn't light "rolling over"; require a real decline.
    v, dt = _yoy(panel, "PERMIT")
    add("housing", "Housing permits (YoY)", "macro", "≈ 6–9 mo",
        v, f"{v:+.1f}%" if np.isfinite(v) else "—", np.isfinite(v) and v < -2.0,
        "critical" if v <= -15 else "high" if v <= -7 else "elevated" if v < -2 else "low",
        "Building permits YoY; lit when clearly contracting (< −2%), not just flat.", dt)

    # 4) Labor cracking (Sahm).
    v, dt = _sahm(panel)
    add("sahm", "Labor: Sahm rule", "macro", "0–6 mo",
        v, f"{v:+.2f} pp" if np.isfinite(v) else "—", np.isfinite(v) and v >= 0.5,
        "critical" if v >= 0.5 else "high" if v >= 0.4 else "elevated" if v >= 0.2 else "low",
        "3-mo unemployment vs its 12-mo low; triggers at +0.5.", dt)

    # 5) Recession ensemble (the synthesized 12-month forward read).
    v = _ensemble(probit)
    add("ensemble", "Recession ensemble (12-mo)", "macro", "12-mo model",
        v, f"{v:.0f}%" if np.isfinite(v) else "—", np.isfinite(v) and v > 30,
        "critical" if v > 50 else "high" if v > 30 else "elevated" if v > 15 else "low",
        "Four-model 12-month-ahead probability; warning >30%, elevated >50%.", None)

    # 6) Breadth deterioration.
    v = _below_trend(lame)
    add("breadth", "Labor breadth below trend", "financial", "weeks–mo",
        v, f"{v:.0f}%" if np.isfinite(v) else "—", np.isfinite(v) and v >= 55,
        "critical" if v >= 70 else "high" if v >= 55 else "elevated" if v >= 40 else "low",
        "Share of labor indicators below their own trend.", None)

    # 7) Financial conditions (NFCI).
    v, dt = _last(panel, "NFCI")
    add("nfci", "Financial conditions (NFCI)", "financial", "weeks",
        v, f"{v:+.2f}" if np.isfinite(v) else "—", np.isfinite(v) and v > 0,
        "critical" if v >= 1.0 else "high" if v >= 0.5 else "elevated" if v > 0 else "low",
        "Chicago Fed NFCI; positive = tighter-than-average conditions.", dt)

    # 8) Acute market stress (VIX) — fastest, latest.
    v, dt = _last(panel, "VIXCLS")
    add("vix", "Acute stress: VIX", "financial", "days–wks",
        v, f"{v:.0f}" if np.isfinite(v) else "—", np.isfinite(v) and v >= 25,
        "critical" if v >= 40 else "high" if v >= 30 else "elevated" if v >= 20 else "low",
        "Equity-vol spikes (≥ 25) flag acute financial stress.", dt)

    return rungs


def ladder_summary(rungs: list[dict]) -> dict:
    """Headline read: how many rungs are lit and how deep the escalation goes.

    'Deepest lit' (lowest in the lead-time order) sets the stage, since the
    bottom rungs are the shortest-lead / nearest-to-the-event signals.
    """
    total = len(rungs)
    if total == 0:
        return {"n_lit": 0, "n_total": 0, "stage": "—", "severity": "low", "deepest": None}
    lit_idx = [i for i, r in enumerate(rungs) if r["lit"]]
    if not lit_idx:
        return {"n_lit": 0, "n_total": total, "stage": "ALL CLEAR", "severity": "low", "deepest": None}
    deepest = max(lit_idx)
    frac = deepest / (total - 1) if total > 1 else 1.0
    if frac < 0.34:
        stage, sev = "EARLY · distant", "elevated"
    elif frac < 0.67:
        stage, sev = "BUILDING · approaching", "high"
    else:
        stage, sev = "ACUTE · near / here", "critical"
    return {
        "n_lit": len(lit_idx), "n_total": total, "stage": stage,
        "severity": sev, "deepest": rungs[deepest]["label"],
    }
