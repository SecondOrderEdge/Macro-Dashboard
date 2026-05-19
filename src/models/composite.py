"""Composite top-level risk score.

The dashboard headline collapses three indicators into a single 0–100 number:

* the recession ensemble probability (already in percent),
* the LAME labor composite (z-score, lower is worse),
* the 10Y–3M Treasury spread (percentage points, lower is worse).

LAME and the curve are mapped to 0–100 "risk" using simple anchored linear
transforms; the composite is then a weighted sum (50/25/25).
"""

from __future__ import annotations

import numpy as np


def lame_to_risk(lame_z: float) -> float:
    """Convert a LAME z-score to a 0-100 risk reading.

    Anchors: z = +2 → risk = 0, z = 0 → risk = 50, z = -2 → risk = 100.
    """
    if lame_z is None or not np.isfinite(lame_z):
        return float("nan")
    return float(np.clip(50.0 - lame_z * 25.0, 0.0, 100.0))


def curve_to_risk(spread_10y3m: float) -> float:
    """Convert the 10Y-3M spread (in percentage points) to a 0-100 risk reading.

    Anchors: spread = +2.5pp → risk = 0, spread = 0 → risk = 50,
    spread = -2.5pp → risk = 100.
    """
    if spread_10y3m is None or not np.isfinite(spread_10y3m):
        return float("nan")
    return float(np.clip(50.0 - spread_10y3m * 20.0, 0.0, 100.0))


def _band(score: float) -> str:
    if score < 20:
        return "LOW"
    if score < 40:
        return "ELEVATED"
    if score < 60:
        return "HIGH"
    return "CRITICAL"


def composite_risk(ensemble_pct: float, lame_z: float, curve_10y3m: float) -> dict:
    """Combine the three modules into a single 0–100 composite.

    Weights: 50% ensemble probability, 25% LAME-risk, 25% curve-risk.
    """
    ensemble_risk = float(np.clip(ensemble_pct, 0.0, 100.0)) if np.isfinite(ensemble_pct) else float("nan")
    lame_risk = lame_to_risk(lame_z)
    curve_risk = curve_to_risk(curve_10y3m)

    parts = [
        ("ensemble", 0.50, ensemble_risk),
        ("lame", 0.25, lame_risk),
        ("curve", 0.25, curve_risk),
    ]

    # If any part is missing, redistribute its weight across the available ones.
    available = [(name, w, v) for name, w, v in parts if np.isfinite(v)]
    if not available:
        return {"composite": 0, "band": "LOW", "contributions": {n: 0.0 for n, _, _ in parts}}

    total_w = sum(w for _, w, _ in available)
    composite = sum((w / total_w) * v for _, w, v in available)
    contributions = {name: (w / total_w) * v if np.isfinite(v) else 0.0 for name, w, v in parts}

    score = int(round(np.clip(composite, 0.0, 100.0)))
    return {
        "composite": score,
        "band": _band(score),
        "contributions": contributions,
    }
