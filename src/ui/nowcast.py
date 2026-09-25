"""'In recession now?' nowcast panel + headline wording for each recession state.

The headline ensemble answers "does a *new* NBER recession start within the
next 12 months?" and is trained only on months not already in a recession. The
nowcast panel (Chauvet-Piger + Sahm rule) answers a different question — "are
we in one now?" — and is descriptive only (never averaged into the ensemble).

How the headline reads per state (see ``recession_probit.recession_state``):

* ``expansion``      — headline shown normally; panel shown below as context.
* ``nowcast_flag``   — a nowcast indicator is over its threshold but NBER has
  not dated a recession: headline shown with a caveat, panel shown first and
  highlighted (NBER dates peaks months after the fact).
* ``nber_recession`` — latest USREC = 1: the start-dated probability has no
  training data for this state, so the headline number is withheld and the
  panel is shown first and highlighted.

Pure functions returning strings so the wiring can be unit-tested without
Streamlit.
"""

from __future__ import annotations

from html import escape

from src.ui.theme import PALETTE

HEADLINE_LABEL = "Probability a new recession starts within 12 months"

_STATE_NOTES = {
    "expansion": "",
    "nowcast_flag": (
        "Nowcast indicators flag that a recession may already be under way. The headline "
        "assumes no recession has started yet, so read it together with the nowcast panel."
    ),
    "nber_recession": (
        "The latest NBER-dated month is a recession month. The start-of-recession "
        "probability is not applicable in this state (the model is trained only on months "
        "not already in a recession), so it is withheld; see the nowcast panel."
    ),
}


def state_of(report: dict | None) -> str:
    if not report or "error" in report:
        return "expansion"
    return str(report.get("recession_state") or (report.get("nowcast") or {}).get("state") or "expansion")


def headline_applicable(report: dict | None) -> bool:
    return state_of(report) != "nber_recession"


def headline_value_text(report: dict | None) -> str:
    """Headline number as displayed ("—" when withheld or unavailable)."""
    if not report or "error" in report or not headline_applicable(report):
        return "—"
    val = report.get("ensemble_probability")
    try:
        return f"{float(val):.0f}"
    except (TypeError, ValueError):
        return "—"


def state_note(report: dict | None) -> str:
    return _STATE_NOTES.get(state_of(report), "")


def panel_is_prominent(report: dict | None) -> bool:
    return state_of(report) in ("nowcast_flag", "nber_recession")


def _fmt(ind: dict) -> str:
    v = ind.get("value")
    if v is None:
        return "—"
    unit = ind.get("unit", "")
    return f"{v:.0f}%" if unit == "%" else f"{v:+.2f} {unit}".strip()


def nowcast_rows(report: dict | None) -> list[dict]:
    """Display rows for the nowcast panel (Chauvet-Piger, Sahm rule)."""
    nc = (report or {}).get("nowcast") or {}
    rows = []
    for name, ind in (nc.get("indicators") or {}).items():
        thr = ind.get("threshold")
        unit = ind.get("unit", "")
        thr_txt = f"{thr:.0f}%" if unit == "%" else f"{thr:.2f} {unit}"
        rows.append({
            "name": name,
            "value": _fmt(ind),
            "as_of": ind.get("as_of") or "—",
            "threshold": thr_txt,
            "signal": bool(ind.get("signal")),
            "source": ind.get("source", ""),
        })
    return rows


def nowcast_panel_html(report: dict | None, *, compact: bool = False) -> str:
    """HTML for the 'In recession now?' panel. Highlighted when prominent."""
    rows = nowcast_rows(report)
    nc = (report or {}).get("nowcast") or {}
    prominent = panel_is_prominent(report)
    border = PALETTE["risk_critical"] if prominent else PALETTE["panel_border"]
    body = ""
    for r in rows:
        color = PALETTE["risk_critical"] if r["signal"] else PALETTE["risk_low"]
        status = "over threshold" if r["signal"] else "below threshold"
        extra = "" if compact else (
            f'<span style="color:{PALETTE["text_tiny"]};font-size:10px;"> · {escape(r["source"])}</span>'
        )
        body += (
            f'<div class="submodel-row"><span class="name">{escape(r["name"])}{extra}</span>'
            f'<span class="value"><span style="color:{color};">{escape(r["value"])}</span>'
            f'<span style="color:{PALETTE["text_muted"]};font-size:10px;"> · {escape(r["as_of"])} · '
            f'threshold {escape(r["threshold"])} · {status}</span></span></div>'
        )
    usrec_txt = ""
    if nc.get("usrec_as_of"):
        in_rec = (nc.get("usrec_latest") or 0) >= 0.5
        usrec_txt = (
            f'<div class="submodel-row"><span class="name">NBER-dated (USREC)</span>'
            f'<span class="value">{"recession" if in_rec else "no recession dated"} · '
            f'through {escape(nc["usrec_as_of"])}</span></div>'
        )
    note = state_note(report)
    note_html = (
        f'<div style="margin-top:8px;font-size:12px;line-height:1.6;color:{PALETTE["text_primary"]};">'
        f"<b>{escape(note)}</b></div>" if note else ""
    )
    foot = "" if compact else (
        f'<div style="margin-top:8px;font-size:11px;line-height:1.6;color:{PALETTE["text_muted"]};">'
        "Descriptive only: these coincident indicators are not part of the ensemble and are not "
        "scored. Chauvet-Piger is FRED's smoothed Markov-switching series, re-estimated each "
        "vintage, so its history is not real-time or out-of-sample. The Sahm rule stays above "
        "its threshold for a while after troughs, and NBER dates turning points months later."
        "</div>"
    )
    if not rows and not usrec_txt:
        body = '<div class="submodel-row"><span class="name">—</span><span class="value">nowcast unavailable</span></div>'
    return (
        f'<div class="panel" style="border-color:{border};">'
        '<div class="panel-header"><span>In recession now? · nowcast panel</span>'
        f'<span style="color:{border if prominent else PALETTE["text_muted"]};">'
        f'{"HIGHLIGHTED" if prominent else "descriptive"}</span></div>'
        f'<div class="panel-body">{body}{usrec_txt}{note_html}{foot}</div></div>'
    )
