"""CrewAI tools for policy retrieval and appointment-status lookup."""

import os
from statistics import quantiles
from typing import Any

from crewai.tools import tool

from dataset import APPOINTMENTS
from rag.retrieve import answer

RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "kb_sentence")
_APPOINTMENTS_BY_ID = {
    str(appointment["record_id"]).upper(): appointment for appointment in APPOINTMENTS
}
DAYS_SINCE_CREATED_P80 = quantiles(
    [int(appointment["days_since_created"]) for appointment in APPOINTMENTS],
    n=5,
    method="inclusive",
)[3]
TOOL_CALLS: list[str] = []


def _rag_threshold() -> float:
    value = os.getenv("RAG_SIMILARITY_THRESHOLD")
    if value is None:
        raise RuntimeError(
            "Set RAG_SIMILARITY_THRESHOLD using the scores from calibrate.py "
            "before running policy retrieval."
        )
    try:
        threshold = float(value)
    except ValueError as exc:
        raise ValueError("RAG_SIMILARITY_THRESHOLD must be a number") from exc
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("RAG_SIMILARITY_THRESHOLD must be between -1 and 1")
    return threshold


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Search the clinic policy knowledge base for an answer to a query."""
    TOOL_CALLS.append("rag_lookup")
    return answer(
        query,
        collection_name=RAG_COLLECTION_NAME,
        threshold=_rag_threshold(),
    )


@tool("check_appointment_status")
def check_appointment_status(record_id: str) -> dict[str, Any]:
    """Look up appointment status, consultation fee, and escalation score."""
    TOOL_CALLS.append("check_appointment_status")
    appointment = _APPOINTMENTS_BY_ID.get(record_id.strip().upper())
    if appointment is None:
        raise ValueError(f"Appointment {record_id!r} was not found")

    follow_up_required = bool(appointment["follow_up_required"])
    days_since_created = int(appointment["days_since_created"])
    # Follow-up is the stronger direct signal; appointment age is a secondary, scaled contributor.
    escalation_score = 0.6 * int(follow_up_required) + 0.4 * (
        days_since_created / 30
    )
    return {
        "status": appointment["status"],
        "consultation_fee_inr": appointment["consultation_fee_inr"],
        "escalation_score": round(escalation_score, 4),
    }


def reset_tool_call_log() -> None:
    """Clear the tool-call log before an individual demo run."""
    TOOL_CALLS.clear()
