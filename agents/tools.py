"""CrewAI tools for policy retrieval and appointment-status lookup."""

import os
from statistics import quantiles
from typing import Any

from crewai.tools import tool

from dataset import APPOINTMENTS
from rag.retrieve import answer_with_context

RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "kb_fixed")
RAG_SIMILARITY_THRESHOLD = 0.33285
_APPOINTMENTS_BY_ID = {
    str(appointment["record_id"]).upper(): appointment for appointment in APPOINTMENTS
}
DAYS_SINCE_CREATED_P80 = quantiles(
    [int(appointment["days_since_created"]) for appointment in APPOINTMENTS],
    n=5,
    method="inclusive",
)[3]


def _escalation_score(follow_up_required: bool, days_since_created: float) -> float:
    return 0.6 * int(follow_up_required) + 0.4 * (days_since_created / 30)


def calculate_escalation_threshold() -> float:
    """Derive the escalation threshold from follow-up-required records at P80 age."""
    return _escalation_score(True, DAYS_SINCE_CREATED_P80)


ESCALATION_THRESHOLD = calculate_escalation_threshold()
TOOL_CALLS: list[str] = []
LAST_RETRIEVAL: dict[str, Any] = {"sources": [], "context": ""}
LAST_APPOINTMENT_LOOKUP: dict[str, Any] = {}


def _rag_threshold() -> float:
    value = os.getenv("RAG_SIMILARITY_THRESHOLD")
    if value is None:
        return RAG_SIMILARITY_THRESHOLD
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
    answer_text, sources, context = answer_with_context(
        query,
        collection_name=RAG_COLLECTION_NAME,
        threshold=_rag_threshold(),
    )
    LAST_RETRIEVAL.update(sources=sources, context=context)
    return answer_text


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
    escalation_score = _escalation_score(
        follow_up_required, days_since_created
    )
    result = {
        "status": appointment["status"],
        "consultation_fee_inr": appointment["consultation_fee_inr"],
        "escalation_score": round(escalation_score, 4),
        "escalate": escalation_score >= ESCALATION_THRESHOLD,
    }
    LAST_APPOINTMENT_LOOKUP.clear()
    LAST_APPOINTMENT_LOOKUP.update(result)
    return result


def reset_tool_call_log() -> None:
    """Clear per-request tool output before an individual run."""
    TOOL_CALLS.clear()
    LAST_RETRIEVAL.update(sources=[], context="")
    LAST_APPOINTMENT_LOOKUP.clear()
