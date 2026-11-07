"""Demonstrate session-isolated chat history and write a transcript."""

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from agents.memory import SESSION_HISTORY, with_session_history

FIRST_SESSION_ID = "memory-demo-first"
SECOND_SESSION_ID = "memory-demo-second"


def _respond(inputs: dict[str, Any]) -> str:
    question = str(inputs["input"])
    history = inputs.get("history", [])
    previous_user_messages = [
        message.content
        for message in history
        if isinstance(message, HumanMessage) and isinstance(message.content, str)
    ]
    history_text = "\n".join(previous_user_messages)

    appointment_match = re.search(r"\bAPT-\d+\b", history_text, re.IGNORECASE)
    doctor_match = re.search(
        r"\bDr\.?\s+[A-Z][a-z]+", history_text, flags=re.IGNORECASE
    )
    specialty_match = re.search(
        r"\b(cardiology|dermatology|pediatrics|orthopedics|general medicine)\b",
        history_text,
        flags=re.IGNORECASE,
    )

    if not history_text:
        if question.lower().startswith("which specialty"):
            return "I don't have that context in this session."
        return "Got it. I'll keep that appointment detail in this session."
    if question.lower().startswith("which specialty"):
        specialty = specialty_match.group(1) if specialty_match else "unknown"
        doctor = doctor_match.group(0) if doctor_match else "unknown"
        return f"You mentioned {specialty} and {doctor}."
    if question.lower().startswith("what was the appointment id"):
        if appointment_match:
            return f"The appointment ID was {appointment_match.group(0)}."
        return "I don't have an appointment ID in this session."
    return "I can use information from earlier turns in this session."


def run_demo() -> str:
    """Run three turns in one session, then ask the same context in another."""
    SESSION_HISTORY.pop(FIRST_SESSION_ID, None)
    SESSION_HISTORY.pop(SECOND_SESSION_ID, None)
    conversation = with_session_history(RunnableLambda(_respond))
    transcript: list[str] = []

    first_session_turns = (
        "I have a cardiology appointment APT-0007 with Dr. Rao.",
        "Which specialty and doctor did I mention?",
        "What was the appointment ID?",
    )
    for turn_number, question in enumerate(first_session_turns, start=1):
        response = conversation.invoke(
            {"input": question},
            config={"configurable": {"session_id": FIRST_SESSION_ID}},
        )
        transcript.extend(
            (
                f"Session {FIRST_SESSION_ID} — Turn {turn_number}",
                f"User: {question}",
                f"Assistant: {response}",
                "",
            )
        )

    isolated_question = "Which specialty and doctor did I mention earlier?"
    isolated_response = conversation.invoke(
        {"input": isolated_question},
        config={"configurable": {"session_id": SECOND_SESSION_ID}},
    )
    transcript.extend(
        (
            f"Session {SECOND_SESSION_ID} — isolated from {FIRST_SESSION_ID}",
            f"User: {isolated_question}",
            f"Assistant: {isolated_response}",
            "Expected: the first session's appointment details are unavailable.",
        )
    )

    output = "\n".join(transcript) + "\n"
    transcript_path = Path(__file__).with_name("memory_transcript.txt")
    transcript_path.write_text(output, encoding="utf-8")
    print(output, end="")
    return output


if __name__ == "__main__":
    run_demo()
