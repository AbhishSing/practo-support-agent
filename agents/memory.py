"""In-memory chat history keyed by session ID."""

from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import Runnable, RunnableWithMessageHistory

SESSION_HISTORY: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the shared message history for a session, creating it if needed."""
    if not session_id:
        raise ValueError("session_id must not be empty")
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = InMemoryChatMessageHistory()
    return SESSION_HISTORY[session_id]


def with_session_history(runnable: Runnable[Any, Any]) -> RunnableWithMessageHistory:
    """Wrap a runnable that accepts ``input`` and ``history`` with session memory."""
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
