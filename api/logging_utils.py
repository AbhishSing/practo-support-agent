"""JSON Lines logger with trace IDs."""

import json
import threading
from pathlib import Path
from typing import Any

from agents.guardrails import mask_pii

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "app.jsonl"
_LOG_LOCK = threading.Lock()


def log_event(
    trace_id: str,
    route: str,
    request_text: str,
    duration_ms: float,
    status: int,
) -> None:
    """Append one event, masking phone numbers in request text before writing."""
    masked_request_text, _ = mask_pii(request_text)
    event: dict[str, Any] = {
        "trace_id": trace_id,
        "route": route,
        "request_text": masked_request_text,
        "duration_ms": round(duration_ms, 3),
        "status": status,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with _LOG_LOCK, LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{line}\n")
