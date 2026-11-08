"""Verify phone masking in request logs through the FastAPI /ask route."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi.testclient import TestClient

from api.main import app
from api.logging_utils import LOG_PATH

PHONE_NUMBER = "9876543210"
QUERY = f"Please check APT-0007 for {PHONE_NUMBER}"


def main() -> None:
    previous_line_count = (
        len(LOG_PATH.read_text(encoding="utf-8").splitlines())
        if LOG_PATH.exists()
        else 0
    )
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"session_id": "s2", "query": QUERY},
        )
    if not isinstance(response, httpx.Response):
        raise TypeError("TestClient did not return an httpx.Response.")

    new_lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[
        previous_line_count:
    ]
    events = [json.loads(line) for line in new_lines]
    request_event = next(
        (
            event
            for event in events
            if event["route"] == "/ask"
            and "APT-0007" in event["request_text"]
        ),
        None,
    )
    if request_event is None:
        raise RuntimeError("The /ask request was not written to the JSONL log.")
    if "[PHONE_MASKED]" not in request_event["request_text"]:
        raise AssertionError("The logged request did not contain PHONE_MASKED.")
    if PHONE_NUMBER in request_event["request_text"]:
        raise AssertionError("The raw phone number appeared in the request log.")

    print(f"HTTP status: {response.status_code}")
    print(f"Logged request: {request_event['request_text']}")
    print(f"Raw phone number appeared in this log entry: no")


if __name__ == "__main__":
    main()
