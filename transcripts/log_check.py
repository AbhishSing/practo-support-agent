"""Validate JSON Lines API logs and print a short sanitized excerpt."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "app.jsonl"


def main() -> None:
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    required_fields = {
        "trace_id",
        "route",
        "request_text",
        "duration_ms",
        "status",
    }
    events = [json.loads(line) for line in lines]
    if any(not required_fields.issubset(event) for event in events):
        raise ValueError("An API log line is missing required fields.")

    raw_phone_found = any("9876543210" in line for line in lines)
    if raw_phone_found:
        raise ValueError("A raw phone number was found in the API logs.")

    print(f"JSON Lines count: {len(lines)}")
    print("First two lines:")
    for line in lines[:2]:
        print(line)
    print("Raw phone number 9876543210 appeared: no")


if __name__ == "__main__":
    main()
