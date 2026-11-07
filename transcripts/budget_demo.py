"""Verify the runtime token budget through the FastAPI /ask endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import app


def main() -> None:
    with TestClient(app) as client:
        normal = client.post(
            "/ask",
            json={
                "session_id": "budget-demo",
                "query": "Can I cancel my appointment?",
            },
        )
        print(f"Normal query status: {normal.status_code}")
        print(f"Normal query response: {normal.json()}")

        oversized = client.post(
            "/ask",
            json={
                "session_id": "budget-demo",
                "query": "appointment " * 1001,
            },
        )
        print(f"Oversized query status: {oversized.status_code}")
        print(f"Oversized query detail: {oversized.json()['detail']}")


if __name__ == "__main__":
    main()
