"""Exercise HTTP and WebSocket API routes, including abrupt disconnect recovery."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import websockets

BASE_URL = "http://127.0.0.1:8000"
WEBSOCKET_URL = "ws://127.0.0.1:8000/ws/chat"


async def _exercise_websockets() -> None:
    first = await websockets.connect(WEBSOCKET_URL)
    await first.send("Can I cancel my appointment?")
    print(f"First WebSocket reply: {json.loads(await first.recv())}")
    first.transport.abort()
    await asyncio.sleep(0.2)
    print("First WebSocket disconnected abruptly.")

    async with websockets.connect(WEBSOCKET_URL) as second:
        await second.send("How much is a cardiology consultation?")
        print(f"Second WebSocket reply: {json.loads(await second.recv())}")


def main() -> None:
    ask = httpx.post(
        f"{BASE_URL}/ask",
        json={
            "session_id": "s1",
            "query": "Can I cancel my appointment?",
        },
        timeout=120,
    )
    print(f"/ask status: {ask.status_code}")
    print(f"/ask response: {ask.json()}")

    add_document = httpx.post(
        f"{BASE_URL}/add-document",
        json={
            "doc_id": "demo-temp",
            "text": "Temporary demo document for indexing.",
        },
        timeout=120,
    )
    print(f"/add-document status: {add_document.status_code}")
    print(f"/add-document response: {add_document.json()}")

    asyncio.run(_exercise_websockets())


if __name__ == "__main__":
    main()
