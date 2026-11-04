"""FastAPI endpoints for clinic support."""

import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from agents.schemas import AgentResponse
from api.logging_utils import log_event
from governance.policy import BudgetExceeded

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "kb"

app = FastAPI()


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)


class AddDocRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def _run_query(query: str) -> AgentResponse:
    from agents.crew import build_crew, kickoff_validated

    return kickoff_validated(build_crew(), query)


def _append_document(doc_id: str, text: str) -> tuple[str, int]:
    filename = doc_id if doc_id.endswith(".md") else f"{doc_id}.md"
    if Path(filename).name != filename or filename in {".md", ".."}:
        raise ValueError("doc_id must be a filename inside kb/, not a path")

    KB_DIR.mkdir(parents=True, exist_ok=True)
    document_path = KB_DIR / filename
    existing_text = (
        document_path.read_text(encoding="utf-8")
        if document_path.exists()
        else ""
    )
    combined_text = f"{existing_text.rstrip()}\n\n{text}" if existing_text else text
    document_path.write_text(combined_text, encoding="utf-8")

    from rag.index import index_document

    return filename, index_document(filename, combined_text)


@app.post("/ask", response_model=AgentResponse)
def ask(request: AskRequest) -> AgentResponse:
    trace_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    status = 500
    try:
        response = _run_query(request.query)
        status = 200
        return response
    except BudgetExceeded as error:
        status = 413
        raise HTTPException(status_code=413, detail=str(error)) from error
    finally:
        log_event(
            trace_id,
            "/ask",
            request.query,
            (time.perf_counter() - started_at) * 1000,
            status,
        )


@app.post("/add-document")
def add_document(request: AddDocRequest) -> dict[str, int | str]:
    trace_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    status = 500
    try:
        filename, count = _append_document(request.doc_id, request.text)
        status = 200
        return {"doc_id": filename, "count": count}
    finally:
        log_event(
            trace_id,
            "/add-document",
            request.text,
            (time.perf_counter() - started_at) * 1000,
            status,
        )


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    trace_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    request_text: list[str] = []
    status = 101
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            request_text.append(query)
            try:
                response = await run_in_threadpool(_run_query, query)
            except BudgetExceeded as error:
                await websocket.send_json({"error": str(error)})
                continue
            await websocket.send_json(response.model_dump(mode="json"))
    except WebSocketDisconnect as disconnect:
        status = disconnect.code
    finally:
        log_event(
            trace_id,
            "/ws/chat",
            "\n".join(request_text),
            (time.perf_counter() - started_at) * 1000,
            status,
        )
