"""Validated response shape for support-agent outputs."""

from typing import Any

from pydantic import BaseModel, ValidationError


class AgentResponse(BaseModel):
    answer: str
    sources: list[str]
    tool_used: list[str]
    escalation_score: float | None
    grounded: bool


def validate_agent_response(data: dict[str, Any]) -> AgentResponse:
    """Validate response data and print Pydantic validation errors before raising."""
    try:
        return AgentResponse(**data)
    except ValidationError as error:
        print(f"AgentResponse validation error:\n{error}")
        raise
