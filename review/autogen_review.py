"""Offline AutoGen review team with deterministic structured verdicts."""

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

UNSUPPORTED_CLAIM_MARKER = "Consultations are free on Sundays"
DEMO_CONTEXT = (
    "General Medicine consultations start at ₹500. Cardiology and Orthopedics "
    "carry a standard consultation fee of ₹1,200. Diagnostic screenings and "
    "minor in-office procedures are billed separately."
)
DEMO_DRAFT = "Cardiology consultations have a standard fee of ₹1,200."
REVISED_DRAFT = (
    "Cardiology consultations have a standard fee of ₹1,200. "
    f"{UNSUPPORTED_CLAIM_MARKER}."
)
_DRAFT_PATTERN = re.compile(
    r"Composer draft:\s*(.*?)\s*\n\s*Retrieved context:",
    re.DOTALL | re.IGNORECASE,
)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class DeterministicReviewClient(ChatCompletionClient):
    """Small local-only AutoGen client for predictable review demonstrations."""

    def __init__(self) -> None:
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        }

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
    ) -> int:
        del tools
        return sum(len(str(getattr(message, "content", ""))) // 4 for message in messages)

    def remaining_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
    ) -> int:
        return max(0, 4096 - self.count_tokens(messages, tools=tools))

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        del tools, extra_create_args, cancellation_token
        prompt = _message_text(messages)
        input_tokens = self.count_tokens(messages)

        if json_output is Verdict:
            response_text = self._verdict_json(prompt)
        else:
            response_text = self._reviewer_response(prompt)

        completion_tokens = max(1, len(response_text) // 4)
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + input_tokens,
            completion_tokens=self._usage.completion_tokens + completion_tokens,
        )
        return CreateResult(
            finish_reason="stop",
            content=response_text,
            usage=RequestUsage(
                prompt_tokens=input_tokens,
                completion_tokens=completion_tokens,
            ),
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        result = await self.create(
            messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    async def close(self) -> None:
        return None

    def _reviewer_response(self, prompt: str) -> str:
        draft = _extract_draft(prompt)
        if UNSUPPORTED_CLAIM_MARKER.casefold() in draft.casefold():
            return (
                "The draft's Sunday-free-consultation claim is not supported by "
                "the retrieved context. The final editor should remove it."
            )
        return "The draft is supported by the retrieved context; keep it unchanged."

    def _verdict_json(self, prompt: str) -> str:
        draft = _extract_draft(prompt)
        if UNSUPPORTED_CLAIM_MARKER.casefold() not in draft.casefold():
            verdict = Verdict(
                approved=True,
                final_answer=draft,
                reason="The answer is supported by the supplied retrieved context.",
            )
        else:
            revised_answer = _remove_unsupported_claim(draft)
            if not revised_answer:
                revised_answer = (
                    "The retrieved context does not establish that consultations "
                    "are free on Sundays."
                )
            verdict = Verdict(
                approved=False,
                final_answer=revised_answer,
                reason="Removed the unsupported Sunday consultation claim.",
            )
        return verdict.model_dump_json()


def build_review_team() -> RoundRobinGroupChat:
    """Create the two-turn policy reviewer and structured final editor team."""
    model_client = DeterministicReviewClient()
    policy_reviewer = AssistantAgent(
        name="policy_reviewer",
        model_client=model_client,
        system_message=(
            "Check the Composer draft against the retrieved context. Identify "
            "unsupported claims and tell the final editor what to change."
        ),
    )
    final_editor = AssistantAgent(
        name="final_editor",
        model_client=model_client,
        system_message=(
            "Return a final review verdict. Preserve a supported draft unchanged; "
            "remove any claim the reviewer identifies as unsupported."
        ),
        output_content_type=Verdict,
    )
    return RoundRobinGroupChat(
        [policy_reviewer, final_editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[Verdict]],
    )


async def review_draft(draft: str, context: str) -> Verdict:
    """Run the deterministic two-agent review and return the final verdict."""
    team = build_review_team()
    result = await team.run(
        task=(
            f"Composer draft:\n{draft}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Review the draft using only the supplied context."
        )
    )
    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage):
            return Verdict.model_validate(message.content)
    raise RuntimeError("The review team did not return a structured Verdict.")


async def run_demo() -> list[Verdict]:
    """Run supported and deliberately unsupported drafts and save the outputs."""
    verdicts = [
        await review_draft(DEMO_DRAFT, DEMO_CONTEXT),
        await review_draft(REVISED_DRAFT, DEMO_CONTEXT),
    ]
    transcript = "\n".join(
        f"Run {index}:\n{verdict.model_dump_json(indent=2)}"
        for index, verdict in enumerate(verdicts, start=1)
    )
    transcript_path = Path(__file__).resolve().parents[1] / "transcripts" / "review_demo.txt"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(f"{transcript}\n", encoding="utf-8")
    print(transcript)
    return verdicts


def _message_text(messages: Sequence[LLMMessage]) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _extract_draft(prompt: str) -> str:
    match = _DRAFT_PATTERN.search(prompt)
    if match is None:
        raise ValueError("The review prompt is missing the Composer draft.")
    return match.group(1).strip()


def _remove_unsupported_claim(draft: str) -> str:
    kept_sentences = [
        sentence.strip()
        for sentence in _SENTENCE_PATTERN.split(draft)
        if UNSUPPORTED_CLAIM_MARKER.casefold() not in sentence.casefold()
    ]
    return " ".join(sentence for sentence in kept_sentences if sentence)


if __name__ == "__main__":
    asyncio.run(run_demo())
