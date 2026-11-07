"""Deterministic CrewAI LLM for offline agent development and demos."""

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from crewai.llms.base_llm import BaseLLM

_APPOINTMENT_ID_PATTERN = re.compile(r"\bAPT-\d+\b", re.IGNORECASE)
_TOOL_BLOCK_PATTERN = re.compile(
    r"Tool Name:\s*(?P<name>[^\n]+)\s*\n"
    r"Tool Arguments:\s*(?P<arguments>.*?)(?=\nTool Description:|\nTool Name:|\Z)",
    re.DOTALL,
)
_QUESTION_PATTERN = re.compile(
    r"(?:^|\n)(?:Current Task:\s*)?Question:\s*(.+?)(?=\n(?:Thought:|Observation:|Final Answer:)"
    r"|\n\n(?:This is the expected criteria|Begin!)|\Z)",
    re.DOTALL,
)
_ID_ARGUMENT_NAMES = {"appointment_id", "booking_id", "record_id"}
_QUERY_ARGUMENT_NAMES = {"query", "question", "search_query"}
_CLINIC_KEYWORDS = {
    "account",
    "appointment",
    "booking",
    "cancel",
    "cancellation",
    "cardiology",
    "clinic",
    "consultation",
    "dermatology",
    "doctor",
    "emergency",
    "fee",
    "followup",
    "health",
    "insurance",
    "lab",
    "medical",
    "medicine",
    "orthopedics",
    "patient",
    "pediatrics",
    "prescription",
    "privacy",
    "refill",
    "reschedule",
    "telemedicine",
    "test",
    "visit",
}
_CLINIC_PHRASES = {"follow-up", "home visit", "second opinion"}
_STATUS_WORDS = {
    "status",
    "scheduled",
    "completed",
    "cancelled",
    "canceled",
    "rescheduled",
}


class MockLLM(BaseLLM):
    """Produce predictable ReAct tool calls and summaries without network access."""

    def __init__(
        self, model: str = "MOCK_LLM", temperature: float = 0.0
    ) -> None:
        super().__init__(model=model, temperature=temperature)

    def supports_function_calling(self) -> bool:
        """Use CrewAI's text-based ReAct tool protocol, not native function calls."""
        return False

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
    ) -> str:
        del callbacks, available_functions
        prompt = _message_text(messages)
        query = _latest_user_text(messages)
        schemas = _tool_schemas(tools, prompt)
        tool_output = _latest_tool_output(
            messages, {name for name, _ in schemas}
        )
        if tool_output is not None:
            return _final_answer(
                f"Summary of the latest tool output: {_to_text(tool_output)}"
            )

        selected = _select_tool(query, schemas)
        if selected is not None:
            tool_name, arguments = selected
            return (
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(arguments, sort_keys=True)}"
            )

        return _final_answer("No tool output is available to summarize.")


def _message_text(messages: str | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(
        _to_text(message.get("content", ""))
        for message in messages
        if isinstance(message, Mapping)
    )


def _latest_user_text(messages: str | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(messages, str):
        question_matches = _QUESTION_PATTERN.findall(messages)
        return question_matches[-1].strip() if question_matches else messages.strip()

    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = _to_text(message.get("content", "")).strip()
            question_matches = _QUESTION_PATTERN.findall(content)
            return question_matches[-1].strip() if question_matches else content
    return ""


def _tool_schemas(
    tools: list[dict[str, Any]] | None, prompt: str
) -> list[tuple[str, dict[str, Any]]]:
    schemas: list[tuple[str, dict[str, Any]]] = []
    for tool in tools or []:
        schema = _schema_from_tool(tool)
        if schema is not None:
            schemas.append(schema)

    known_names = {name for name, _ in schemas}
    for match in _TOOL_BLOCK_PATTERN.finditer(prompt):
        name = match.group("name").strip()
        if name in known_names:
            continue
        arguments = _parse_argument_schema(match.group("arguments"))
        if arguments is not None:
            schemas.append((name, arguments))
            known_names.add(name)
    return schemas


def _schema_from_tool(tool: Any) -> tuple[str, dict[str, Any]] | None:
    if isinstance(tool, Mapping):
        function = tool.get("function", tool)
        if not isinstance(function, Mapping):
            return None
        name = function.get("name")
        parameters = function.get("parameters", function.get("args_schema"))
    else:
        name = getattr(tool, "name", None)
        parameters = getattr(tool, "args_schema", None)

    if not isinstance(name, str) or not name:
        return None

    if hasattr(parameters, "model_json_schema"):
        parameters = parameters.model_json_schema()
    elif hasattr(parameters, "schema"):
        parameters = parameters.schema()

    if isinstance(parameters, Mapping):
        properties = parameters.get("properties", parameters)
        if isinstance(properties, Mapping):
            return name, {
                "properties": dict(properties),
                "required": list(parameters.get("required", [])),
            }
    return None


def _parse_argument_schema(text: str) -> dict[str, Any] | None:
    value = text.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, Mapping):
            properties = parsed.get("properties", parsed)
            if isinstance(properties, Mapping):
                return {
                    "properties": dict(properties),
                    "required": list(parsed.get("required", [])),
                }
    return None


def _select_tool(
    query: str, schemas: list[tuple[str, dict[str, Any]]]
) -> tuple[str, dict[str, str]] | None:
    normalized_query = query.lower()
    appointment_id_match = _APPOINTMENT_ID_PATTERN.search(query)
    id_requested = appointment_id_match is not None or (
        "appointment" in normalized_query
        and any(word in normalized_query.split() for word in _STATUS_WORDS)
    )

    if id_requested:
        for name, schema in schemas:
            properties = schema["properties"]
            id_argument = next(
                (key for key in properties if key.lower() in _ID_ARGUMENT_NAMES),
                None,
            )
            if id_argument is not None and appointment_id_match is not None:
                return name, {id_argument: appointment_id_match.group(0)}
        if appointment_id_match is None:
            return None

    query_words = set(re.findall(r"[a-z]+", normalized_query))
    if not query_words.intersection(_CLINIC_KEYWORDS) and not any(
        phrase in normalized_query for phrase in _CLINIC_PHRASES
    ):
        return None

    for name, schema in schemas:
        properties = schema["properties"]
        query_argument = next(
            (key for key in properties if key.lower() in _QUERY_ARGUMENT_NAMES),
            None,
        )
        if query_argument is None:
            continue

        required = schema["required"]
        if all(argument == query_argument for argument in required):
            return name, {query_argument: query}
    return None


def _latest_tool_output(
    messages: str | Sequence[Mapping[str, Any]], tool_names: set[str]
) -> Any | None:
    names_pattern = "|".join(re.escape(name) for name in sorted(tool_names))
    if isinstance(messages, str):
        if not tool_names:
            return None
        actions = list(
            re.finditer(
                rf"(?m)^Action:\s*(?:{names_pattern})\s*$",
                messages,
            )
        )
        for action in reversed(actions):
            action_input = re.search(
                r"(?m)^Action Input:\s*.*$", messages[action.end() :]
            )
            if action_input is None:
                continue
            observation_start = action.end() + action_input.end()
            observations = re.findall(
                r"(?:^|\n)Observation:\s*(.*?)(?=\n(?:Thought:|Action:|Final Answer:)|\Z)",
                messages[observation_start:],
                flags=re.DOTALL,
            )
            if observations:
                return observations[-1].strip()
        return None

    normalized = [message for message in messages if isinstance(message, Mapping)]
    last_action_index = max(
        (
            index
            for index, message in enumerate(normalized)
            if message.get("role") == "assistant"
            and re.search(r"(?:^|\n)Action:\s*\S+", _to_text(message.get("content", "")))
        ),
        default=-1,
    )
    if last_action_index < 0:
        for message in reversed(normalized):
            if message.get("role") in {"tool", "function"}:
                return message.get("content", "")
        for message in reversed(normalized):
            if message.get("role") != "user":
                continue
            summaries = re.findall(
                r"Summary of the latest tool output:\s*(.*?)(?=\n(?:Summary of the latest tool output:|Task output:|Thought:)|\n\nBegin!|\Z)",
                _to_text(message.get("content", "")),
                flags=re.DOTALL,
            )
            if summaries:
                return summaries[-1].strip()
        return None

    if last_action_index >= 0:
        action_message = _to_text(
            normalized[last_action_index].get("content", "")
        )
        action_match = re.search(
            rf"(?m)^Action:\s*(?:{names_pattern})\s*$",
            action_message,
        )
        if action_match is not None:
            action_input = re.search(
                r"(?m)^Action Input:\s*.*$",
                action_message[action_match.end() :],
            )
            if action_input is not None:
                observation_start = (
                    action_match.end() + action_input.end()
                )
                observations = re.findall(
                    r"(?:^|\n)Observation:\s*(.*?)(?=\n(?:Thought:|Action:|Final Answer:|Now it's time)|\Z)",
                    action_message[observation_start:],
                    flags=re.DOTALL,
                )
                if observations:
                    return observations[-1].strip()

        for message in reversed(normalized[last_action_index + 1 :]):
            if message.get("role") in {"tool", "function"}:
                return message.get("content", "")

        for message in reversed(normalized[:last_action_index]):
            if message.get("role") != "assistant":
                continue
            content = _to_text(message.get("content", ""))
            if not re.search(
                rf"(?m)^Action:\s*(?:{names_pattern})\s*$",
                content,
            ):
                return content

    for message in reversed(normalized):
        if message.get("role") == "user":
            content = _to_text(message.get("content", ""))
            observations = re.findall(
                r"(?:^|\n)Observation:\s*(.*?)(?=\n(?:Thought:|Action:|Final Answer:|Now it's time)|\Z)",
                content,
                flags=re.DOTALL,
            )
            if observations:
                return observations[-1].strip()
    return None


def _final_answer(text: str) -> str:
    return f"Thought: I now know the final answer\nFinal Answer: {text}"


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) or isinstance(value, list):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)
