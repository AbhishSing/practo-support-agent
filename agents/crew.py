"""Sequential Retrieval, Lookup, and Composer crew with deterministic demos."""

import asyncio
import os
import re

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from crewai import Agent, Crew, Process, Task

from agents.guardrails import (
    detect_injection,
    has_relevant_context,
    is_grounded,
    mask_pii,
)
from agents.tools import (
    DAYS_SINCE_CREATED_P80,
    LAST_APPOINTMENT_LOOKUP,
    LAST_RETRIEVAL,
    TOOL_CALLS,
    check_appointment_status,
    rag_lookup,
    reset_tool_call_log,
)
from agents.schemas import AgentResponse, validate_agent_response
from governance.cache import cached_call, normalize_query
from governance.policy import (
    check_budget,
    test_lookup_agent_is_only_agent_with_appointment_tool,
)
from llm.mock_llm import MockLLM

DEMO_QUERIES = (
    ("policy question", "Can I cancel my appointment?"),
    ("appointment status", "What is the status of APT-0007?"),
)
LAST_REVIEW_VERDICT: object | None = None


def build_crew() -> Crew:
    """Create the three-agent sequential support crew."""
    llm = MockLLM()
    retrieval_agent = Agent(
        role="Retrieval",
        goal="Retrieve clinic policy evidence only when the query concerns a policy.",
        backstory="You use clinic policy sources and do not look up appointments.",
        tools=[rag_lookup],
        llm=llm,
        verbose=False,
    )
    lookup_agent = Agent(
        role="Lookup",
        goal="Look up appointment records only when the user asks about a specific appointment.",
        backstory="You check appointment records and do not retrieve clinic policies.",
        tools=[check_appointment_status],
        llm=llm,
        verbose=False,
    )
    composer_agent = Agent(
        role="Composer",
        goal="Combine the useful outputs from retrieval and appointment lookup into a concise response.",
        backstory="You only use the outputs of the preceding agents and have no tools.",
        llm=llm,
        verbose=False,
    )
    test_lookup_agent_is_only_agent_with_appointment_tool(
        {
            "Retrieval": retrieval_agent.tools,
            "Lookup": lookup_agent.tools,
            "Composer": composer_agent.tools,
        }
    )

    retrieval_task = Task(
        description="Question: {query}",
        expected_output="The relevant policy evidence, or that no policy lookup was needed.",
        agent=retrieval_agent,
    )
    lookup_task = Task(
        description="Question: {query}",
        expected_output="The requested appointment details, or that no appointment lookup was needed.",
        agent=lookup_agent,
    )
    composer_task = Task(
        description="Question: {query}",
        expected_output="A concise response combining the applicable preceding task outputs.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
    )
    return Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, composer_task],
        process=Process.sequential,
        verbose=False,
    )


def kickoff_validated(crew: Crew, query: str) -> AgentResponse:
    """Run the crew and validate its response against the API response schema."""
    masked_query, _flags = mask_pii(query)
    if detect_injection(masked_query):
        return validate_agent_response(
            {
                "answer": "I can't process that request.",
                "sources": [],
                "tool_used": [],
                "escalation_score": None,
                "grounded": False,
            }
        )

    check_budget(masked_query)
    cache_key = (
        f"{normalize_query(masked_query)}|hook="
        f"{os.getenv('GUARDRAIL_TEST_INJECT_UNGROUNDED', '0')}"
    )
    return cached_call(
        cache_key,
        lambda: _kickoff_validated(crew, masked_query),
    )


def _kickoff_validated(crew: Crew, query: str) -> AgentResponse:
    global LAST_REVIEW_VERDICT
    LAST_REVIEW_VERDICT = None
    reset_tool_call_log()
    result = crew.kickoff(inputs={"query": query})
    raw_answer = getattr(result, "raw", result)
    answer = raw_answer if isinstance(raw_answer, str) else str(raw_answer)
    answer = re.sub(
        r"^\s*Summary of the latest tool output:\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    appointment_record = ""
    if LAST_APPOINTMENT_LOOKUP:
        record_id_match = re.search(r"\bAPT-\d+\b", query, re.IGNORECASE)
        record_id = record_id_match.group(0).upper() if record_id_match else ""
        appointment_record = (
            f"Appointment record: {record_id} "
            f"status {LAST_APPOINTMENT_LOOKUP['status']}; "
            "consultation fee "
            f"{LAST_APPOINTMENT_LOOKUP['consultation_fee_inr']} INR; "
            "escalation score "
            f"{LAST_APPOINTMENT_LOOKUP['escalation_score']}."
        )
        answer_record_id = record_id or "The appointment"
        answer = (
            f"{answer_record_id} is {LAST_APPOINTMENT_LOOKUP['status']}. "
            "The consultation fee is "
            f"₹{LAST_APPOINTMENT_LOOKUP['consultation_fee_inr']}."
        )
        if "rag_lookup" in TOOL_CALLS and LAST_RETRIEVAL["context"]:
            answer = f"{answer}\n\n{LAST_RETRIEVAL['context']}"
    elif "rag_lookup" in TOOL_CALLS and LAST_RETRIEVAL["context"]:
        answer = str(LAST_RETRIEVAL["context"])

    if os.getenv("GUARDRAIL_TEST_INJECT_UNGROUNDED") == "1":
        answer = (
            f"{answer.rstrip().rstrip('.')}."
            " Consultations are free on Sundays."
        )

    draft = answer
    policy_context = str(LAST_RETRIEVAL["context"])
    grounding_context = "\n".join(
        part for part in (policy_context, appointment_record) if part
    )
    grounded = is_grounded(draft, grounding_context) and (
        "rag_lookup" not in TOOL_CALLS
        or has_relevant_context(query, grounding_context)
    )
    if not grounded:
        answer = "I don't have that information in the clinic policy documents."
    else:
        from review.autogen_review import review_draft

        verdict = asyncio.run(review_draft(draft, grounding_context))
        LAST_REVIEW_VERDICT = verdict
        answer = verdict.final_answer if not verdict.approved else draft

    response_data = {
        "answer": answer,
        "sources": list(LAST_RETRIEVAL["sources"]),
        "tool_used": list(TOOL_CALLS),
        "escalation_score": (
            LAST_APPOINTMENT_LOOKUP["escalation_score"]
            if LAST_APPOINTMENT_LOOKUP
            else None
        ),
        "grounded": grounded,
    }
    return validate_agent_response(response_data)


def run_demo() -> None:
    """Run a policy query and appointment-status query, reporting called tools."""
    print(
        "80th percentile of days_since_created "
        f"(inclusive): {DAYS_SINCE_CREATED_P80:.1f} days"
    )
    for label, query in DEMO_QUERIES:
        reset_tool_call_log()
        result = kickoff_validated(build_crew(), query)
        called_tools = ", ".join(TOOL_CALLS) if TOOL_CALLS else "none"
        print(f"\nRun: {label} — {query}")
        print(f"Tools called: {called_tools}")
        print(f"Response validated: {isinstance(result, AgentResponse)}")
        print(f"Response: {result}")


if __name__ == "__main__":
    run_demo()
