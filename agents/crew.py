"""Sequential Retrieval, Lookup, and Composer crew with deterministic demos."""

import asyncio
import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from crewai import Agent, Crew, Process, Task

from agents.guardrails import detect_injection, is_grounded, mask_pii
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
from governance.cache import cached_call
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
    return cached_call(
        masked_query,
        lambda: _kickoff_validated(crew, masked_query),
    )


def _kickoff_validated(crew: Crew, query: str) -> AgentResponse:
    global LAST_REVIEW_VERDICT
    reset_tool_call_log()
    result = crew.kickoff(inputs={"query": query})
    raw_answer = getattr(result, "raw", result)
    answer = raw_answer if isinstance(raw_answer, str) else str(raw_answer)
    if os.getenv("GUARDRAIL_TEST_INJECT_UNGROUNDED") == "1":
        answer = (
            f"{answer.rstrip().rstrip('.')}."
            " Consultations are free on Sundays."
        )

    grounded = is_grounded(answer, str(LAST_RETRIEVAL["context"]))
    draft = answer
    guarded_answer = (
        draft
        if grounded
        else "I don't have that information in the clinic policy documents."
    )

    from review.autogen_review import review_draft

    verdict = asyncio.run(
        review_draft(draft, str(LAST_RETRIEVAL["context"]))
    )
    LAST_REVIEW_VERDICT = verdict
    answer = verdict.final_answer if not verdict.approved else guarded_answer

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
        print(f"Response: {result}")


if __name__ == "__main__":
    run_demo()
