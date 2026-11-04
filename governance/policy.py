"""Governance guards.

The risk category is High because the system handles patient appointment and
clinical-policy data; the specification places medical data in the High category.
"""

from collections.abc import Mapping, Sequence
from typing import Any

TOKEN_BUDGET = 1000
RISK_LEVEL = "High"
APPOINTMENT_TOOL_NAME = "check_appointment_status"


class BudgetExceeded(ValueError):
    """Raised when an input query exceeds the configured token budget."""


def check_budget(query: str) -> float:
    """Estimate query tokens as 1.3 times its whitespace-delimited word count."""
    estimated_tokens = len(query.split()) * 1.3
    if estimated_tokens > TOKEN_BUDGET:
        raise BudgetExceeded(
            f"Estimated query size ({estimated_tokens:.1f} tokens) exceeds "
            f"the {TOKEN_BUDGET}-token budget."
        )
    return estimated_tokens


def test_lookup_agent_is_only_agent_with_appointment_tool(
    agent_tools: Mapping[str, Sequence[Any]],
) -> None:
    """Assert the appointment lookup tool is available to Lookup and nobody else."""
    tool_owners = {
        agent_name: {
            getattr(tool, "name", None)
            for tool in tools
        }
        for agent_name, tools in agent_tools.items()
    }

    assert "Lookup" in tool_owners, "The crew must contain a Lookup agent."
    assert APPOINTMENT_TOOL_NAME in tool_owners["Lookup"], (
        "The Lookup agent must have the appointment-status tool."
    )
    # Least autonomy: grant each agent only the capabilities needed for its role.
    for agent_name, tool_names in tool_owners.items():
        if agent_name != "Lookup":
            assert APPOINTMENT_TOOL_NAME not in tool_names, (
                f"{agent_name} must not have {APPOINTMENT_TOOL_NAME}."
            )


def run_budget_demo() -> None:
    """Demonstrate rejection of a query that exceeds the token budget."""
    oversized_query = "appointment " * (TOKEN_BUDGET + 1)
    try:
        check_budget(oversized_query)
    except BudgetExceeded as error:
        print(f"Query rejected: {error}")
    else:
        raise AssertionError("The oversized query unexpectedly passed the budget check.")


if __name__ == "__main__":
    run_budget_demo()
