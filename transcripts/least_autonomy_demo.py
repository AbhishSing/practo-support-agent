"""Verify that only the Lookup agent receives appointment-status access."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.crew import build_crew
from agents.tools import check_appointment_status
from governance.policy import test_lookup_agent_is_only_agent_with_appointment_tool


def main() -> None:
    crew = build_crew()
    agent_tools = {agent.role: list(agent.tools) for agent in crew.agents}
    print(
        "Lookup has check_appointment_status:",
        any(
            tool.name == "check_appointment_status"
            for tool in agent_tools["Lookup"]
        ),
    )

    modified_tools = {
        agent_name: list(tools)
        for agent_name, tools in agent_tools.items()
    }
    modified_tools["Composer"].append(check_appointment_status)
    try:
        test_lookup_agent_is_only_agent_with_appointment_tool(modified_tools)
    except AssertionError as error:
        print(f"Blocked as expected: {error}")
    else:
        raise AssertionError("The invalid Composer tool assignment was accepted.")


if __name__ == "__main__":
    main()
