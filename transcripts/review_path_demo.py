"""Verify AutoGen review runs on drafts from the live crew request path."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import agents.crew as crew_module
from governance.cache import reset_cache


def main() -> None:
    for index, (query, inject_ungrounded) in enumerate(
        (
            ("How much is a cardiology consultation?", False),
            ("Can I cancel my appointment?", True),
        )
    ):
        if index:
            print()
        reset_cache()
        if inject_ungrounded:
            os.environ["GUARDRAIL_TEST_INJECT_UNGROUNDED"] = "1"
        else:
            os.environ.pop("GUARDRAIL_TEST_INJECT_UNGROUNDED", None)

        response = crew_module.kickoff_validated(
            crew_module.build_crew(),
            query,
        )
        verdict = crew_module.LAST_REVIEW_VERDICT
        print(f"Query: {query}")
        if verdict is None:
            print("Review verdict: skipped after groundedness refusal")
        else:
            print(f"Review verdict:\n{verdict.model_dump_json(indent=2)}")
        print(f"AgentResponse: {response.model_dump(mode='json')}")

    os.environ.pop("GUARDRAIL_TEST_INJECT_UNGROUNDED", None)


if __name__ == "__main__":
    main()
