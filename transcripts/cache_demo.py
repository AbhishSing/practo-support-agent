"""Verify normalized-query caching on the actual crew request path."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.crew import build_crew, kickoff_validated
from governance.cache import get_call_count, reset_cache


def main() -> None:
    reset_cache()
    started = time.perf_counter()
    first = kickoff_validated(
        build_crew(),
        "Can I cancel my appointment?",
    )
    first_duration = time.perf_counter() - started
    print(f"Generation calls after first query: {get_call_count()}")
    print(f"First call wall-clock seconds: {first_duration:.6f}")

    started = time.perf_counter()
    second = kickoff_validated(
        build_crew(),
        "  CAN I cancel   my appointment?  ",
    )
    second_duration = time.perf_counter() - started
    print(f"Generation calls after normalized repeat: {get_call_count()}")
    print(f"Second call wall-clock seconds: {second_duration:.6f}")
    print(f"Results match: {first == second}")

    os.environ["GUARDRAIL_TEST_INJECT_UNGROUNDED"] = "1"
    hooked = kickoff_validated(
        build_crew(),
        "Can I cancel my appointment?",
    )
    os.environ.pop("GUARDRAIL_TEST_INJECT_UNGROUNDED", None)
    print(f"Generation calls after enabling test hook: {get_call_count()}")
    print(f"Hooked result differs from cached normal result: {hooked != first}")


if __name__ == "__main__":
    main()
