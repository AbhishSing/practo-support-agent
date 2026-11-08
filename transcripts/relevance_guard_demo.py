"""Check the lexical relevance heuristic on six held-out queries."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from agents.guardrails import has_relevant_context
from agents.tools import RAG_SIMILARITY_THRESHOLD
from rag.retrieve import FALLBACK_ANSWER, answer_with_context

COLLECTION_NAME = "kb_fixed"
CASES = (
    ("How early can I book a visit?", True),
    ("Is there a fee for a second opinion?", True),
    ("What happens if I cancel late?", True),
    ("What is the weather in Paris?", False),
    ("Who won the football match?", False),
    ("Recommend a laptop for coding.", False),
)


def main() -> None:
    matches = 0
    for query, expected in CASES:
        answer, _sources, context = answer_with_context(
            query,
            COLLECTION_NAME,
            RAG_SIMILARITY_THRESHOLD,
        )
        actual = has_relevant_context(query, context)
        matched = actual == expected
        matches += int(matched)
        print(f"Query: {query}")
        print(f"Expected relevant: {expected}")
        print(f"Has relevant context: {actual}")
        if answer == FALLBACK_ANSWER and not context:
            print("Fallback returned empty context; has_relevant_context is False.")
        print(f"Expected result matched: {matched}")
        print()
    print(f"Held-out relevance matches: {matches}/{len(CASES)}")


if __name__ == "__main__":
    main()
