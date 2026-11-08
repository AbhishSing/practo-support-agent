"""Verify approval, revision, and groundedness refusal on the live crew path."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import agents.crew as crew_module
from agents.guardrails import is_grounded
from agents.tools import RAG_COLLECTION_NAME, RAG_SIMILARITY_THRESHOLD
from governance.cache import reset_cache
from rag.retrieve import answer_with_context


def _set_hooks(*, ungrounded: bool = False, review_claim: bool = False) -> None:
    if ungrounded:
        os.environ["GUARDRAIL_TEST_INJECT_UNGROUNDED"] = "1"
    else:
        os.environ.pop("GUARDRAIL_TEST_INJECT_UNGROUNDED", None)
    if review_claim:
        os.environ["GUARDRAIL_TEST_INJECT_REVIEW_CLAIM"] = "1"
    else:
        os.environ.pop("GUARDRAIL_TEST_INJECT_REVIEW_CLAIM", None)


def _print_answer(answer: str) -> None:
    print("\n".join(line.rstrip() for line in answer.splitlines()))


def main() -> None:
    query = "How much is a cardiology consultation?"
    reset_cache()
    _set_hooks()
    response = crew_module.kickoff_validated(crew_module.build_crew(), query)
    verdict = crew_module.LAST_REVIEW_VERDICT
    assert verdict is not None and verdict.approved
    print("Case A: ordinary supported policy question")
    print(f"Review approved: {verdict.approved}")
    print("Final answer:")
    _print_answer(response.answer)

    reset_cache()
    _set_hooks(review_claim=True)
    retrieved_answer, _sources, context = answer_with_context(
        query,
        RAG_COLLECTION_NAME,
        RAG_SIMILARITY_THRESHOLD,
    )
    review_claim_draft = (
        f"{retrieved_answer.rstrip().rstrip('.')}."
        f" {crew_module.REVIEW_CLAIM_TEST_SENTENCE}"
    )
    grounded = is_grounded(review_claim_draft, context)
    print("\nCase B: review-claim test hook")
    print(f"Review claim draft: {crew_module.REVIEW_CLAIM_TEST_SENTENCE}")
    print(f"is_grounded(draft, retrieved context): {grounded}")
    if not grounded:
        raise AssertionError("Review-claim test draft did not pass groundedness.")
    response = crew_module.kickoff_validated(crew_module.build_crew(), query)
    verdict = crew_module.LAST_REVIEW_VERDICT
    assert verdict is not None and not verdict.approved
    assert "sundays" not in response.answer.casefold()
    print(f"Review approved: {verdict.approved}")
    print(f"Verdict reason: {verdict.reason}")
    print("Final answer:")
    _print_answer(response.answer)

    reset_cache()
    _set_hooks(ungrounded=True)
    query = "Can I cancel my appointment?"
    response = crew_module.kickoff_validated(crew_module.build_crew(), query)
    assert crew_module.LAST_REVIEW_VERDICT is None
    assert response.answer == (
        "I don't have that information in the clinic policy documents."
    )
    print("\nCase C: deliberately ungrounded test hook")
    print(f"Review skipped: {crew_module.LAST_REVIEW_VERDICT is None}")
    print(f"Groundedness refusal: {response.answer}")

    _set_hooks()


if __name__ == "__main__":
    main()
