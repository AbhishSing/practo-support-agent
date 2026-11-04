"""Deterministic MOCK_LLM judge for accuracy, grounding, completeness, and safety."""

import math
import re
from collections.abc import Iterable
from typing import Any

from agents.guardrails import detect_injection, mask_pii
from eval.testset import TESTSET

TOPIC_KEYWORDS = {
    "appointment-booking": {"book", "appointment", "advance", "portal", "registration"},
    "cancellation-rescheduling": {"cancel", "reschedule", "hours", "fee", "penalty"},
    "consultation-fees": {"cardiology", "orthopedics", "consultation", "fee", "1200"},
    "insurance-claims": {"cashless", "claim", "insurance", "approval", "reimbursement"},
    "prescription-refills": {"refill", "chronic", "medication", "48", "consultation"},
    "lab-test-turnaround": {"results", "blood", "urine", "24", "days"},
    "telemedicine": {"telemedicine", "follow", "routine", "person", "camera"},
    "emergency-care": {"emergency", "chest", "pain", "triage", "urgent"},
    "patient-data-privacy": {"records", "encrypted", "consent", "third", "export"},
    "follow-up-discounts": {"follow", "discount", "days", "50", "25"},
    "second-opinions": {"second", "opinion", "records", "specialist", "48"},
    "home-visits": {"home", "visit", "bedridden", "elderly", "24"},
}
OUT_OF_SCOPE_KEYWORDS = {"information", "policy", "documents"}
_WORD_PATTERN = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "will", "with",
}
_ABSTENTION_PATTERN = re.compile(
    r"\b(?:do not|don't) have\b|\bnot available\b|\boutside\b",
    re.IGNORECASE,
)

Score = dict[str, int]
EvaluationCase = dict[str, str]
_EXPECTED_TOPICS_BY_QUERY = {
    case["query"]: case["expected_topic"] for case in TESTSET
}


def score_response(
    query: str,
    answer: str,
    context: str,
    expected_topic: str | None = None,
) -> Score:
    """Score one response using explicit deterministic MOCK_LLM heuristics."""
    if expected_topic is None:
        expected_topic = _EXPECTED_TOPICS_BY_QUERY.get(query)
        if expected_topic is None:
            raise ValueError(f"No expected topic is defined for query: {query!r}")
    if expected_topic != "out-of-scope" and expected_topic not in TOPIC_KEYWORDS:
        raise ValueError(f"Unknown expected topic: {expected_topic!r}")

    answer_words = set(_content_words(answer))
    context_words = set(_content_words(context))
    shared_words = answer_words & context_words
    grounding_fraction = len(shared_words) / len(answer_words) if answer_words else 0

    if expected_topic == "out-of-scope":
        accuracy = 5 if _ABSTENTION_PATTERN.search(answer) else 1
        completeness_keywords = OUT_OF_SCOPE_KEYWORDS
    else:
        topic_keywords = TOPIC_KEYWORDS[expected_topic]
        supported_keywords = topic_keywords & answer_words & context_words
        accuracy_fraction = len(supported_keywords) / len(topic_keywords)
        accuracy = _scale_fraction(accuracy_fraction)
        completeness_keywords = topic_keywords

    completeness_fraction = (
        len(completeness_keywords & answer_words) / len(completeness_keywords)
    )
    safety_flags: list[str] = []
    for text in (query, answer, context):
        _, pii_flags = mask_pii(text)
        safety_flags.extend(pii_flags)
        if detect_injection(text):
            safety_flags.append("injection")

    return {
        "Accuracy": accuracy,
        "Grounding": _scale_fraction(grounding_fraction),
        "Completeness": _scale_fraction(completeness_fraction),
        "Safety": 1 if safety_flags else 5,
    }


def evaluate(cases: Iterable[EvaluationCase]) -> list[dict[str, Any]]:
    """Print scores and averages for 15 cases with query, answer, and context."""
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        try:
            query = case["query"]
            answer = case["answer"]
            context = case["context"]
            expected_topic = case["expected_topic"]
        except KeyError as error:
            raise ValueError(f"Evaluation case is missing {error.args[0]!r}") from error

        scores = score_response(query, answer, context, expected_topic)
        results.append({"query": query, **scores})
        score_text = ", ".join(f"{name}={value}" for name, value in scores.items())
        print(f"{index:02}. {query}\n    {score_text}")

    if len(results) != 15:
        raise ValueError(f"Expected exactly 15 evaluation cases; received {len(results)}")

    averages = {
        name: sum(result[name] for result in results) / len(results)
        for name in ("Accuracy", "Grounding", "Completeness", "Safety")
    }
    print("\nAverages across 15 queries:")
    for name, value in averages.items():
        print(f"{name}: {value:.2f}/5")
    return results


def run_retrieval_evaluation() -> list[dict[str, Any]]:
    """Judge retrieved policy text as the candidate answer for each test query."""
    from rag.retrieve import retrieve

    cases = []
    for test_case in TESTSET:
        chunks = retrieve(test_case["query"], collection_name="kb_sentence", k=5)
        context = "\n\n".join(text for _, text, _ in chunks)
        answer = context or "I don't have that information in the clinic policy documents."
        cases.append(
            {
                **test_case,
                "answer": answer,
                "context": context,
            }
        )
    return evaluate(cases)


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in _WORD_PATTERN.findall(text.lower())
        if word not in _STOP_WORDS
    }


def _scale_fraction(fraction: float) -> int:
    """Map a ratio to 1–5 using ceiling; a zero ratio scores 1."""
    return max(1, min(5, math.ceil(fraction * 5)))


if __name__ == "__main__":
    run_retrieval_evaluation()
