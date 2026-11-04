"""Print similarity scores for sample in-scope and out-of-scope queries."""

from collections.abc import Sequence

from rag.index import COLLECTIONS
from rag.retrieve import retrieve

IN_SCOPE_QUERIES = (
    "How much is a cardiology consultation?",
    "Can I cancel my appointment?",
    "How long do lab results take?",
)
OUT_OF_SCOPE_QUERIES = (
    "What is the capital of France?",
    "Recommend a good laptop.",
)


def _top_score(query: str, collection_name: str) -> float | None:
    results = retrieve(query, collection_name, k=1)
    return results[0][2] if results else None


def _print_cluster(
    label: str, queries: Sequence[str], scores: Sequence[float | None]
) -> None:
    print(f"{label} top-1 similarities:")
    for query, score in zip(queries, scores):
        formatted_score = f"{score:.4f}" if score is not None else "unavailable"
        print(f"  {query} -> {formatted_score}")


def calibrate() -> None:
    for collection_name in COLLECTIONS.values():
        print(f"\n{collection_name}")
        in_scope_scores = [
            _top_score(query, collection_name) for query in IN_SCOPE_QUERIES
        ]
        out_of_scope_scores = [
            _top_score(query, collection_name) for query in OUT_OF_SCOPE_QUERIES
        ]
        _print_cluster("In-scope", IN_SCOPE_QUERIES, in_scope_scores)
        _print_cluster("Out-of-scope", OUT_OF_SCOPE_QUERIES, out_of_scope_scores)

        scores = in_scope_scores + out_of_scope_scores
        if all(score is not None for score in scores):
            in_scope_values = [score for score in in_scope_scores if score is not None]
            out_of_scope_values = [
                score for score in out_of_scope_scores if score is not None
            ]
            gap = min(in_scope_values) - max(out_of_scope_values)
            print(f"Cluster gap (min in-scope - max out-of-scope): {gap:.4f}")
        else:
            print("Cluster gap: unavailable (index is missing or empty)")


if __name__ == "__main__":
    calibrate()
