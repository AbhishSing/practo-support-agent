"""Evaluate document-level retrieval precision and recall per collection."""

from collections.abc import Sequence
from pathlib import Path

from eval_queries import EVAL_QUERIES
from rag.retrieve import retrieve
from relevance_labels import RELEVANCE_LABELS

KB_DIR = Path(__file__).resolve().parents[1] / "kb"
COLLECTION_NAMES = ("kb_fixed", "kb_sentence")
RESULTS_PER_QUERY = 5


def _evaluate_query(
    query: str, collection_name: str
) -> tuple[int, int, int, int]:
    retrieved_chunks = retrieve(query, collection_name, k=RESULTS_PER_QUERY)
    retrieved_doc_ids = list(dict.fromkeys(doc_id for doc_id, _, _ in retrieved_chunks))
    relevant_doc_ids = RELEVANCE_LABELS[query]
    relevant_retrieved = len(set(retrieved_doc_ids) & relevant_doc_ids)
    retrieved_count = len(retrieved_doc_ids)
    relevant_count = len(relevant_doc_ids)
    return relevant_retrieved, retrieved_count, relevant_retrieved, relevant_count


def _average(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def evaluate() -> None:
    scores_by_collection: dict[str, tuple[list[float], list[float]]] = {}
    missing_labels = sorted(
        set().union(*(RELEVANCE_LABELS[query] for query in EVAL_QUERIES))
        - {path.name for path in KB_DIR.glob("*.md")}
    )
    if missing_labels:
        print(
            "Warning: labelled relevant documents are missing from kb/: "
            + ", ".join(missing_labels)
        )

    for collection_name in COLLECTION_NAMES:
        precisions = []
        recalls = []
        print(f"\n{collection_name}")

        for query in EVAL_QUERIES:
            precision_num, precision_den, recall_num, recall_den = _evaluate_query(
                query, collection_name
            )
            precision = precision_num / precision_den if precision_den else 0.0
            recall = recall_num / recall_den if recall_den else 0.0
            precisions.append(precision)
            recalls.append(recall)
            print(f"  Query: {query}")
            print(
                f"    Precision: {precision_num}/{precision_den} = {precision:.3f}"
            )
            print(f"    Recall:    {recall_num}/{recall_den} = {recall:.3f}")

        average_precision = _average(precisions)
        average_recall = _average(recalls)
        scores_by_collection[collection_name] = (precisions, recalls)
        print(f"  Average precision (macro): {average_precision:.3f}")
        print(f"  Average recall (macro):    {average_recall:.3f}")

    fixed_precision = _average(scores_by_collection["kb_fixed"][0])
    fixed_recall = _average(scores_by_collection["kb_fixed"][1])
    sentence_precision = _average(scores_by_collection["kb_sentence"][0])
    sentence_recall = _average(scores_by_collection["kb_sentence"][1])
    print("\nComparison:")
    print(
        f"  kb_fixed has macro precision {fixed_precision:.3f} and recall "
        f"{fixed_recall:.3f}; kb_sentence has macro precision "
        f"{sentence_precision:.3f} and recall {sentence_recall:.3f}."
    )
    print(
        "  "
        + (
            "kb_fixed has higher precision."
            if fixed_precision > sentence_precision
            else "kb_sentence has higher precision."
            if sentence_precision > fixed_precision
            else "Both collections have equal precision."
        )
        + " "
        + (
            "kb_fixed has higher recall."
            if fixed_recall > sentence_recall
            else "kb_sentence has higher recall."
            if sentence_recall > fixed_recall
            else "Both collections have equal recall."
        )
    )


if __name__ == "__main__":
    evaluate()
