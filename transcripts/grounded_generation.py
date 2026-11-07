"""Demonstrate thresholded retrieval for in-scope and out-of-scope queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.tools import RAG_COLLECTION_NAME, RAG_SIMILARITY_THRESHOLD
from eval_queries import EVAL_QUERIES
from rag.retrieve import FALLBACK_ANSWER, answer_with_context, retrieve

OUT_OF_SCOPE_QUERY = "What is the capital of France?"


def main() -> None:
    for index, query in enumerate((*EVAL_QUERIES, OUT_OF_SCOPE_QUERY)):
        if index:
            print()
        answer, sources, _ = answer_with_context(
            query,
            RAG_COLLECTION_NAME,
            RAG_SIMILARITY_THRESHOLD,
        )
        matches = retrieve(query, RAG_COLLECTION_NAME, k=1)
        similarity = matches[0][2] if matches else None
        print(f"Query: {query}")
        print(f"Top-1 similarity: {similarity}")
        print(f"Fallback fired: {answer == FALLBACK_ANSWER}")
        print(f"Sources: {sources}")
        formatted_answer = "\n".join(
            line.rstrip() for line in answer.splitlines()
        )
        print(f"Answer: {formatted_answer}")


if __name__ == "__main__":
    main()
