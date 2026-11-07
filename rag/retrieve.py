"""Retrieve relevant knowledge-base chunks from a ChromaDB collection."""

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = str(Path(__file__).resolve().parents[1] / "chroma_db")
FALLBACK_ANSWER = "I don't have that information in the clinic policy documents."


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def retrieve(
    query: str, collection_name: str, k: int = 3
) -> list[tuple[str, str, float]]:
    """Return matching document IDs, chunk text, and cosine similarities."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    collection = _get_client().get_collection(name=collection_name)
    if collection.count() == 0:
        return []

    query_embedding = _get_model().encode([query], show_progress_bar=False)[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        (metadata["doc_id"], text, 1.0 - distance)
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def answer_with_context(
    query: str, collection_name: str, threshold: float
) -> tuple[str, list[str], str]:
    """Return an answer with its unique source IDs and retrieved context."""
    chunks = retrieve(query, collection_name)
    if not chunks or chunks[0][2] < threshold:
        return FALLBACK_ANSWER, [], ""

    sources = list(dict.fromkeys(doc_id for doc_id, _, _ in chunks))
    context = "\n\n".join(text for _, text, _ in chunks)
    return context, sources, context


def answer(query: str, collection_name: str, threshold: float) -> str:
    """Return retrieved chunk text or the policy-document fallback."""
    answer_text, _, _ = answer_with_context(query, collection_name, threshold)
    return answer_text
