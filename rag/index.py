"""Embed KB documents and upsert fixed and sentence chunks into ChromaDB."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import fixed_chunks, sentence_chunks

KB_DIR = Path(__file__).resolve().parents[1] / "kb"
CHROMA_PATH = str(Path(__file__).resolve().parents[1] / "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTIONS = {
    "fixed": "kb_fixed",
    "sentence": "kb_sentence",
}


def load_documents() -> list[tuple[str, str]]:
    """Load Markdown files in stable filename order."""
    return [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(KB_DIR.glob("*.md"))
    ]


def index_knowledge_base() -> None:
    documents = load_documents()
    chunks_by_strategy: dict[str, list[tuple[str, int, str]]] = {
        "fixed": [],
        "sentence": [],
    }

    for doc_id, text in documents:
        for strategy, chunker in (
            ("fixed", fixed_chunks),
            ("sentence", sentence_chunks),
        ):
            chunks_by_strategy[strategy].extend(
                (doc_id, index, chunk)
                for index, chunk in enumerate(chunker(text))
            )

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for strategy, collection_name in COLLECTIONS.items():
        chunks = chunks_by_strategy[strategy]
        client.get_or_create_collection(name=collection_name)
        client.delete_collection(name=collection_name)
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if chunks:
            doc_ids = [doc_id for doc_id, _, _ in chunks]
            chunk_indices = [index for _, index, _ in chunks]
            texts = [chunk for _, _, chunk in chunks]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            collection.upsert(
                ids=[
                    f"{strategy}-{doc_id}-{index}"
                    for doc_id, index, _ in chunks
                ],
                documents=texts,
                metadatas=[
                    {
                        "doc_id": doc_id,
                        "chunk_index": index,
                        "strategy": strategy,
                    }
                    for doc_id, index in zip(doc_ids, chunk_indices)
                ],
                embeddings=embeddings,
            )

        print(f"{collection_name}: {len(chunks)} chunks indexed")


def index_document(doc_id: str, text: str) -> int:
    """Upsert one document's chunks into both collections and return their total."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    total_chunks = 0

    for strategy, collection_name in COLLECTIONS.items():
        chunker = fixed_chunks if strategy == "fixed" else sentence_chunks
        chunks = chunker(text)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if chunks:
            collection.upsert(
                ids=[f"{strategy}-{doc_id}-{index}" for index in range(len(chunks))],
                documents=chunks,
                metadatas=[
                    {
                        "doc_id": doc_id,
                        "chunk_index": index,
                        "strategy": strategy,
                    }
                    for index in range(len(chunks))
                ],
                embeddings=model.encode(chunks, show_progress_bar=False).tolist(),
            )
        total_chunks += len(chunks)

    return total_chunks


if __name__ == "__main__":
    index_knowledge_base()
