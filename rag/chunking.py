"""Text chunking strategies used to build the knowledge-base index."""

import re


def fixed_chunks(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character chunks with a fixed overlap."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between zero and size - 1")

    text = text.strip()
    if not text:
        return []

    step = size - overlap
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return chunks


def sentence_chunks(text: str, max_sentences: int = 3) -> list[str]:
    """Group sentences into chunks containing at most ``max_sentences``."""
    if max_sentences <= 0:
        raise ValueError("max_sentences must be greater than zero")

    text = text.strip()
    if not text:
        return []

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)]
    sentences = [sentence for sentence in sentences if sentence]
    return [
        " ".join(sentences[start : start + max_sentences])
        for start in range(0, len(sentences), max_sentences)
    ]
