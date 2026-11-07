"""PII masking, prompt-injection detection, and answer grounding checks."""

import re

_INDIAN_MOBILE_PATTERN = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")
_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(?:your\s+)?instructions\b", re.IGNORECASE),
)
_CONTENT_WORD_PATTERN = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_NON_CONTENT_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Mask Indian mobile numbers without attempting to identify other PII."""
    masked_text, replacements = _INDIAN_MOBILE_PATTERN.subn(
        "[PHONE_MASKED]", text
    )
    flags = ["phone"] if replacements else []
    return masked_text, flags


def detect_injection(text: str) -> bool:
    """Detect common attempts to override or expose system instructions."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def is_grounded(answer: str, context: str) -> bool:
    """Require overall and sentence-level answer overlap of at least 40%."""
    answer_words = set(_content_words(answer))
    if not answer_words:
        return True

    context_words = set(_content_words(context))
    if len(answer_words & context_words) / len(answer_words) < 0.4:
        return False

    return all(
        len(sentence_words & context_words) / len(sentence_words) >= 0.4
        for sentence in _SENTENCE_PATTERN.split(answer)
        if (sentence_words := set(_content_words(sentence)))
    )


def _content_words(text: str) -> list[str]:
    return [
        word
        for word in _CONTENT_WORD_PATTERN.findall(text.lower())
        if word not in _NON_CONTENT_WORDS
    ]
