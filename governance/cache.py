"""Thread-safe in-memory cache for normalized grounded-generation queries."""

import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

CACHE: dict[str, Any] = {}
CALL_COUNT = 0
_CACHE_LOCK = threading.RLock()


def normalize_query(query: str) -> str:
    """Normalize casing and whitespace for cache lookup."""
    return " ".join(query.lower().split())


def cached_call(query: str, generate: Callable[[], T]) -> T:
    """Return a cached result or call and cache the supplied generator once."""
    global CALL_COUNT
    key = normalize_query(query)
    with _CACHE_LOCK:
        if key in CACHE:
            return CACHE[key]

        result = generate()
        CACHE[key] = result
        CALL_COUNT += 1
        return result


def get_call_count() -> int:
    """Return the number of cache-miss generation calls."""
    with _CACHE_LOCK:
        return CALL_COUNT


def reset_cache() -> None:
    """Clear cached results and reset the generation-call counter."""
    global CALL_COUNT
    with _CACHE_LOCK:
        CACHE.clear()
        CALL_COUNT = 0


def run_demo() -> None:
    """Show that an identical normalized query generates only once."""
    reset_cache()
    query = "What is my appointment status?"
    print(f"Counter before: {get_call_count()}")
    first_result = cached_call(query, lambda: "Your appointment is scheduled.")
    second_result = cached_call(
        "  WHAT is my appointment   status?  ",
        lambda: "This should not be generated.",
    )
    print(f"Counter after: {get_call_count()}")
    print(f"Results match: {first_result == second_result}")


if __name__ == "__main__":
    run_demo()
