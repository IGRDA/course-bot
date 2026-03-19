"""Generic reusable filters — ChainFilters, IdempotencyFilter, MentionFilter."""

from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar

from bot.webhook.types import Filter, FilterFunc

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

class ChainFilters(Generic[T]):
    """Combines multiple filters — all must pass for the event to proceed."""

    def __init__(self, *filters: Filter[T]) -> None:
        self._filters = filters

    def should_process(self, event: T) -> tuple[bool, str]:
        for f in self._filters:
            ok, reason = f.should_process(event)
            if not ok:
                return False, reason
        return True, ""


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class MemoryIdempotencyStore:
    """Simple in-memory idempotency store (single-instance only)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: set[str] = set()

    def has_been_processed(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    def mark_processed(self, key: str) -> None:
        with self._lock:
            self._keys.add(key)


class IdempotencyFilter(Generic[T]):
    """Prevents duplicate processing using an idempotency key."""

    def __init__(
        self,
        key_extractor: Callable[[T], str],
        store: MemoryIdempotencyStore | None = None,
    ) -> None:
        self._key_extractor = key_extractor
        self._store = store or MemoryIdempotencyStore()

    def should_process(self, event: T) -> tuple[bool, str]:
        key = self._key_extractor(event)
        if not key:
            return False, "empty idempotency key"
        if self._store.has_been_processed(key):
            return False, f"already processed (idempotency key: {key})"
        self._store.mark_processed(key)
        return True, ""


# ---------------------------------------------------------------------------
# Mention
# ---------------------------------------------------------------------------

class MentionFilter(Generic[T]):
    """Checks if a specific username is mentioned in the event text."""

    def __init__(self, username: str, text_extractor: Callable[[T], str]) -> None:
        self._username = username
        self._text_extractor = text_extractor

    def should_process(self, event: T) -> tuple[bool, str]:
        text = self._text_extractor(event)
        if not text:
            return False, "empty text"
        if self._username not in text:
            return False, "bot not mentioned"
        return True, ""
