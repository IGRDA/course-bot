"""Generic reusable acknowledgers — Chain, NoOp, Logging, Conditional."""

from __future__ import annotations

import logging
from typing import Callable, Generic, TypeVar

from bot.webhook.types import Acknowledger

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ChainAcknowledgers(Generic[T]):
    """Combines multiple acknowledgers — all are executed in order.

    Errors are logged but do not stop execution.
    """

    def __init__(self, *acknowledgers: Acknowledger[T]) -> None:
        self._acknowledgers = acknowledgers

    async def acknowledge(self, event: T) -> None:
        for ack in self._acknowledgers:
            try:
                await ack.acknowledge(event)
            except Exception:
                logger.warning("Acknowledger in chain failed", exc_info=True)


class NoOpAcknowledger(Generic[T]):
    """An acknowledger that does nothing. Useful as a default or for testing."""

    async def acknowledge(self, event: T) -> None:
        pass


class LoggingAcknowledger(Generic[T]):
    """Logs when an event is acknowledged."""

    def __init__(self, event_name: str, inner: Acknowledger[T] | None = None) -> None:
        self._event_name = event_name
        self._inner = inner

    async def acknowledge(self, event: T) -> None:
        logger.info("Acknowledging event: %s", self._event_name)
        if self._inner is not None:
            await self._inner.acknowledge(event)


class ConditionalAcknowledger(Generic[T]):
    """Only acknowledges if a condition is met."""

    def __init__(self, condition: Callable[[T], bool], inner: Acknowledger[T]) -> None:
        self._condition = condition
        self._inner = inner

    async def acknowledge(self, event: T) -> None:
        if self._condition(event):
            await self._inner.acknowledge(event)
