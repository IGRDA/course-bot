"""Core webhook types — EventHandler, Filter, Acknowledger, EventRegistration.

These are generic, source-agnostic interfaces. Slack-specific implementations
live in app.slack.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")

# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------

EventType = str
"""Webhook event type identifier (e.g. 'app_mention', 'message')."""


# ---------------------------------------------------------------------------
# Core protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EventHandler(Protocol[T]):
    """Processes a webhook event of type T.

    Implementations might enqueue to a task queue, process in-place,
    or forward the event to another service.
    """

    async def handle(self, event: T) -> None: ...


@runtime_checkable
class Filter(Protocol[T]):
    """Decides whether an event should be processed.

    Return (True, '') to process, or (False, reason) to skip.
    """

    def should_process(self, event: T) -> tuple[bool, str]: ...


@runtime_checkable
class Acknowledger(Protocol[T]):
    """Sends an acknowledgement when an event is received.

    Runs *before* the main handler processes the event.
    Common uses: add emoji reaction, post a 'processing' message.
    """

    async def acknowledge(self, event: T) -> None: ...


# ---------------------------------------------------------------------------
# Function adapters
# ---------------------------------------------------------------------------


class EventHandlerFunc(Generic[T]):
    """Wraps a plain async function as an EventHandler."""

    def __init__(self, fn: Callable[[T], Awaitable[None]]) -> None:
        self._fn = fn

    async def handle(self, event: T) -> None:
        await self._fn(event)


class FilterFunc(Generic[T]):
    """Wraps a plain function as a Filter."""

    def __init__(self, fn: Callable[[T], tuple[bool, str]]) -> None:
        self._fn = fn

    def should_process(self, event: T) -> tuple[bool, str]:
        return self._fn(event)


class AcknowledgerFunc(Generic[T]):
    """Wraps a plain async function as an Acknowledger."""

    def __init__(self, fn: Callable[[T], Awaitable[None]]) -> None:
        self._fn = fn

    async def acknowledge(self, event: T) -> None:
        await self._fn(event)


# ---------------------------------------------------------------------------
# EventRegistration
# ---------------------------------------------------------------------------


@dataclass
class EventRegistration(Generic[T]):
    """Wraps a handler with its filters and optional acknowledger."""

    handler: EventHandler[T]
    filters: list[Filter[T]] = field(default_factory=list)
    acknowledger: Acknowledger[T] | None = None


# ---------------------------------------------------------------------------
# EventHandlerOption — functional option for building registrations
# ---------------------------------------------------------------------------

EventHandlerOption = Callable[["EventRegistration[Any]"], None]
"""Configures an EventRegistration (adds filters, sets acknowledger, etc.)."""


def with_filter(f: Filter[T]) -> EventHandlerOption:
    """Add a filter to an event handler registration."""

    def _apply(reg: EventRegistration[T]) -> None:
        reg.filters.append(f)

    return _apply


def with_filter_func(fn: Callable[[T], tuple[bool, str]]) -> EventHandlerOption:
    """Add a filter function to an event handler registration."""
    return with_filter(FilterFunc(fn))


def with_acknowledger(ack: Acknowledger[T]) -> EventHandlerOption:
    """Set the acknowledger for an event handler registration."""

    def _apply(reg: EventRegistration[T]) -> None:
        reg.acknowledger = ack

    return _apply


def with_acknowledger_func(fn: Callable[[T], Awaitable[None]]) -> EventHandlerOption:
    """Set an acknowledger function for an event handler registration."""
    return with_acknowledger(AcknowledgerFunc(fn))


def new_registration(
    handler: EventHandler[T],
    *opts: EventHandlerOption,
) -> EventRegistration[T]:
    """Create a new EventRegistration with the given handler and options."""
    reg = EventRegistration(handler=handler)
    for opt in opts:
        opt(reg)
    return reg


def new_registration_func(
    fn: Callable[[T], Awaitable[None]],
    *opts: EventHandlerOption,
) -> EventRegistration[T]:
    """Create a new EventRegistration from a handler function."""
    return new_registration(EventHandlerFunc(fn), *opts)


# ---------------------------------------------------------------------------
# Handler interfaces (for HTTP server registration)
# ---------------------------------------------------------------------------


@runtime_checkable
class HTTPHandler(Protocol):
    """Webhook handler that can be registered with the HTTP server."""

    def path(self) -> str:
        """Return the default registration path (e.g. '/slack/events')."""
        ...

    async def handle_request(self, request: Any) -> Any:
        """Handle an incoming HTTP request."""
        ...


@runtime_checkable
class GracefulHandler(Protocol):
    """Implemented by handlers that need cleanup on shutdown."""

    async def shutdown(self) -> None: ...
