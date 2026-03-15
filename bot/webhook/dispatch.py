"""Event dispatch — runs filters, acknowledgers, and handlers.

- Filters run synchronously (they are not async)
- Acknowledger runs before the handler (awaited inline)
- Handler runs as a background asyncio.Task, tracked for graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

from bot.webhook.types import EventRegistration, EventType

logger = logging.getLogger(__name__)

T = TypeVar("T")


def dispatch_event(
    event_type: EventType,
    event: T,
    registrations: list[EventRegistration[T]],
    background_tasks: set[asyncio.Task[None]],
) -> bool:
    """Dispatch an event through registered handlers (no shutdown context).

    Convenience wrapper around dispatch_event_with_shutdown.
    """
    return dispatch_event_with_shutdown(
        event_type=event_type,
        event=event,
        registrations=registrations,
        background_tasks=background_tasks,
        shutdown_event=None,
    )


def dispatch_event_with_shutdown(
    event_type: EventType,
    event: T,
    registrations: list[EventRegistration[T]],
    background_tasks: set[asyncio.Task[None]],
    shutdown_event: asyncio.Event | None = None,
) -> bool:
    """Dispatch an event through registered handlers.

    For each registration:
      1. Run filters (sync) — skip if any filter rejects
      2. Run acknowledger (sync-ish, awaited inline via create_task)
      3. Run handler in a background asyncio.Task

    Returns True if at least one registration existed.
    """
    if not registrations:
        return False

    for reg in registrations:
        # 1. Run filters (synchronous)
        skip = False
        for f in reg.filters:
            ok, reason = f.should_process(event)
            if not ok:
                logger.debug(
                    "Event filtered: event_type=%s reason=%s",
                    event_type,
                    reason,
                )
                skip = True
                break
        if skip:
            continue

        # 2. Run acknowledger (sync-ish — schedule and let it run)
        if reg.acknowledger is not None:
            task = asyncio.create_task(_run_acknowledger(event_type, reg, event))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        # 3. Run handler in background task
        task = asyncio.create_task(_run_handler(event_type, reg, event, shutdown_event))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    return True


async def _run_acknowledger(
    event_type: EventType,
    reg: EventRegistration[T],
    event: T,
) -> None:
    """Run an acknowledger, logging errors without stopping processing."""
    try:
        if reg.acknowledger is not None:
            await reg.acknowledger.acknowledge(event)
    except Exception:
        logger.warning("Acknowledger failed for event_type=%s", event_type, exc_info=True)


async def _run_handler(
    event_type: EventType,
    reg: EventRegistration[T],
    event: T,
    shutdown_event: asyncio.Event | None,
) -> None:
    """Run a handler, logging errors."""
    try:
        await reg.handler.handle(event)
    except Exception:
        logger.error("Handler failed for event_type=%s", event_type, exc_info=True)


async def wait_for_tasks(
    tasks: set[asyncio.Task[None]],
    timeout: float | None = None,
) -> None:
    """Wait for background tasks, cancelling any that exceed the timeout.

    Args:
        tasks: The set of tracked background tasks.
        timeout: Maximum seconds to wait. None means wait forever.
    """
    if not tasks:
        return

    pending = list(tasks)
    try:
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        if still_pending:
            logger.warning(
                "Cancelling %d task(s) that exceeded %.0fs timeout",
                len(still_pending),
                timeout or 0,
            )
            for task in still_pending:
                task.cancel()
            await asyncio.wait(still_pending, timeout=10)
    except Exception:
        logger.warning("Error waiting for background tasks", exc_info=True)
