"""Slack webhook handler — receives events, verifies, dispatches.

1. Verifies the Slack signature
2. Parses the event payload
3. Handles URL verification challenge
4. Returns 200 immediately, then processes the event in the background
5. Dispatches heavy events (app_mention, message) to Cloud Run Jobs
6. Handles lightweight events (reactions, member_joined) in-process

IMPORTANT — immediate response: Slack retries events when it does not
receive a 200 within ~3 seconds.  Because job dispatch involves slow GCP
API calls (list_executions, run_job), the handler returns 200 *before*
dispatching and runs the rest in a background asyncio task.  This prevents
retry storms that produce spurious "still processing" messages.

Deduplication: Slack may still re-deliver events (e.g. when the API server
cold-starts and the socket closes before the 200 is sent).  We keep a
bounded in-memory cache of event IDs as a best-effort first layer.
The worker's hourglass-reaction claim provides a second distributed layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Protocol

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from slack_sdk import WebClient

from bot.job.dispatcher import JobDispatcher
from bot.slack import events as ev
from bot.slack.acknowledgers import app_mention_reaction_ack, message_reaction_ack
from bot.slack.filters import (
    direct_message_only,
    joined_by_user,
    skip_bot_mentions,
    skip_bot_messages,
    skip_edited_mentions,
    skip_message_subtypes,
)
from bot.slack.security import SlackSignatureVerifier
from bot.webhook.dispatch import dispatch_event_with_shutdown, wait_for_tasks
from bot.webhook.types import (
    EventHandlerFunc,
    EventHandlerOption,
    EventRegistration,
    new_registration,
)

logger = logging.getLogger(__name__)

_DEDUP_CACHE_MAX_SIZE = 2048
_DEDUP_TTL_SECONDS = 600  # 10 minutes

# Event types that are dispatched to Cloud Run Jobs
_JOB_EVENT_TYPES = frozenset({ev.EVENT_APP_MENTION, ev.EVENT_MESSAGE})


# ---------------------------------------------------------------------------
# HandlerProvider protocol
# ---------------------------------------------------------------------------


class HandlerProvider(Protocol):
    """Provides handler methods and bot identity for wiring."""

    async def handle_app_mention(self, event: dict[str, Any]) -> None: ...
    async def handle_direct_message(self, event: dict[str, Any]) -> None: ...
    async def handle_member_joined(self, event: dict[str, Any]) -> None: ...
    async def handle_reaction_added(self, event: dict[str, Any]) -> None: ...
    async def handle_reaction_removed(self, event: dict[str, Any]) -> None: ...

    @property
    def bot_user_id(self) -> str: ...


# ---------------------------------------------------------------------------
# SlackWebhookHandler
# ---------------------------------------------------------------------------


class SlackWebhookHandler:
    """HTTP handler for Slack Events API.

    Heavy events (app_mention, message) are dispatched to Cloud Run Jobs.
    Lightweight events (reactions, member_joined) run in-process.
    """

    def __init__(
        self,
        signing_secret: str,
        job_dispatcher: JobDispatcher,
        slack_client: WebClient | None = None,
    ) -> None:
        self._verifier = SlackSignatureVerifier(signing_secret)
        self._router = APIRouter()
        self._job_dispatcher = job_dispatcher
        self._slack_client = slack_client

        # Background tasks for lightweight in-process handlers
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._shutdown_event = asyncio.Event()

        # Event registrations by type
        self._app_mention_regs: list[EventRegistration[dict[str, Any]]] = []
        self._message_regs: list[EventRegistration[dict[str, Any]]] = []
        self._reaction_added_regs: list[EventRegistration[dict[str, Any]]] = []
        self._reaction_removed_regs: list[EventRegistration[dict[str, Any]]] = []
        self._member_joined_regs: list[EventRegistration[dict[str, Any]]] = []

        self._seen_events: OrderedDict[str, float] = OrderedDict()

        self._router.add_api_route(
            "/slack/events",
            self._handle_request,
            methods=["POST"],
        )

    @property
    def router(self) -> APIRouter:
        return self._router

    def path(self) -> str:
        return "/slack/events"

    # -- Registration methods -----------------------------------------------

    def on_app_mention(
        self,
        handler: Any,
        *opts: EventHandlerOption,
    ) -> None:
        if callable(handler) and not hasattr(handler, "handle"):
            handler = EventHandlerFunc(handler)
        reg = new_registration(handler, *opts)
        self._app_mention_regs.append(reg)

    def on_message(
        self,
        handler: Any,
        *opts: EventHandlerOption,
    ) -> None:
        if callable(handler) and not hasattr(handler, "handle"):
            handler = EventHandlerFunc(handler)
        reg = new_registration(handler, *opts)
        self._message_regs.append(reg)

    def on_reaction_added(
        self,
        handler: Any,
        *opts: EventHandlerOption,
    ) -> None:
        if callable(handler) and not hasattr(handler, "handle"):
            handler = EventHandlerFunc(handler)
        reg = new_registration(handler, *opts)
        self._reaction_added_regs.append(reg)

    def on_reaction_removed(
        self,
        handler: Any,
        *opts: EventHandlerOption,
    ) -> None:
        if callable(handler) and not hasattr(handler, "handle"):
            handler = EventHandlerFunc(handler)
        reg = new_registration(handler, *opts)
        self._reaction_removed_regs.append(reg)

    def on_member_joined(
        self,
        handler: Any,
        *opts: EventHandlerOption,
    ) -> None:
        if callable(handler) and not hasattr(handler, "handle"):
            handler = EventHandlerFunc(handler)
        reg = new_registration(handler, *opts)
        self._member_joined_regs.append(reg)

    # -- HTTP handling ------------------------------------------------------

    async def _handle_request(self, request: Request) -> JSONResponse:
        body = await self._verifier.verify(request)
        payload = json.loads(body)

        if payload.get("type") == "url_verification":
            return JSONResponse({"challenge": payload["challenge"]})

        event_id = payload.get("event_id", "")
        retry_num = request.headers.get("X-Slack-Retry-Num")

        if event_id and self._is_duplicate(event_id):
            retry_reason = request.headers.get("X-Slack-Retry-Reason", "unknown")
            logger.info(
                "Skipping duplicate event_id=%s (retry #%s, reason=%s)",
                event_id,
                retry_num or "0",
                retry_reason,
            )
            return JSONResponse({"status": "ok"})

        if event_id:
            self._mark_seen(event_id)

        if retry_num is not None:
            logger.info(
                "Accepting Slack retry #%s for event_id=%s (not previously seen)",
                retry_num,
                event_id,
            )

        if payload.get("type") == "event_callback":
            event = payload.get("event", {})
            event_type = event.get("type", "")

            if event_type in _JOB_EVENT_TYPES:
                task = asyncio.create_task(
                    self._safe_dispatch_to_job(event, event_type),
                    name=f"job-dispatch-{event.get('ts', 'unknown')}",
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                self._dispatch_lightweight(event, event_type)

        return JSONResponse({"status": "ok"})

    # -- Job dispatch (heavy events) ----------------------------------------

    async def _safe_dispatch_to_job(
        self,
        event: dict[str, Any],
        event_type: str,
    ) -> None:
        """Background wrapper that ensures dispatch exceptions are logged."""
        try:
            await self._dispatch_to_job(event, event_type)
        except Exception:
            logger.exception(
                "Background job dispatch failed: event_type=%s channel=%s ts=%s",
                event_type,
                event.get("channel", ""),
                event.get("ts", ""),
            )

    async def _dispatch_to_job(
        self,
        event: dict[str, Any],
        event_type: str,
    ) -> None:
        """Run filters + acknowledgers locally, then dispatch to a Cloud Run Job."""
        regs = self._regs_for_event_type(event_type)
        if not regs:
            logger.debug("No registrations for event type: %s", event_type)
            return

        # Apply filters — if all registrations are filtered out, skip dispatch
        passed_regs: list[EventRegistration[dict[str, Any]]] = []
        for reg in regs:
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
            if not skip:
                passed_regs.append(reg)

        if not passed_regs:
            return

        # Run acknowledgers locally (fast emoji reaction)
        for reg in passed_regs:
            if reg.acknowledger is not None:
                try:
                    await reg.acknowledger.acknowledge(event)
                except Exception:
                    logger.warning("Acknowledger failed", exc_info=True)

        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts", "")
        thread_key = f"{channel}:{thread_ts}"

        available, reason = self._job_dispatcher.check_availability(thread_key)
        if not available:
            logger.info(
                "Job dispatch blocked: thread_key=%s reason=%s",
                thread_key,
                reason,
            )
            if self._slack_client and channel:
                try:
                    self._slack_client.chat_postMessage(
                        channel=channel,
                        text=reason,
                        thread_ts=thread_ts or None,
                    )
                except Exception:
                    logger.warning("Failed to send busy message", exc_info=True)
            return

        try:
            self._job_dispatcher.dispatch(event, thread_key)
        except Exception:
            logger.exception(
                "Failed to dispatch job: event_type=%s thread_key=%s",
                event_type,
                thread_key,
            )

    # -- Lightweight dispatch (in-process) ----------------------------------

    def _dispatch_lightweight(
        self,
        event: dict[str, Any],
        event_type: str,
    ) -> None:
        """Dispatch lightweight events in-process (reactions, member_joined)."""
        regs = self._regs_for_event_type(event_type)
        if not regs:
            logger.debug("Unhandled Slack event type: %s", event_type)
            return

        dispatch_event_with_shutdown(
            event_type=event_type,
            event=event,
            registrations=regs,
            background_tasks=self._background_tasks,
            shutdown_event=self._shutdown_event,
        )

    # -- Helpers ------------------------------------------------------------

    def _regs_for_event_type(
        self,
        event_type: str,
    ) -> list[EventRegistration[dict[str, Any]]]:
        if event_type == ev.EVENT_APP_MENTION:
            return self._app_mention_regs
        if event_type == ev.EVENT_MESSAGE:
            return self._message_regs
        if event_type == ev.EVENT_REACTION_ADDED:
            return self._reaction_added_regs
        if event_type == ev.EVENT_REACTION_REMOVED:
            return self._reaction_removed_regs
        if event_type == ev.EVENT_MEMBER_JOINED_CHANNEL:
            return self._member_joined_regs
        return []

    # -- Deduplication ------------------------------------------------------

    def _is_duplicate(self, event_id: str) -> bool:
        return event_id in self._seen_events

    def _mark_seen(self, event_id: str) -> None:
        now = time.monotonic()
        self._seen_events[event_id] = now
        self._seen_events.move_to_end(event_id)

        cutoff = now - _DEDUP_TTL_SECONDS
        while self._seen_events:
            oldest_key, oldest_ts = next(iter(self._seen_events.items()))
            if oldest_ts > cutoff:
                break
            del self._seen_events[oldest_key]

        while len(self._seen_events) > _DEDUP_CACHE_MAX_SIZE:
            self._seen_events.popitem(last=False)

    # -- Graceful shutdown --------------------------------------------------

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Wait for in-flight background tasks (dispatches + lightweight handlers)."""
        self._shutdown_event.set()
        if self._background_tasks:
            logger.info(
                "Waiting for %d background tasks (timeout=%.0fs)...",
                len(self._background_tasks),
                timeout,
            )
            await wait_for_tasks(self._background_tasks, timeout=timeout)


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def new_bot_webhook(
    signing_secret: str,
    provider: HandlerProvider,
    slack_client: Any = None,
    job_dispatcher: JobDispatcher | None = None,
) -> SlackWebhookHandler:
    """Create a SlackWebhookHandler wired to a HandlerProvider.

    Filters and acknowledgers for each event type:
    - app_mention: skip bots, skip edits, reaction ack
    - message: DM only, skip bots, skip subtypes, reaction ack
    - member_joined: only when the bot itself joins
    - reaction_added / reaction_removed: no filters
    """
    if job_dispatcher is None:
        raise ValueError("job_dispatcher is required")

    handler = SlackWebhookHandler(
        signing_secret,
        job_dispatcher=job_dispatcher,
        slack_client=slack_client,
    )

    # -- app_mention --
    mention_opts: list[EventHandlerOption] = [
        skip_bot_mentions(),
        skip_edited_mentions(),
    ]
    if slack_client is not None:
        mention_opts.append(app_mention_reaction_ack(slack_client))
    handler.on_app_mention(provider.handle_app_mention, *mention_opts)

    # -- message (DMs) --
    message_opts: list[EventHandlerOption] = [
        direct_message_only(),
        skip_bot_messages(),
        skip_message_subtypes(),
    ]
    if slack_client is not None:
        message_opts.append(message_reaction_ack(slack_client))
    handler.on_message(provider.handle_direct_message, *message_opts)

    # -- member_joined_channel (bot joins only) --
    handler.on_member_joined(
        provider.handle_member_joined,
        joined_by_user(provider.bot_user_id),
    )

    # -- reactions --
    handler.on_reaction_added(provider.handle_reaction_added)
    handler.on_reaction_removed(provider.handle_reaction_removed)

    return handler
