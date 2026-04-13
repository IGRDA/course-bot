"""Slack-specific acknowledgers — reaction ack, privacy notice, etc."""

from __future__ import annotations

import logging
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from bot.webhook.types import EventHandlerOption, EventRegistration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reaction acknowledger
# ---------------------------------------------------------------------------


class ReactionAcknowledger:
    """Adds an emoji reaction to acknowledge receipt of an event.

    Works with both app_mention and message events (they share
    'channel' and 'ts' fields).
    """

    def __init__(self, client: WebClient, emoji: str = "eyes") -> None:
        self._client = client
        self._emoji = emoji

    async def acknowledge(self, event: dict[str, Any]) -> None:
        channel = event.get("channel", "")
        timestamp = event.get("ts", "")
        if not channel or not timestamp:
            logger.warning("Cannot add reaction — missing channel or ts")
            return
        try:
            self._client.reactions_add(
                channel=channel,
                timestamp=timestamp,
                name=self._emoji,
            )
        except SlackApiError as e:
            # 'already_reacted' is harmless
            if e.response.get("error") != "already_reacted":
                logger.warning("Failed to add reaction: %s", e.response.get("error"))


# ---------------------------------------------------------------------------
# Convenience option builders
# ---------------------------------------------------------------------------


def app_mention_reaction_ack(
    client: WebClient,
    emoji: str = "eyes",
) -> EventHandlerOption:
    """EventHandlerOption that adds a reaction acknowledger for app_mention events."""
    ack = ReactionAcknowledger(client, emoji)

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.acknowledger = ack

    return _apply


def message_reaction_ack(
    client: WebClient,
    emoji: str = "eyes",
) -> EventHandlerOption:
    """EventHandlerOption that adds a reaction acknowledger for message events."""
    ack = ReactionAcknowledger(client, emoji)

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.acknowledger = ack

    return _apply
