"""Slack-specific event filters.

Each filter returns an EventHandlerOption that adds a Filter to the registration.
"""

from __future__ import annotations

from typing import Any

from bot.webhook.types import EventHandlerOption, EventRegistration, FilterFunc

# ---------------------------------------------------------------------------
# AppMention filters
# ---------------------------------------------------------------------------


def skip_bot_mentions() -> EventHandlerOption:
    """Skip app_mention events from bots."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        if event.get("bot_id"):
            return False, "bot mention — skipped"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


def skip_edited_mentions() -> EventHandlerOption:
    """Skip app_mention events that are edits (have 'edited' or 'message' key)."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        if event.get("edited") or event.get("message"):
            return False, "edited mention — skipped"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


def skip_threaded_mentions() -> EventHandlerOption:
    """Skip app_mention events that are in threads (have thread_ts != ts)."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        thread_ts = event.get("thread_ts")
        ts = event.get("ts")
        if thread_ts and thread_ts != ts:
            return False, "threaded mention — skipped"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


def mention_in_channel(*channel_ids: str) -> EventHandlerOption:
    """Only process app_mention events from specific channels."""
    allowed = set(channel_ids)

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        channel = event.get("channel", "")
        if channel not in allowed:
            return False, f"channel {channel} not in allowed list"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


# ---------------------------------------------------------------------------
# Message filters
# ---------------------------------------------------------------------------


def direct_message_only() -> EventHandlerOption:
    """Only process messages with channel_type 'im' (direct messages)."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        if event.get("channel_type") != "im":
            return False, "not a direct message"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


def skip_bot_messages() -> EventHandlerOption:
    """Skip messages from bots."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return False, "bot message — skipped"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


def skip_message_subtypes() -> EventHandlerOption:
    """Skip message events that have a subtype (edits, joins, etc.)."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        if event.get("subtype"):
            return False, f"message subtype '{event['subtype']}' — skipped"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


def message_in_channel(*channel_ids: str) -> EventHandlerOption:
    """Only process message events from specific channels."""
    allowed = set(channel_ids)

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        channel = event.get("channel", "")
        if channel not in allowed:
            return False, f"channel {channel} not in allowed list"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


# ---------------------------------------------------------------------------
# Reaction filters
# ---------------------------------------------------------------------------


def reaction_is(emoji: str) -> EventHandlerOption:
    """Only process reaction events for a specific emoji."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        if event.get("reaction") != emoji:
            return False, f"reaction '{event.get('reaction')}' is not '{emoji}'"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


# ---------------------------------------------------------------------------
# MemberJoinedChannel filters
# ---------------------------------------------------------------------------


def joined_by_user(user_id: str) -> EventHandlerOption:
    """Only process member_joined_channel events for a specific user."""

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        if event.get("user") != user_id:
            return False, f"user {event.get('user')} is not {user_id}"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply


def joined_channel(*channel_ids: str) -> EventHandlerOption:
    """Only process member_joined_channel events for specific channels."""
    allowed = set(channel_ids)

    def _filter(event: dict[str, Any]) -> tuple[bool, str]:
        channel = event.get("channel", "")
        if channel not in allowed:
            return False, f"channel {channel} not in allowed list"
        return True, ""

    def _apply(reg: EventRegistration[dict[str, Any]]) -> None:
        reg.filters.append(FilterFunc(_filter))

    return _apply
