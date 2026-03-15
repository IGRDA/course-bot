"""Slack event type constants for the Events API.

See: https://api.slack.com/events
"""

from bot.webhook.types import EventType

EVENT_APP_MENTION: EventType = "app_mention"
"""Sent when the bot is mentioned in a channel."""

EVENT_MESSAGE: EventType = "message"
"""Sent for all message events, including DMs."""

EVENT_REACTION_ADDED: EventType = "reaction_added"
"""Sent when a reaction is added to a message."""

EVENT_REACTION_REMOVED: EventType = "reaction_removed"
"""Sent when a reaction is removed from a message."""

EVENT_MEMBER_JOINED_CHANNEL: EventType = "member_joined_channel"
"""Sent when a user joins a channel."""

EVENT_URL_VERIFICATION: EventType = "url_verification"
"""Challenge event for Slack URL verification (handled automatically)."""
