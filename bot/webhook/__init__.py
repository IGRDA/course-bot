"""Generic webhook framework — event handler pipeline with filters and acknowledgers."""

from bot.webhook.types import (
    Acknowledger,
    AcknowledgerFunc,
    EventHandler,
    EventHandlerFunc,
    EventRegistration,
    EventType,
    Filter,
    FilterFunc,
    GracefulHandler,
    HTTPHandler,
)
from bot.webhook.dispatch import dispatch_event, dispatch_event_with_shutdown
from bot.webhook.errors import (
    WebhookError,
    InvalidPayloadError,
    InvalidSignatureError,
    MissingSecretError,
)

__all__ = [
    "Acknowledger",
    "AcknowledgerFunc",
    "EventHandler",
    "EventHandlerFunc",
    "EventRegistration",
    "EventType",
    "Filter",
    "FilterFunc",
    "GracefulHandler",
    "HTTPHandler",
    "dispatch_event",
    "dispatch_event_with_shutdown",
    "WebhookError",
    "InvalidPayloadError",
    "InvalidSignatureError",
    "MissingSecretError",
]
