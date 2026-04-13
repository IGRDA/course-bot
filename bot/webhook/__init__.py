"""Generic webhook framework — event handler pipeline with filters and acknowledgers."""

from bot.webhook.dispatch import dispatch_event, dispatch_event_with_shutdown
from bot.webhook.errors import (
    InvalidPayloadError,
    InvalidSignatureError,
    MissingSecretError,
    WebhookError,
)
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
    "InvalidPayloadError",
    "InvalidSignatureError",
    "MissingSecretError",
    "WebhookError",
    "dispatch_event",
    "dispatch_event_with_shutdown",
]
