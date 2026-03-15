"""Webhook error types."""

from __future__ import annotations


class WebhookError(Exception):
    """Wraps an error with webhook source and event type context."""

    def __init__(self, source: str, event_type: str = "", message: str = "") -> None:
        self.source = source
        self.event_type = event_type
        if event_type:
            super().__init__(f"{source} webhook error ({event_type}): {message}")
        else:
            super().__init__(f"{source} webhook error: {message}")


class InvalidPayloadError(WebhookError):
    """The webhook payload cannot be parsed."""

    def __init__(self, source: str = "", event_type: str = "") -> None:
        super().__init__(source, event_type, "invalid webhook payload")


class InvalidSignatureError(WebhookError):
    """Webhook signature validation failed."""

    def __init__(self, source: str = "", event_type: str = "") -> None:
        super().__init__(source, event_type, "invalid webhook signature")


class MissingSecretError(WebhookError):
    """Webhook secret is not configured."""

    def __init__(self, source: str = "", event_type: str = "") -> None:
        super().__init__(source, event_type, "webhook secret is required")
