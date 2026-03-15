"""Slack request signature verification."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from slack_sdk.signature import SignatureVerifier

logger = logging.getLogger(__name__)


class SlackSignatureVerifier:
    """Verifies incoming Slack requests using the signing secret."""

    def __init__(self, signing_secret: str) -> None:
        if not signing_secret:
            raise RuntimeError("Slack signing secret must not be empty")
        self._verifier = SignatureVerifier(signing_secret=signing_secret)

    async def verify(self, request: Request) -> bytes:
        """Verify the request was sent by Slack.

        Returns the raw request body on success.
        Raises HTTPException(403) on failure.
        """
        body = await request.body()
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")

        logger.debug(
            "Signature check — timestamp=%s, signature=%s, body_len=%d",
            timestamp,
            signature[:20] + "..." if signature else "(empty)",
            len(body),
        )

        if not self._verifier.is_valid(
            body=body.decode("utf-8"),
            timestamp=timestamp,
            signature=signature,
        ):
            logger.warning(
                "Slack signature verification FAILED — timestamp=%s sig=%s",
                timestamp,
                signature,
            )
            raise HTTPException(status_code=403, detail="Invalid Slack signature")

        return body
