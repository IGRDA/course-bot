"""Platform abstraction — decouples business logic from Slack API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread message dataclass
# ---------------------------------------------------------------------------


@dataclass
class ThreadMessage:
    """A single message from a Slack thread (platform-agnostic)."""

    id: str
    user_id: str
    user_name: str
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------


class MessagePlatform(Protocol):
    """Platform-agnostic interface for sending messages and managing reactions."""

    async def send_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> str:
        """Send a message. Returns the message timestamp (ID)."""
        ...

    async def get_conversation_replies(
        self,
        channel: str,
        thread_ts: str,
    ) -> list[ThreadMessage]:
        """Fetch all messages in a thread. Returns list of ThreadMessage."""
        ...

    async def add_reaction(
        self,
        channel: str,
        timestamp: str,
        emoji: str,
    ) -> None:
        """Add an emoji reaction to a message."""
        ...

    async def remove_reaction(
        self,
        channel: str,
        timestamp: str,
        emoji: str,
    ) -> None:
        """Remove an emoji reaction from a message."""
        ...

    async def download_file(
        self,
        url: str,
        dest_path: str,
    ) -> str:
        """Download a file from a private URL. Returns the destination path."""
        ...

    async def upload_file(
        self,
        channel: str,
        file_path: str,
        title: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> None:
        """Upload a file to a channel or thread."""
        ...


# ---------------------------------------------------------------------------
# Slack implementation
# ---------------------------------------------------------------------------


class SlackPlatform:
    """Concrete MessagePlatform backed by the Slack Web API.

    Mirrors agents internal/slack.SlackPlatform.
    """

    def __init__(self, client: WebClient) -> None:
        self._client = client
        self._user_cache: dict[str, str] = {}  # user_id -> display name

    async def send_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> str:
        """Send a message to a Slack channel or thread.

        Returns the message timestamp (Slack's unique message ID).
        """
        kwargs: dict = {
            "channel": channel,
            "text": text,
        }
        if thread_ts:
            kwargs["thread_ts"] = thread_ts

        try:
            response = self._client.chat_postMessage(**kwargs)
            ts: str = response.get("ts", "")
            logger.info("Sent message to channel=%s ts=%s", channel, ts)
            return ts
        except SlackApiError as e:
            logger.error("Failed to send message to Slack: %s", e.response.get("error"))
            raise

    async def get_conversation_replies(
        self,
        channel: str,
        thread_ts: str,
    ) -> list[ThreadMessage]:
        """Fetch all messages in a Slack thread.

        Uses conversations.replies to get the parent message and all replies.
        """
        try:
            response = self._client.conversations_replies(
                channel=channel,
                ts=thread_ts,
            )
            raw_messages = response.get("messages", [])
        except SlackApiError as e:
            error_code = e.response.get("error", "")
            if error_code == "missing_scope":
                logger.error(
                    "Missing OAuth scope for conversations.replies. "
                    "Add 'channels:history', 'groups:history', 'im:history', "
                    "and 'mpim:history' to your Slack bot token scopes at "
                    "https://api.slack.com/apps → OAuth & Permissions."
                )
            else:
                logger.error(
                    "Failed to fetch conversation replies: %s",
                    error_code,
                )
            return []

        messages: list[ThreadMessage] = []
        for msg in raw_messages:
            ts_str = msg.get("ts", "")
            try:
                ts_float = float(ts_str)
                timestamp = datetime.fromtimestamp(ts_float, tz=UTC)
            except (ValueError, TypeError):
                timestamp = datetime.now(tz=UTC)

            user_id = msg.get("user", "")
            user_name = self._resolve_user(user_id)

            messages.append(
                ThreadMessage(
                    id=ts_str,
                    user_id=user_id,
                    user_name=user_name,
                    text=msg.get("text", ""),
                    timestamp=timestamp,
                )
            )

        logger.info(
            "Fetched %d thread replies for channel=%s thread_ts=%s",
            len(messages),
            channel,
            thread_ts,
        )
        return messages

    def _resolve_user(self, user_id: str) -> str:
        """Resolve a Slack user ID to a display name (cached)."""
        if not user_id:
            return "unknown"

        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            response = self._client.users_info(user=user_id)
            user = response.get("user", {})
            # Prefer real_name, then display_name from profile, then fallback
            name = user.get("real_name") or user.get("profile", {}).get("display_name") or user.get("name") or user_id
            self._user_cache[user_id] = name
            return name
        except SlackApiError:
            logger.warning("Failed to resolve user %s", user_id)
            self._user_cache[user_id] = user_id
            return user_id

    async def download_file(
        self,
        url: str,
        dest_path: str,
    ) -> str:
        """Download a Slack-hosted file using the bot token for auth.

        Slack's url_private_download requires a Bearer token header.
        Returns the destination path on success.
        """
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        headers = {"Authorization": f"Bearer {self._client.token}"}
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as http:
                resp = await http.get(url, headers=headers)
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
            logger.info("Downloaded file: %s -> %s (%d bytes)", url[:80], dest_path, len(resp.content))
            return dest_path
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error downloading file: %s — %s", e.response.status_code, url[:80])
            raise
        except Exception:
            logger.exception("Failed to download file from %s", url[:80])
            raise

    async def download_event_files(
        self,
        event: dict[str, Any],
        upload_dir: str,
    ) -> list[str]:
        """Extract and download all files attached to a Slack event.

        Creates the upload_dir if needed. Returns list of local file paths
        for successfully downloaded files.
        """
        files = event.get("files", [])
        if not files:
            return []

        os.makedirs(upload_dir, exist_ok=True)
        downloaded: list[str] = []

        for file_info in files:
            url = file_info.get("url_private_download") or file_info.get("url_private")
            name = file_info.get("name", f"file_{file_info.get('id', 'unknown')}")
            if not url:
                logger.warning("Slack file %s has no download URL — skipping", name)
                continue

            dest = os.path.join(upload_dir, name)
            try:
                await self.download_file(url, dest)
                downloaded.append(dest)
            except Exception:
                logger.warning("Skipping file %s — download failed", name)

        return downloaded

    async def upload_file(
        self,
        channel: str,
        file_path: str,
        title: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> None:
        """Upload a file to a Slack channel or thread.

        Uses files_upload_v2 which handles large files and posts to the
        channel in a single API call.
        """
        filename = os.path.basename(file_path)
        kwargs: dict[str, Any] = {
            "channel": channel,
            "file": file_path,
            "filename": filename,
            "title": title or filename,
        }
        if initial_comment:
            kwargs["initial_comment"] = initial_comment
        if thread_ts:
            kwargs["thread_ts"] = thread_ts

        try:
            self._client.files_upload_v2(**kwargs)
            logger.info(
                "Uploaded file %s to channel=%s thread_ts=%s",
                filename,
                channel,
                thread_ts,
            )
        except SlackApiError as e:
            logger.error(
                "Failed to upload file %s to Slack: %s",
                filename,
                e.response.get("error"),
            )
            raise

    async def add_reaction(
        self,
        channel: str,
        timestamp: str,
        emoji: str,
    ) -> None:
        """Add an emoji reaction to a message."""
        try:
            self._client.reactions_add(
                channel=channel,
                timestamp=timestamp,
                name=emoji,
            )
        except SlackApiError as e:
            if e.response.get("error") != "already_reacted":
                logger.warning("Failed to add reaction: %s", e.response.get("error"))

    async def remove_reaction(
        self,
        channel: str,
        timestamp: str,
        emoji: str,
    ) -> None:
        """Remove an emoji reaction from a message."""
        try:
            self._client.reactions_remove(
                channel=channel,
                timestamp=timestamp,
                name=emoji,
            )
        except SlackApiError as e:
            if e.response.get("error") != "no_reaction":
                logger.warning("Failed to remove reaction: %s", e.response.get("error"))
