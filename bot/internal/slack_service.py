"""Slack service — implements HandlerProvider with business logic.

In the API server, only lightweight handlers (member_joined, reactions)
are invoked in-process.  The heavy handlers (app_mention, direct_message)
are registered for the filter/acknowledger pipeline but the actual work
is dispatched to Cloud Run Job executions.  The worker entry point calls
``_process_and_respond()`` directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from typing import Any

from bot.conversation.types import WorkspaceInfo
from bot.internal.claude_client import ClaudeClient
from bot.internal.platform import MessagePlatform, SlackPlatform, ThreadMessage

logger = logging.getLogger(__name__)

_UPLOADS_BASE_DIR = os.environ.get("UPLOADS_BASE_DIR", "/tmp/slack-uploads")


class SlackService:
    """Handles Slack events using the platform abstraction.

    Implements the HandlerProvider protocol expected by
    bot.slack.webhook_handler.new_bot_webhook.
    """

    def __init__(
        self,
        platform: MessagePlatform,
        bot_user_id: str,
        claude_client: ClaudeClient | None = None,
    ) -> None:
        self._platform = platform
        self._bot_user_id = bot_user_id
        self._claude = claude_client

    @property
    def bot_user_id(self) -> str:
        return self._bot_user_id

    # -- Event handlers (registered for filters/acknowledgers) ---------------
    #
    # handle_app_mention and handle_direct_message are registered so the
    # webhook handler can apply filters and acknowledgers, but the actual
    # processing is done by the Cloud Run Job worker which calls
    # _process_and_respond() directly.

    async def handle_app_mention(self, event: dict[str, Any]) -> None:
        logger.debug("handle_app_mention called — should be handled by worker job")

    async def handle_direct_message(self, event: dict[str, Any]) -> None:
        logger.debug("handle_direct_message called — should be handled by worker job")

    async def handle_member_joined(self, event: dict[str, Any]) -> None:
        channel = event.get("channel", "")
        logger.info("Bot joined channel=%s", channel)
        await self._platform.send_message(
            channel=channel,
            text="Hello! I'm here to help. Mention me with `@course-bot` to get started!",
        )

    async def handle_reaction_added(self, event: dict[str, Any]) -> None:
        logger.debug(
            "Reaction added: %s by user=%s",
            event.get("reaction"),
            event.get("user"),
        )

    async def handle_reaction_removed(self, event: dict[str, Any]) -> None:
        logger.debug(
            "Reaction removed: %s by user=%s",
            event.get("reaction"),
            event.get("user"),
        )

    # -- Core processing (called by the worker) ------------------------------

    async def _process_and_respond(
        self,
        event: dict[str, Any],
        channel: str,
        thread_ts: str | None,
        text: str,
        workspace_info: WorkspaceInfo | None,
    ) -> None:
        """Download files, build prompt, invoke Claude, and send response.

        The Slack reply is held until Claude finishes AND every subprocess it
        spawned has exited, so the output directory is guaranteed to be
        complete before we zip and upload it.
        """
        downloaded_files = await self._download_event_files(
            event=event,
            channel=channel,
            thread_ts=thread_ts,
            workspace_info=workspace_info,
        )

        prompt = await self._build_thread_prompt(
            channel=channel,
            thread_ts=thread_ts,
            fallback_text=text,
            downloaded_files=downloaded_files,
        )

        agent_cwd = os.path.join(workspace_info.workspace_dir, "engine") if workspace_info is not None else None

        if self._claude is None:
            raise RuntimeError("ClaudeClient is required for _process_and_respond")

        baseline_pids = self._get_running_pids()

        response = await self._claude.generate_response(prompt, cwd=agent_cwd)

        try:
            await self._wait_for_subprocesses(baseline_pids)
        except BaseException:
            logger.warning(
                "Subprocess wait raised unexpectedly, proceeding to send response",
                exc_info=True,
            )

        effective_cwd = agent_cwd or self._claude.cwd
        output_dir = os.path.join(effective_cwd, "output")

        await self._send_response(
            channel=channel,
            text=response,
            thread_ts=thread_ts,
            output_dir=output_dir,
        )

    # -- Subprocess lifecycle ------------------------------------------------

    @staticmethod
    def _get_running_pids() -> set[int]:
        """Return the set of PIDs currently running in the container."""
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return set()
        pids: set[int] = set()
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            with contextlib.suppress(ValueError):
                pids.add(int(line))  # skip non-integer lines like "PID" header
        return pids

    async def _wait_for_subprocesses(
        self,
        baseline_pids: set[int],
        timeout: float = 1800.0,
        poll_interval: float = 10.0,
    ) -> None:
        """Block until every process spawned after *baseline_pids* has exited.

        This prevents sending the Slack reply (and zipping the output
        directory) while background work started by Claude is still running.

        Catches ``CancelledError`` on the poll sleep so that stale cancel
        scopes leaked by the claude-agent-sdk async generator cleanup
        cannot crash the worker and prevent the Slack reply from being sent.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            current = self._get_running_pids()
            spawned = current - baseline_pids
            if not spawned:
                logger.info("All spawned subprocesses have exited")
                return
            logger.info(
                "Waiting for %d spawned subprocess(es) to finish: %s",
                len(spawned),
                spawned,
            )
            try:
                await asyncio.sleep(poll_interval)
            except (asyncio.CancelledError, Exception) as exc:
                logger.warning(
                    "Subprocess wait interrupted (%s: %s), proceeding with %d subprocess(es) still running",
                    type(exc).__name__,
                    exc,
                    len(spawned),
                )
                return

        remaining = self._get_running_pids() - baseline_pids
        if remaining:
            logger.warning(
                "Timed out after %.0fs waiting for %d subprocess(es): %s",
                timeout,
                len(remaining),
                remaining,
            )

    # -- Helpers ------------------------------------------------------------

    async def _send_response(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Send a text response and attach the output directory as a zip."""
        await self._platform.send_message(
            channel=channel,
            text=text,
            thread_ts=thread_ts,
        )

        zip_path = self._zip_directory(output_dir) if output_dir else None
        if zip_path is None:
            return

        try:
            if isinstance(self._platform, SlackPlatform):
                await self._platform.upload_file(
                    channel=channel,
                    file_path=zip_path,
                    title="output.zip",
                    thread_ts=thread_ts,
                )
                logger.info("Uploaded output.zip to channel=%s", channel)
            else:
                logger.warning(
                    "File upload not supported on this platform — skipping output.zip",
                )
        finally:
            try:
                os.remove(zip_path)
            except OSError:
                logger.warning("Failed to remove temp zip: %s", zip_path)
            if output_dir:
                try:
                    shutil.rmtree(output_dir)
                    logger.info("Cleaned up output directory: %s", output_dir)
                except OSError:
                    pass

    @staticmethod
    def _zip_directory(output_dir: str) -> str | None:
        """Zip the contents of *output_dir* if it exists and has files."""
        if not os.path.isdir(output_dir):
            return None

        all_files: list[str] = []
        for root, _dirs, files in os.walk(output_dir):
            for fname in files:
                all_files.append(os.path.join(root, fname))

        if not all_files:
            logger.debug("Output directory exists but is empty — skipping zip")
            return None

        tmp_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="output_")
        os.close(tmp_fd)

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fpath in all_files:
                    arcname = os.path.relpath(fpath, output_dir)
                    zf.write(fpath, arcname)
            logger.info(
                "Zipped %d file(s) from %s -> %s",
                len(all_files),
                output_dir,
                zip_path,
            )
            return zip_path
        except Exception:
            logger.exception("Failed to create zip from output directory")
            with contextlib.suppress(OSError):
                os.remove(zip_path)
        return None

    async def _download_event_files(
        self,
        event: dict[str, Any],
        channel: str,
        thread_ts: str | None,
        workspace_info: WorkspaceInfo | None = None,
    ) -> list[str]:
        files = event.get("files", [])
        if not files:
            return []

        if not isinstance(self._platform, SlackPlatform):
            logger.warning("File download not supported on this platform")
            return []

        if workspace_info is not None:
            upload_dir = os.path.join(workspace_info.workspace_dir, "engine", "uploads")
        else:
            conv_key = f"{channel}:{thread_ts or 'root'}"
            dir_hash = hashlib.sha1(conv_key.encode()).hexdigest()[:12]
            upload_dir = os.path.join(_UPLOADS_BASE_DIR, dir_hash)

        logger.info("Downloading %d file(s) to %s", len(files), upload_dir)
        downloaded = await self._platform.download_event_files(event, upload_dir)

        if downloaded:
            logger.info(
                "Downloaded %d/%d files: %s",
                len(downloaded),
                len(files),
                [os.path.basename(p) for p in downloaded],
            )
        return downloaded

    async def _build_thread_prompt(
        self,
        channel: str,
        thread_ts: str | None,
        fallback_text: str,
        downloaded_files: list[str] | None = None,
    ) -> str:
        if not thread_ts:
            base = fallback_text or "hello"
            if downloaded_files:
                base += self._format_file_attachment_prompt(downloaded_files)
            return base

        messages = await self._platform.get_conversation_replies(
            channel=channel,
            thread_ts=thread_ts,
        )

        if not messages or len(messages) <= 1:
            base = fallback_text or "hello"
            if downloaded_files:
                base += self._format_file_attachment_prompt(downloaded_files)
            return base

        prompt = self._build_prompt_from_messages(messages)

        now = datetime.now(tz=UTC)
        context_info = (
            f"Context: You were mentioned in a Slack thread.\n"
            f"Current datetime: {now.strftime('%A, %B %d, %Y at %I:%M %p')} UTC\n\n"
        )

        result = context_info + prompt

        if downloaded_files:
            result += self._format_file_attachment_prompt(downloaded_files)

        return result

    def _build_prompt_from_messages(self, messages: list[ThreadMessage]) -> str:
        lines = ["Thread messages:"]
        for msg in messages:
            cleaned = self._strip_bot_mention(msg.text)
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"({timestamp_str}) [{msg.user_name}] {cleaned}")
        lines.append("")
        return "\n".join(lines)

    def _strip_bot_mention(self, text: str) -> str:
        pattern = rf"<@{re.escape(self._bot_user_id)}>\s*"
        return re.sub(pattern, "", text, count=1).strip()

    @staticmethod
    def _format_file_attachment_prompt(file_paths: list[str]) -> str:
        if not file_paths:
            return ""
        lines = [
            "\n\n--- Attached Files ---",
            "The user shared the following file(s) in Slack.",
            "They have been downloaded and saved locally.",
            "Use the Read tool to examine their contents.\n",
        ]
        for path in file_paths:
            lines.append(f"  - {path}")
        lines.append("--- End Attached Files ---")
        return "\n".join(lines)
