"""Cloud Run Job worker — processes a single Slack event.

Entry point for Cloud Run Job executions.  Reads the Slack event from
the ``JOB_EVENT_PAYLOAD`` env var, bootstraps the required services,
and runs the same processing pipeline as the original in-process handler.

Usage (set as the Job CMD):
    python -m bot.job.worker
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from bot.conversation.types import ConversationRequest, WorkspaceInfo
from bot.internal.claude_client import ClaudeClient
from bot.internal.platform import SlackPlatform
from bot.internal.slack_service import SlackService
from bot.workspace.manager import WorkspaceManager

# ---------------------------------------------------------------------------
# GCP-friendly structured JSON logging (same as main.py)
# ---------------------------------------------------------------------------

class _GCPFormatter(logging.Formatter):
    """Emit one JSON object per log line using Cloud Logging severity names."""

    _LEVEL_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        execution_id = os.environ.get("CLOUD_RUN_EXECUTION", "")
        log_entry: dict = {
            "severity": self._LEVEL_MAP.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": self.formatTime(record, self.datefmt),
        }
        if execution_id:
            log_entry["logging.googleapis.com/labels"] = {
                "execution_id": execution_id,
            }
        thread_key = os.environ.get("JOB_THREAD_KEY", "")
        if thread_key:
            log_entry.setdefault(
                "logging.googleapis.com/labels", {},
            )["thread_key"] = thread_key
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def _setup_logging() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_GCPFormatter())
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )
    logging.getLogger("claude_agent_sdk").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

CLAIM_EMOJI = "hourglass_flowing_sand"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

async def _process_event(event: dict) -> None:
    """Run the full Slack event processing pipeline."""
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    slack_client = WebClient(token=slack_token)
    platform = SlackPlatform(slack_client)

    bot_user_id = os.environ.get("BOT_USER_ID", "")
    if not bot_user_id:
        try:
            bot_user_id = slack_client.auth_test().get("user_id", "")
        except Exception:
            logger.warning("Could not resolve bot user ID", exc_info=True)

    repo_url = os.environ.get("REPO_URL", "")
    workspace_manager = WorkspaceManager(
        repo_url=repo_url,
        base_dir=os.environ.get("WORKSPACE_BASE_DIR", "/tmp"),
        default_branch=os.environ.get("DEFAULT_BRANCH", "main"),
        image_source_dir=os.environ.get("IMAGE_SOURCE_DIR", "/app"),
    ) if repo_url else None

    agent_dir = os.environ.get(
        "CLAUDE_AGENT_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "engine"),
    )
    claude_client = ClaudeClient(
        agent_cwd=agent_dir,
        response_timeout=float(os.environ.get("CLAUDE_RESPONSE_TIMEOUT", "5400")),
    )

    slack_service = SlackService(
        platform=platform,
        bot_user_id=bot_user_id,
        claude_client=claude_client,
    )

    channel = event.get("channel", "")
    ts = event.get("ts", "")
    thread_ts = event.get("thread_ts") or ts
    event_type = event.get("type", "")

    # Claim via hourglass reaction
    claimed = _try_claim(slack_client, channel, ts)
    if not claimed:
        logger.info("Event already claimed, exiting: channel=%s ts=%s", channel, ts)
        return

    try:
        workspace_info: WorkspaceInfo | None = None
        if workspace_manager:
            workspace_id = f"{channel}:{thread_ts}"
            workspace_info = await workspace_manager.setup(workspace_id)
            logger.info(
                "Workspace ready: workspace_id=%s branch=%s dir=%s",
                workspace_id,
                workspace_info.branch_name,
                workspace_info.workspace_dir,
            )

        if event_type == "app_mention":
            import re
            pattern = rf"<@{re.escape(bot_user_id)}>\s*"
            text = re.sub(pattern, "", event.get("text", ""), count=1).strip()
        else:
            text = event.get("text", "")

        await slack_service._process_and_respond(
            event=event,
            channel=channel,
            thread_ts=thread_ts,
            text=text,
            workspace_info=workspace_info,
        )

        if workspace_manager and workspace_info:
            try:
                await workspace_manager.finalize(workspace_info)
            except BaseException:
                logger.warning("Workspace finalize failed (non-fatal)", exc_info=True)

    finally:
        _release_claim(slack_client, channel, ts)


def _try_claim(client: WebClient, channel: str, ts: str) -> bool:
    """Add hourglass reaction to claim the event. Returns False if already claimed."""
    if not channel or not ts:
        return True
    try:
        client.reactions_add(channel=channel, timestamp=ts, name=CLAIM_EMOJI)
        return True
    except SlackApiError as e:
        if e.response.get("error") == "already_reacted":
            return False
        logger.warning("Claim reaction failed: %s", e.response.get("error"))
        return True


def _release_claim(client: WebClient, channel: str, ts: str) -> None:
    """Remove the hourglass reaction (best-effort)."""
    if not channel or not ts:
        return
    try:
        client.reactions_remove(channel=channel, timestamp=ts, name=CLAIM_EMOJI)
    except SlackApiError:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    payload_b64 = os.environ.get("JOB_EVENT_PAYLOAD", "")
    if not payload_b64:
        logger.error("JOB_EVENT_PAYLOAD env var is missing")
        sys.exit(1)

    try:
        event = json.loads(base64.b64decode(payload_b64))
    except Exception:
        logger.exception("Failed to decode JOB_EVENT_PAYLOAD")
        sys.exit(1)

    logger.info(
        "Worker starting: event_type=%s channel=%s ts=%s execution=%s",
        event.get("type"),
        event.get("channel"),
        event.get("ts"),
        os.environ.get("CLOUD_RUN_EXECUTION", "unknown"),
    )

    try:
        asyncio.run(_process_event(event))
        logger.info("Worker completed successfully")
    except Exception:
        logger.exception("Worker failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
