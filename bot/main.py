"""Application entry point — lightweight API server.

Receives Slack events, runs filters and acknowledgers, and dispatches
heavy work (app_mention, DM) to Cloud Run Job executions.  Lightweight
events (reactions, member_joined) are handled in-process.

Usage:
    uvicorn bot.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import logging
import sys

from slack_sdk import WebClient

from bot.config import Config
from bot.internal.platform import SlackPlatform
from bot.internal.slack_service import SlackService
from bot.job.dispatcher import JobDispatcher
from bot.server.server import AppServer
from bot.slack.webhook_handler import new_bot_webhook


# ---------------------------------------------------------------------------
# GCP-friendly structured JSON logging
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
        log_entry: dict = {
            "severity": self._LEVEL_MAP.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": self.formatTime(record, self.datefmt),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

config = Config()

_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(_GCPFormatter())
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    handlers=[_handler],
    force=True,
)
logger = logging.getLogger(__name__)

slack_client = WebClient(token=config.slack_bot_token)

if not config.bot_user_id:
    try:
        auth_response = slack_client.auth_test()
        config.bot_user_id = auth_response.get("user_id", "")
        logger.info("Resolved bot user ID: %s", config.bot_user_id)
    except Exception:
        logger.warning("Could not resolve bot user ID via auth.test", exc_info=True)

platform = SlackPlatform(slack_client)

# ---------------------------------------------------------------------------
# Job dispatcher (Cloud Run Jobs)
# ---------------------------------------------------------------------------

job_dispatcher = JobDispatcher(
    project_id=config.gcp_project_id,
    region=config.gcp_region,
    job_name=config.job_name,
    max_concurrent=config.max_concurrent_jobs,
)
logger.info(
    "Job dispatcher ready: project=%s region=%s job=%s max_concurrent=%d",
    config.gcp_project_id,
    config.gcp_region,
    config.job_name,
    config.max_concurrent_jobs,
)

# ---------------------------------------------------------------------------
# Slack service (lightweight — only handles in-process events)
# ---------------------------------------------------------------------------

slack_service = SlackService(
    platform=platform,
    bot_user_id=config.bot_user_id,
    claude_client=None,
)

slack_webhook = new_bot_webhook(
    signing_secret=config.slack_signing_secret,
    provider=slack_service,
    slack_client=slack_client,
    job_dispatcher=job_dispatcher,
)

# HTTP server
server = AppServer(shutdown_timeout=config.shutdown_timeout)
server.register(slack_webhook)

app = server.app

logger.info("Course Bot API server started (bot_user_id=%s)", config.bot_user_id)
