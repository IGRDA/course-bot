"""Application configuration loaded from environment variables.

API server config: Slack credentials + Cloud Run Job dispatch settings.
Worker-only config (REPO_URL, CLAUDE_RESPONSE_TIMEOUT, etc.) is read
directly from env vars in the worker entry point.
"""

from __future__ import annotations

import os


class Config:
    """API server configuration from environment variables."""

    def __init__(self) -> None:
        self.slack_signing_secret: str = self._require("SLACK_SIGNING_SECRET")
        self.slack_bot_token: str = self._require("SLACK_BOT_TOKEN")
        self.bot_user_id: str = os.environ.get("BOT_USER_ID", "")
        self.port: int = int(os.environ.get("PORT", "8080"))
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO")
        self.shutdown_timeout: float = float(os.environ.get("SHUTDOWN_TIMEOUT", "30"))

        # Cloud Run Job dispatch
        self.gcp_project_id: str = self._require("GCP_PROJECT_ID")
        self.gcp_region: str = os.environ.get("GCP_REGION", "europe-west1")
        self.job_name: str = os.environ.get("JOB_NAME", "course-bot-worker")
        self.max_concurrent_jobs: int = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))

    @staticmethod
    def _require(name: str) -> str:
        value = os.environ.get(name, "")
        if not value:
            raise RuntimeError(f"Required environment variable {name} is not set")
        return value
