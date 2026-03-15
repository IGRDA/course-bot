"""Cloud Run Job dispatcher — creates job executions for Slack events.

Dispatches long-running Slack event handling (app_mention, DM) to
Cloud Run Job executions, passing event context via env var overrides.
Also checks running executions for capacity and same-thread gating.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from google.cloud import run_v2

logger = logging.getLogger(__name__)

THREAD_KEY_ENV = "JOB_THREAD_KEY"
EVENT_PAYLOAD_ENV = "JOB_EVENT_PAYLOAD"


class JobDispatcher:
    """Dispatches Slack events to Cloud Run Job executions."""

    def __init__(
        self,
        project_id: str,
        region: str,
        job_name: str,
        max_concurrent: int = 5,
    ) -> None:
        self._job_path = (
            f"projects/{project_id}/locations/{region}/jobs/{job_name}"
        )
        self._max_concurrent = max_concurrent
        self._jobs_client = run_v2.JobsClient()
        self._executions_client = run_v2.ExecutionsClient()

    def check_availability(
        self, thread_key: str,
    ) -> tuple[bool, str]:
        """Check whether a new execution can be dispatched.

        Returns (available, reason).  When *available* is False, *reason*
        contains a user-facing message suitable for posting to Slack.
        """
        running = self._list_running_executions()

        if len(running) >= self._max_concurrent:
            return False, (
                f"I'm currently at maximum capacity "
                f"({self._max_concurrent} concurrent requests). "
                f"Please try again in a few minutes."
            )

        for exe in running:
            if self._extract_thread_key(exe) == thread_key:
                return False, (
                    "I'm still processing a previous request in this thread. "
                    "I'll respond when I'm done — then you can send your follow-up."
                )

        return True, ""

    def dispatch(self, event: dict[str, Any], thread_key: str) -> str:
        """Create a Cloud Run Job execution for the given Slack event.

        Returns the execution name.
        """
        payload_b64 = base64.b64encode(
            json.dumps(event).encode()
        ).decode()

        request = run_v2.RunJobRequest(
            name=self._job_path,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        env=[
                            run_v2.EnvVar(
                                name=EVENT_PAYLOAD_ENV,
                                value=payload_b64,
                            ),
                            run_v2.EnvVar(
                                name=THREAD_KEY_ENV,
                                value=thread_key,
                            ),
                        ],
                    ),
                ],
                task_count=1,
            ),
        )

        operation = self._jobs_client.run_job(request=request)
        execution_name = operation.metadata.name
        logger.info(
            "Dispatched job execution: name=%s thread_key=%s",
            execution_name,
            thread_key,
        )
        return execution_name

    # ------------------------------------------------------------------

    def _list_running_executions(self) -> list[run_v2.Execution]:
        """Return executions that have not yet completed."""
        try:
            executions = self._executions_client.list_executions(
                parent=self._job_path,
            )
            return [
                e for e in executions
                if not e.completion_time
            ]
        except Exception:
            logger.warning(
                "Failed to list executions — allowing dispatch",
                exc_info=True,
            )
            return []

    @staticmethod
    def _extract_thread_key(execution: run_v2.Execution) -> str | None:
        """Best-effort extraction of the thread key from an execution."""
        try:
            for container in execution.template.containers:
                for env in container.env:
                    if env.name == THREAD_KEY_ENV:
                        return env.value
        except Exception:
            pass
        return None
