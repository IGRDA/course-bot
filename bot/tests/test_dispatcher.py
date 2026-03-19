"""Tests for the Cloud Run Job dispatcher."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from bot.job.dispatcher import (
    EVENT_PAYLOAD_ENV,
    THREAD_KEY_ENV,
    JobDispatcher,
)


@pytest.fixture
def dispatcher():
    with patch("bot.job.dispatcher.run_v2"):
        d = JobDispatcher(
            project_id="test-project",
            region="europe-west1",
            job_name="test-job",
            max_concurrent=5,
        )
        d._jobs_client = MagicMock()
        d._executions_client = MagicMock()
        yield d


class TestCheckAvailability:
    def test_allows_when_no_running_executions(self, dispatcher):
        dispatcher._executions_client.list_executions.return_value = []

        available, reason = dispatcher.check_availability("C123:1234.5678")
        assert available is True
        assert reason == ""

    def test_blocks_at_max_capacity(self, dispatcher):
        running = []
        for i in range(5):
            exe = MagicMock()
            exe.completion_time = None
            exe.template.containers = []
            running.append(exe)

        dispatcher._executions_client.list_executions.return_value = running

        available, reason = dispatcher.check_availability("C123:1234.5678")
        assert available is False
        assert "maximum capacity" in reason

    def test_blocks_same_thread_busy(self, dispatcher):
        env_var = MagicMock()
        env_var.name = THREAD_KEY_ENV
        env_var.value = "C123:1234.5678"

        container = MagicMock()
        container.env = [env_var]

        exe = MagicMock()
        exe.completion_time = None
        exe.template.containers = [container]

        dispatcher._executions_client.list_executions.return_value = [exe]

        available, reason = dispatcher.check_availability("C123:1234.5678")
        assert available is False
        assert "still processing" in reason

    def test_allows_different_thread(self, dispatcher):
        env_var = MagicMock()
        env_var.name = THREAD_KEY_ENV
        env_var.value = "C123:other.thread"

        container = MagicMock()
        container.env = [env_var]

        exe = MagicMock()
        exe.completion_time = None
        exe.template.containers = [container]

        dispatcher._executions_client.list_executions.return_value = [exe]

        available, reason = dispatcher.check_availability("C123:1234.5678")
        assert available is True

    def test_ignores_completed_executions(self, dispatcher):
        exe = MagicMock()
        exe.completion_time = MagicMock()  # non-None = completed

        dispatcher._executions_client.list_executions.return_value = [exe]

        available, _ = dispatcher.check_availability("C123:1234.5678")
        assert available is True

    def test_allows_on_api_error(self, dispatcher):
        dispatcher._executions_client.list_executions.side_effect = Exception("API error")

        available, _ = dispatcher.check_availability("C123:1234.5678")
        assert available is True


class TestDispatch:
    def test_creates_execution_with_correct_payload(self):
        """Test dispatch without patching run_v2 — uses real types."""
        from google.cloud import run_v2

        with patch("bot.job.dispatcher.run_v2.JobsClient"), \
             patch("bot.job.dispatcher.run_v2.ExecutionsClient"):
            dispatcher = JobDispatcher(
                project_id="test-project",
                region="europe-west1",
                job_name="test-job",
                max_concurrent=5,
            )
        dispatcher._jobs_client = MagicMock()
        dispatcher._executions_client = MagicMock()

        operation = MagicMock()
        operation.metadata.name = "exec-123"
        dispatcher._jobs_client.run_job.return_value = operation

        event = {"type": "app_mention", "channel": "C123", "ts": "1234.5678"}
        name = dispatcher.dispatch(event, "C123:1234.5678")

        assert name == "exec-123"

        call_args = dispatcher._jobs_client.run_job.call_args
        request = call_args[0][0] if call_args[0] else call_args[1]["request"]

        overrides = request.overrides
        assert overrides.task_count == 1

        env_dict = {
            e.name: e.value
            for e in overrides.container_overrides[0].env
        }
        assert THREAD_KEY_ENV in env_dict
        assert env_dict[THREAD_KEY_ENV] == "C123:1234.5678"

        payload = json.loads(base64.b64decode(env_dict[EVENT_PAYLOAD_ENV]))
        assert payload["type"] == "app_mention"
        assert payload["channel"] == "C123"
