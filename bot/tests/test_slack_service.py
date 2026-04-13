"""Tests for SlackService subprocess-waiting and output-zip logic."""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.internal.slack_service import SlackService

# ---------------------------------------------------------------------------
# _get_running_pids
# ---------------------------------------------------------------------------


class TestGetRunningPids:
    def test_returns_set_of_ints(self):
        pids = SlackService._get_running_pids()
        assert isinstance(pids, set)
        assert os.getpid() in pids

    @patch("bot.internal.slack_service.subprocess.run")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert SlackService._get_running_pids() == set()

    @patch("bot.internal.slack_service.subprocess.run")
    def test_parses_ps_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="  PID\n    1\n   42\n  100\n",
        )
        assert SlackService._get_running_pids() == {1, 42, 100}


# ---------------------------------------------------------------------------
# _wait_for_subprocesses
# ---------------------------------------------------------------------------


class TestWaitForSubprocesses:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_no_new_pids(self):
        """If no new PIDs appeared, the method should return right away."""
        svc = SlackService(
            platform=MagicMock(),
            bot_user_id="U123",
        )
        baseline = {1, 10, 20}
        with patch.object(
            SlackService,
            "_get_running_pids",
            return_value={1, 10, 20},
        ):
            await svc._wait_for_subprocesses(baseline, timeout=5, poll_interval=0.1)

    @pytest.mark.asyncio
    async def test_waits_until_spawned_processes_exit(self):
        """Should poll until spawned PIDs disappear."""
        svc = SlackService(
            platform=MagicMock(),
            bot_user_id="U123",
        )
        baseline = {1, 10}
        call_count = 0

        def _fake_pids() -> set[int]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {1, 10, 99, 100}
            return {1, 10}

        with patch.object(
            SlackService,
            "_get_running_pids",
            side_effect=_fake_pids,
        ):
            await svc._wait_for_subprocesses(
                baseline,
                timeout=5,
                poll_interval=0.05,
            )
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_respects_timeout(self):
        """Should stop waiting after timeout even if processes remain."""
        svc = SlackService(
            platform=MagicMock(),
            bot_user_id="U123",
        )
        baseline = {1}
        with patch.object(
            SlackService,
            "_get_running_pids",
            return_value={1, 999},
        ):
            await svc._wait_for_subprocesses(
                baseline,
                timeout=0.15,
                poll_interval=0.05,
            )

    @pytest.mark.asyncio
    async def test_handles_cancelled_error_gracefully(self):
        """CancelledError during sleep should not crash the method."""
        svc = SlackService(
            platform=MagicMock(),
            bot_user_id="U123",
        )
        baseline = {1}
        with (
            patch.object(
                SlackService,
                "_get_running_pids",
                return_value={1, 50, 221},
            ),
            patch("bot.internal.slack_service.asyncio.sleep", side_effect=asyncio.CancelledError("cancel scope leak")),
        ):
            await svc._wait_for_subprocesses(
                baseline,
                timeout=5,
                poll_interval=0.1,
            )

    @pytest.mark.asyncio
    async def test_handles_runtime_error_during_sleep(self):
        """RuntimeError during sleep (e.g. closed event loop) should not crash."""
        svc = SlackService(
            platform=MagicMock(),
            bot_user_id="U123",
        )
        baseline = {1}
        with (
            patch.object(
                SlackService,
                "_get_running_pids",
                return_value={1, 50},
            ),
            patch("bot.internal.slack_service.asyncio.sleep", side_effect=RuntimeError("Event loop is closed")),
        ):
            await svc._wait_for_subprocesses(
                baseline,
                timeout=5,
                poll_interval=0.1,
            )


# ---------------------------------------------------------------------------
# _zip_directory
# ---------------------------------------------------------------------------


class TestZipDirectory:
    def test_returns_none_when_dir_missing(self):
        assert SlackService._zip_directory("/nonexistent/path") is None

    def test_returns_none_when_dir_empty(self):
        with tempfile.TemporaryDirectory() as td:
            assert SlackService._zip_directory(td) is None

    def test_zips_files(self):
        with tempfile.TemporaryDirectory() as td:
            (fpath := os.path.join(td, "report.pdf"))
            with open(fpath, "w") as f:
                f.write("pdf content")

            os.makedirs(sub := os.path.join(td, "images"))
            with open(os.path.join(sub, "fig1.png"), "w") as f:
                f.write("png data")

            zip_path = SlackService._zip_directory(td)
            assert zip_path is not None
            assert os.path.isfile(zip_path)
            assert zip_path.endswith(".zip")

            import zipfile

            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                assert "report.pdf" in names
                assert os.path.join("images", "fig1.png") in names

            os.remove(zip_path)


# ---------------------------------------------------------------------------
# _process_and_respond — integration-style with mocks
# ---------------------------------------------------------------------------


class TestProcessAndRespondWaitsForSubprocesses:
    @pytest.mark.asyncio
    async def test_waits_before_sending(self):
        """Verify the ordering: generate → wait → send."""
        platform = MagicMock()
        platform.send_message = AsyncMock()
        platform.get_conversation_replies = AsyncMock(return_value=[])

        claude = MagicMock()
        claude.generate_response = AsyncMock(return_value="Done!")
        claude.cwd = "/tmp/test-cwd"

        svc = SlackService(
            platform=platform,
            bot_user_id="U123",
            claude_client=claude,
        )

        call_order: list[str] = []

        original_generate = claude.generate_response

        async def _tracked_generate(*a, **kw):
            call_order.append("generate")
            return await original_generate(*a, **kw)

        async def _tracked_wait(*a, **kw):
            call_order.append("wait")

        async def _tracked_send(*a, **kw):
            call_order.append("send")

        claude.generate_response = _tracked_generate

        with (
            patch.object(svc, "_wait_for_subprocesses", side_effect=_tracked_wait),
            patch.object(svc, "_send_response", side_effect=_tracked_send),
            patch.object(svc, "_download_event_files", new_callable=AsyncMock, return_value=[]),
        ):
            await svc._process_and_respond(
                event={"type": "app_mention", "text": "hi"},
                channel="C1",
                thread_ts="1.2",
                text="hi",
                workspace_info=None,
            )

        assert call_order == ["generate", "wait", "send"]

    @pytest.mark.asyncio
    async def test_sends_response_even_if_subprocess_wait_raises(self):
        """CancelledError in subprocess wait must not prevent the Slack reply."""
        platform = MagicMock()
        platform.send_message = AsyncMock()
        platform.get_conversation_replies = AsyncMock(return_value=[])

        claude = MagicMock()
        claude.generate_response = AsyncMock(return_value="Course generated!")
        claude.cwd = "/tmp/test-cwd"

        svc = SlackService(
            platform=platform,
            bot_user_id="U123",
            claude_client=claude,
        )

        async def _exploding_wait(*a, **kw):
            raise asyncio.CancelledError("cancel scope leak from SDK")

        send_called = False

        async def _tracked_send(*a, **kw):
            nonlocal send_called
            send_called = True

        with (
            patch.object(svc, "_wait_for_subprocesses", side_effect=_exploding_wait),
            patch.object(svc, "_send_response", side_effect=_tracked_send),
            patch.object(svc, "_download_event_files", new_callable=AsyncMock, return_value=[]),
        ):
            await svc._process_and_respond(
                event={"type": "app_mention", "text": "generate course"},
                channel="C1",
                thread_ts="1.2",
                text="generate course",
                workspace_info=None,
            )

        assert send_called, "_send_response must be called even after CancelledError"
