"""Tests for the Cloud Run Job worker."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

from bot.job.worker import _release_claim, _try_claim


class TestTryClaim:
    def test_claims_successfully(self):
        client = MagicMock()
        assert _try_claim(client, "C123", "1234.5678") is True
        client.reactions_add.assert_called_once_with(
            channel="C123",
            timestamp="1234.5678",
            name="hourglass_flowing_sand",
        )

    def test_returns_false_if_already_reacted(self):
        from slack_sdk.errors import SlackApiError

        client = MagicMock()
        err = SlackApiError("", response={"error": "already_reacted"})
        client.reactions_add.side_effect = err

        assert _try_claim(client, "C123", "1234.5678") is False

    def test_returns_true_on_other_errors(self):
        from slack_sdk.errors import SlackApiError

        client = MagicMock()
        err = SlackApiError("", response={"error": "channel_not_found"})
        client.reactions_add.side_effect = err

        assert _try_claim(client, "C123", "1234.5678") is True

    def test_returns_true_on_empty_channel(self):
        client = MagicMock()
        assert _try_claim(client, "", "1234.5678") is True
        client.reactions_add.assert_not_called()


class TestReleaseClaim:
    def test_removes_reaction(self):
        client = MagicMock()
        _release_claim(client, "C123", "1234.5678")
        client.reactions_remove.assert_called_once_with(
            channel="C123",
            timestamp="1234.5678",
            name="hourglass_flowing_sand",
        )

    def test_ignores_errors(self):
        from slack_sdk.errors import SlackApiError

        client = MagicMock()
        err = SlackApiError("", response={"error": "no_reaction"})
        client.reactions_remove.side_effect = err

        _release_claim(client, "C123", "1234.5678")  # should not raise

    def test_noop_on_empty_channel(self):
        client = MagicMock()
        _release_claim(client, "", "1234.5678")
        client.reactions_remove.assert_not_called()


class TestWorkerMain:
    def test_payload_decode(self):
        """Verify the base64+JSON encode/decode roundtrip used by worker."""
        original = {
            "type": "app_mention",
            "channel": "C123",
            "ts": "1234.5678",
            "text": "<@U999> hello",
        }
        payload_b64 = base64.b64encode(json.dumps(original).encode()).decode()
        decoded = json.loads(base64.b64decode(payload_b64))
        assert decoded == original

    @patch.dict(
        "os.environ",
        {
            "JOB_EVENT_PAYLOAD": base64.b64encode(
                json.dumps(
                    {
                        "type": "app_mention",
                        "channel": "C123",
                        "ts": "1234.5678",
                        "text": "<@U999> hello",
                    }
                ).encode()
            ).decode(),
            "SLACK_BOT_TOKEN": "xoxb-test-token",
            "JOB_THREAD_KEY": "C123:1234.5678",
        },
        clear=False,
    )
    @patch("bot.job.worker._process_event")
    def test_main_calls_process_event(self, mock_process):
        """Verify main() decodes event and calls _process_event."""
        from unittest.mock import AsyncMock

        mock_process.side_effect = AsyncMock(return_value=None)

        from bot.job.worker import main

        main()

        mock_process.assert_called_once()
        event_arg = mock_process.call_args[0][0]
        assert event_arg["type"] == "app_mention"
        assert event_arg["channel"] == "C123"
