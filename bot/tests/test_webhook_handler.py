"""Tests for the refactored webhook handler with job dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bot.job.dispatcher import JobDispatcher
from bot.slack.webhook_handler import SlackWebhookHandler, new_bot_webhook


def _make_signing_secret():
    return "test_signing_secret_12345"


def _sign_request(body: bytes, secret: str, timestamp: str | None = None) -> dict:
    ts = timestamp or str(int(time.time()))
    sig_basestring = f"v0:{ts}:{body.decode()}"
    signature = "v0=" + hmac.new(
        secret.encode(), sig_basestring.encode(), hashlib.sha256,
    ).hexdigest()
    return {
        "X-Slack-Signature": signature,
        "X-Slack-Request-Timestamp": ts,
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.bot_user_id = "U_BOT"
    provider.handle_app_mention = AsyncMock()
    provider.handle_direct_message = AsyncMock()
    provider.handle_member_joined = AsyncMock()
    provider.handle_reaction_added = AsyncMock()
    provider.handle_reaction_removed = AsyncMock()
    return provider


@pytest.fixture
def mock_dispatcher():
    d = MagicMock(spec=JobDispatcher)
    d.check_availability.return_value = (True, "")
    d.dispatch.return_value = "exec-123"
    return d


@pytest.fixture
def mock_slack_client():
    client = MagicMock()
    return client


@pytest.fixture
def app(mock_provider, mock_dispatcher, mock_slack_client):
    from fastapi import FastAPI

    handler = new_bot_webhook(
        signing_secret=_make_signing_secret(),
        provider=mock_provider,
        slack_client=mock_slack_client,
        job_dispatcher=mock_dispatcher,
    )
    app = FastAPI()
    app.include_router(handler.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestURLVerification:
    def test_url_verification(self, client):
        body = json.dumps({
            "type": "url_verification",
            "challenge": "test_challenge_abc",
        }).encode()
        headers = _sign_request(body, _make_signing_secret())

        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["challenge"] == "test_challenge_abc"


class TestEventDedup:
    def test_duplicate_event_is_skipped(self, client):
        body = json.dumps({
            "type": "event_callback",
            "event_id": "Ev_dedup_test",
            "event": {"type": "reaction_added", "reaction": "thumbsup", "user": "U123"},
        }).encode()
        headers = _sign_request(body, _make_signing_secret())

        resp1 = client.post("/slack/events", content=body, headers=headers)
        assert resp1.status_code == 200

        headers2 = _sign_request(body, _make_signing_secret())
        headers2["X-Slack-Retry-Num"] = "1"
        resp2 = client.post("/slack/events", content=body, headers=headers2)
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "ok"


class TestJobDispatch:
    def test_app_mention_dispatches_job(self, client, mock_dispatcher):
        body = json.dumps({
            "type": "event_callback",
            "event_id": "Ev_mention_1",
            "event": {
                "type": "app_mention",
                "channel": "C123",
                "ts": "1234.5678",
                "text": "<@U_BOT> hello",
                "user": "U_USER",
            },
        }).encode()
        headers = _sign_request(body, _make_signing_secret())

        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 200

        mock_dispatcher.check_availability.assert_called_once_with("C123:1234.5678")
        mock_dispatcher.dispatch.assert_called_once()

    def test_dm_dispatches_job(self, client, mock_dispatcher):
        body = json.dumps({
            "type": "event_callback",
            "event_id": "Ev_dm_1",
            "event": {
                "type": "message",
                "channel": "D999",
                "channel_type": "im",
                "ts": "5555.0000",
                "text": "hi there",
                "user": "U_USER",
            },
        }).encode()
        headers = _sign_request(body, _make_signing_secret())

        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 200

        mock_dispatcher.dispatch.assert_called_once()

    def test_busy_thread_sends_message(self, client, mock_dispatcher, mock_slack_client):
        mock_dispatcher.check_availability.return_value = (
            False,
            "I'm still processing a previous request in this thread.",
        )

        body = json.dumps({
            "type": "event_callback",
            "event_id": "Ev_busy_1",
            "event": {
                "type": "app_mention",
                "channel": "C123",
                "ts": "9999.0000",
                "text": "<@U_BOT> hello again",
                "user": "U_USER",
            },
        }).encode()
        headers = _sign_request(body, _make_signing_secret())

        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 200

        mock_dispatcher.dispatch.assert_not_called()
        mock_slack_client.chat_postMessage.assert_called_once()
        call_kwargs = mock_slack_client.chat_postMessage.call_args[1]
        assert "still processing" in call_kwargs["text"]


class TestLightweightEvents:
    def test_reaction_handled_in_process(self, client, mock_dispatcher, mock_provider):
        body = json.dumps({
            "type": "event_callback",
            "event_id": "Ev_reaction_1",
            "event": {
                "type": "reaction_added",
                "reaction": "thumbsup",
                "user": "U_USER",
            },
        }).encode()
        headers = _sign_request(body, _make_signing_secret())

        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 200

        mock_dispatcher.dispatch.assert_not_called()

    def test_member_joined_handled_in_process(self, client, mock_dispatcher, mock_provider):
        body = json.dumps({
            "type": "event_callback",
            "event_id": "Ev_joined_1",
            "event": {
                "type": "member_joined_channel",
                "channel": "C123",
                "user": "U_BOT",
            },
        }).encode()
        headers = _sign_request(body, _make_signing_secret())

        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 200

        mock_dispatcher.dispatch.assert_not_called()
