"""Unit tests for videosearch client modules (youtube, bing)."""

import os
from unittest.mock import MagicMock, patch

import requests

# ============================================================
# YouTube utility function tests
# ============================================================


class TestParseIso8601Duration:
    def test_minutes_and_seconds(self):
        from tools.videosearch.youtube.client import _parse_iso8601_duration

        assert _parse_iso8601_duration("PT4M13S") == 253

    def test_hours_minutes_seconds(self):
        from tools.videosearch.youtube.client import _parse_iso8601_duration

        assert _parse_iso8601_duration("PT1H2M3S") == 3723

    def test_seconds_only(self):
        from tools.videosearch.youtube.client import _parse_iso8601_duration

        assert _parse_iso8601_duration("PT45S") == 45

    def test_minutes_only(self):
        from tools.videosearch.youtube.client import _parse_iso8601_duration

        assert _parse_iso8601_duration("PT10M") == 600

    def test_hours_only(self):
        from tools.videosearch.youtube.client import _parse_iso8601_duration

        assert _parse_iso8601_duration("PT2H") == 7200

    def test_empty_string_returns_zero(self):
        from tools.videosearch.youtube.client import _parse_iso8601_duration

        assert _parse_iso8601_duration("") == 0

    def test_invalid_format_returns_zero(self):
        from tools.videosearch.youtube.client import _parse_iso8601_duration

        assert _parse_iso8601_duration("invalid") == 0


class TestParseIsoTimestamp:
    def test_valid_timestamp(self):
        from tools.videosearch.youtube.client import _parse_iso_timestamp

        result = _parse_iso_timestamp("2023-01-15T12:00:00Z")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string_returns_zero(self):
        from tools.videosearch.youtube.client import _parse_iso_timestamp

        assert _parse_iso_timestamp("") == 0

    def test_invalid_format_returns_zero(self):
        from tools.videosearch.youtube.client import _parse_iso_timestamp

        assert _parse_iso_timestamp("not-a-date") == 0


# ============================================================
# YouTube search_videos tests
# ============================================================


class TestYoutubeSearchVideos:
    def test_search_with_api_key_returns_list(self):
        from tools.videosearch.youtube.client import search_videos

        search_mock = {
            "items": [
                {
                    "id": {"videoId": "abc123"},
                    "snippet": {
                        "title": "Test Video",
                        "channelTitle": "Test Channel",
                        "publishedAt": "2023-01-15T12:00:00Z",
                        "thumbnails": {"high": {"url": "https://example.com/thumb.jpg"}},
                    },
                }
            ]
        }
        details_mock = {
            "items": [
                {
                    "id": "abc123",
                    "contentDetails": {"duration": "PT5M30S"},
                    "statistics": {"viewCount": "1000", "likeCount": "50"},
                }
            ]
        }

        search_response = MagicMock()
        search_response.json.return_value = search_mock
        search_response.raise_for_status.return_value = None

        details_response = MagicMock()
        details_response.json.return_value = details_mock
        details_response.raise_for_status.return_value = None

        with (
            patch.dict(os.environ, {"YOUTUBE_API_KEY": "test_key"}),
            patch("requests.get", side_effect=[search_response, details_response]),
        ):
            results = search_videos("python tutorial", max_results=5)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["title"] == "Test Video"
        assert results[0]["duration"] == 330

    def test_search_without_api_key_falls_back(self):
        from tools.videosearch.youtube.client import search_videos

        env = {k: v for k, v in os.environ.items() if k != "YOUTUBE_API_KEY"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("tools.videosearch.youtube.client._search_via_ytdlp_flat", return_value=[]) as mock_fallback,
        ):
            results = search_videos("test")

        mock_fallback.assert_called_once()
        assert results == []

    def test_search_api_failure_falls_back(self):
        from tools.videosearch.youtube.client import search_videos

        with (
            patch.dict(os.environ, {"YOUTUBE_API_KEY": "test_key"}),
            patch("requests.get", side_effect=requests.exceptions.ConnectionError("error")),
            patch("tools.videosearch.youtube.client._search_via_ytdlp_flat", return_value=[]) as mock_fallback,
        ):
            search_videos("test")

        mock_fallback.assert_called_once()

    def test_search_empty_items_returns_empty(self):
        from tools.videosearch.youtube.client import search_videos

        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status.return_value = None

        with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test_key"}), patch("requests.get", return_value=mock_response):
            results = search_videos("obscure query xyz")

        assert results == []


# ============================================================
# Bing video search tests
# ============================================================


class TestBingVideoSearch:
    def test_search_returns_list_on_success(self):
        from tools.videosearch.bing.client import search_videos

        mock_response = MagicMock()
        mock_response.text = '"videoId":"abc12345678"  "videoId":"def12345678"'
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            results = search_videos("python tutorial", max_results=5)

        assert isinstance(results, list)

    def test_search_returns_error_on_request_exception(self):
        from tools.videosearch.bing.client import search_videos

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("error")):
            results = search_videos("test")

        assert isinstance(results, list)
        assert len(results) > 0
        assert "error" in results[0]

    def test_search_returns_error_on_generic_exception(self):
        from tools.videosearch.bing.client import search_videos

        with patch("requests.get", side_effect=Exception("generic error")):
            results = search_videos("test")

        assert isinstance(results, list)
        assert len(results) > 0
        assert "error" in results[0]

    def test_search_respects_max_results(self):
        from tools.videosearch.bing.client import search_videos

        # Generate 20 unique video IDs in the response text
        video_ids = " ".join([f'"videoId":"vid{i:09d}"' for i in range(20)])
        mock_response = MagicMock()
        mock_response.text = video_ids
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            results = search_videos("test", max_results=3)

        assert len(results) <= 3
