"""Unit tests for tools/videosearch/factory.py"""

import pytest

from tools.videosearch.factory import (
    VideoResult,
    available_video_search_providers,
    create_video_search,
)


class TestAvailableVideoSearchProviders:
    def test_returns_list(self):
        assert isinstance(available_video_search_providers(), list)

    def test_returns_two_providers(self):
        assert len(available_video_search_providers()) == 2

    def test_known_providers_present(self):
        providers = available_video_search_providers()
        assert "bing" in providers
        assert "youtube" in providers

    def test_sorted(self):
        providers = available_video_search_providers()
        assert providers == sorted(providers)


class TestVideoResult:
    def test_create_with_required_fields(self):
        result = VideoResult(
            title="Test Video",
            url="https://youtube.com/watch?v=abc",
            duration=120,
            published_at=0,
            thumbnail="",
            channel="Test Channel",
            views=1000,
            likes=50,
        )
        assert result.title == "Test Video"
        assert result.url == "https://youtube.com/watch?v=abc"
        assert result.duration == 120
        assert result.views == 1000

    def test_default_values(self):
        result = VideoResult(
            title="",
            url="",
            duration=0,
            published_at=0,
            thumbnail="",
            channel="",
            views=0,
            likes=0,
        )
        assert result.duration == 0
        assert result.likes == 0


class TestCreateVideoSearch:
    def test_empty_provider_uses_default(self):
        func = create_video_search("")
        assert callable(func)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_video_search("badprovider")

    def test_unknown_provider_error_lists_available(self):
        try:
            create_video_search("badprovider")
        except ValueError as e:
            msg = str(e)
            assert "bing" in msg or "youtube" in msg

    def test_youtube_returns_callable(self):
        func = create_video_search("youtube")
        assert callable(func)

    def test_bing_returns_callable(self):
        func = create_video_search("bing")
        assert callable(func)

    def test_case_insensitive(self):
        func = create_video_search("YouTube")
        assert callable(func)
