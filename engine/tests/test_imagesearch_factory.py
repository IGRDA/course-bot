"""Unit tests for tools/imagesearch/factory.py"""

import pytest

from tools.imagesearch.factory import available_image_search_providers, create_image_search


class TestAvailableImageSearchProviders:
    def test_returns_list(self):
        result = available_image_search_providers()
        assert isinstance(result, list)

    def test_returns_four_providers(self):
        assert len(available_image_search_providers()) == 4

    def test_known_providers_present(self):
        providers = available_image_search_providers()
        assert "bing" in providers
        assert "ddg" in providers
        assert "freepik" in providers
        assert "google" in providers

    def test_sorted(self):
        providers = available_image_search_providers()
        assert providers == sorted(providers)


class TestCreateImageSearch:
    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="Provider is required"):
            create_image_search("")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_image_search("shutterstock")

    def test_unknown_provider_error_lists_available(self):
        try:
            create_image_search("unknown")
        except ValueError as e:
            msg = str(e)
            assert any(p in msg for p in ["bing", "ddg", "freepik", "google"])

    def test_ddg_returns_callable(self):
        func = create_image_search("ddg")
        assert callable(func)

    def test_bing_returns_callable(self):
        func = create_image_search("bing")
        assert callable(func)

    def test_google_returns_callable(self):
        func = create_image_search("google")
        assert callable(func)

    def test_freepik_returns_callable(self):
        func = create_image_search("freepik")
        assert callable(func)
