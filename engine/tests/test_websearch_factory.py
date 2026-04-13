"""Unit tests for tools/websearch/factory.py"""

import pytest

from tools.websearch.factory import available_search_providers, create_web_search


class TestAvailableSearchProviders:
    def test_returns_list(self):
        result = available_search_providers()
        assert isinstance(result, list)

    def test_returns_three_providers(self):
        assert len(available_search_providers()) == 3

    def test_known_providers_present(self):
        providers = available_search_providers()
        assert "ddg" in providers
        assert "tavily" in providers
        assert "wikipedia" in providers

    def test_sorted(self):
        providers = available_search_providers()
        assert providers == sorted(providers)


class TestCreateWebSearch:
    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="Provider is required"):
            create_web_search("")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_web_search("bing")

    def test_unknown_provider_error_lists_available(self):
        try:
            create_web_search("badprovider")
        except ValueError as e:
            msg = str(e)
            assert any(p in msg for p in ["ddg", "tavily", "wikipedia"])

    def test_ddg_returns_callable(self):
        func = create_web_search("ddg")
        assert callable(func)

    def test_tavily_returns_callable(self):
        func = create_web_search("tavily")
        assert callable(func)

    def test_wikipedia_returns_callable(self):
        func = create_web_search("wikipedia")
        assert callable(func)

    def test_case_insensitive(self):
        # Should not raise
        func = create_web_search("DDG")
        assert callable(func)
