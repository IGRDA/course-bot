"""Unit tests for tools/articlesearch/factory.py"""

import pytest

from tools.articlesearch.factory import (
    available_article_search_providers,
    create_article_search,
)


class TestAvailableArticleSearchProviders:
    def test_returns_list(self):
        assert isinstance(available_article_search_providers(), list)

    def test_returns_three_providers(self):
        assert len(available_article_search_providers()) == 3

    def test_known_providers_present(self):
        providers = available_article_search_providers()
        assert "semanticscholar" in providers
        assert "openalex" in providers
        assert "arxiv" in providers

    def test_sorted(self):
        providers = available_article_search_providers()
        assert providers == sorted(providers)


class TestCreateArticleSearch:
    def test_empty_provider_uses_default(self):
        func = create_article_search("")
        assert callable(func)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_article_search("badprovider")

    def test_unknown_provider_error_lists_available(self):
        try:
            create_article_search("badprovider")
        except ValueError as e:
            msg = str(e)
            assert "semanticscholar" in msg or "arxiv" in msg

    def test_semanticscholar_returns_callable(self):
        func = create_article_search("semanticscholar")
        assert callable(func)

    def test_openalex_returns_callable(self):
        func = create_article_search("openalex")
        assert callable(func)

    def test_arxiv_returns_callable(self):
        func = create_article_search("arxiv")
        assert callable(func)

    def test_case_insensitive(self):
        func = create_article_search("ArXiv")
        assert callable(func)

    def test_default_provider(self):
        func = create_article_search()
        assert callable(func)
