"""Unit tests for tools/booksearch/factory.py"""

import pytest

from tools.booksearch.factory import available_book_search_providers, create_book_search


class TestAvailableBookSearchProviders:
    def test_returns_list(self):
        assert isinstance(available_book_search_providers(), list)

    def test_returns_two_providers(self):
        assert len(available_book_search_providers()) == 2

    def test_known_providers_present(self):
        providers = available_book_search_providers()
        assert "googlebooks" in providers
        assert "openlibrary" in providers

    def test_sorted(self):
        providers = available_book_search_providers()
        assert providers == sorted(providers)


class TestCreateBookSearch:
    def test_empty_provider_uses_default(self):
        func = create_book_search("")
        assert callable(func)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_book_search("badprovider")

    def test_unknown_provider_error_lists_available(self):
        try:
            create_book_search("badprovider")
        except ValueError as e:
            msg = str(e)
            assert "googlebooks" in msg or "openlibrary" in msg

    def test_googlebooks_returns_callable(self):
        func = create_book_search("googlebooks")
        assert callable(func)

    def test_openlibrary_returns_callable(self):
        func = create_book_search("openlibrary")
        assert callable(func)

    def test_case_insensitive(self):
        func = create_book_search("OpenLibrary")
        assert callable(func)
