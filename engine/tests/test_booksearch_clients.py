"""Unit tests for booksearch client modules (openlibrary, googlebooks)."""

import os
from unittest.mock import MagicMock, patch

import requests

# ============================================================
# Open Library client tests
# ============================================================


class TestOpenLibraryHelpers:
    def test_format_author_name_first_last(self):
        from tools.booksearch.openlibrary.client import _format_author_name

        assert _format_author_name("John Doe") == "Doe, J."

    def test_format_author_name_single(self):
        from tools.booksearch.openlibrary.client import _format_author_name

        assert _format_author_name("Madonna") == "Madonna"

    def test_format_author_name_already_formatted(self):
        from tools.booksearch.openlibrary.client import _format_author_name

        result = _format_author_name("Smith, John")
        assert "Smith" in result

    def test_format_author_name_three_parts(self):
        from tools.booksearch.openlibrary.client import _format_author_name

        result = _format_author_name("John Michael Doe")
        assert "Doe" in result
        assert "J." in result

    def test_iso_to_openlibrary_lang_mapping(self):
        from tools.booksearch.openlibrary.client import ISO_TO_OPENLIBRARY_LANG

        assert ISO_TO_OPENLIBRARY_LANG["es"] == "spa"
        assert ISO_TO_OPENLIBRARY_LANG["en"] == "eng"
        assert ISO_TO_OPENLIBRARY_LANG["fr"] == "fre"


class TestOpenLibrarySearchBooks:
    def test_search_returns_list_on_success(self):
        from tools.booksearch.openlibrary.client import search_books

        mock_data = {
            "docs": [
                {
                    "key": "/works/OL12345W",
                    "title": "Python Programming",
                    "author_name": ["Guido van Rossum"],
                    "first_publish_year": 2000,
                    "publisher": ["O'Reilly"],
                    "isbn": ["1234567890", "1234567890123"],
                    "cover_i": 12345,
                    "edition_count": 3,
                    "language": ["eng"],
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            results = search_books("Python", max_results=5)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["title"] == "Python Programming"

    def test_search_returns_empty_on_error(self):
        from tools.booksearch.openlibrary.client import search_books

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("network error")):
            results = search_books("query")

        assert results == []

    def test_search_with_language_filter(self):
        from tools.booksearch.openlibrary.client import search_books

        mock_data = {"docs": []}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response) as mock_get:
            search_books("Python", language="es")

        call_args = mock_get.call_args
        params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
        assert "language" in params
        assert params["language"] == "spa"

    def test_search_handles_missing_fields(self):
        from tools.booksearch.openlibrary.client import search_books

        mock_data = {"docs": [{"title": "Minimal Book"}]}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            results = search_books("minimal")

        assert isinstance(results, list)

    def test_search_empty_docs_returns_empty(self):
        from tools.booksearch.openlibrary.client import search_books

        mock_data = {"docs": []}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            results = search_books("unknown title xyz")

        assert results == []


# ============================================================
# Google Books client tests
# ============================================================


class TestGoogleBooksHelpers:
    def test_format_author_name_first_last(self):
        from tools.booksearch.googlebooks.client import _format_author_name

        result = _format_author_name("John Doe")
        assert "Doe" in result
        assert "J." in result

    def test_format_author_name_already_formatted(self):
        from tools.booksearch.googlebooks.client import _format_author_name

        assert _format_author_name("Smith, John") == "Smith, John"

    def test_format_author_name_single(self):
        from tools.booksearch.googlebooks.client import _format_author_name

        assert _format_author_name("Aristotle") == "Aristotle"


class TestGoogleBooksSearchBooks:
    def test_search_returns_list_on_success(self):
        from tools.booksearch.googlebooks.client import search_books

        mock_data = {
            "items": [
                {
                    "id": "abc123",
                    "volumeInfo": {
                        "title": "Clean Code",
                        "authors": ["Robert C. Martin"],
                        "publishedDate": "2008",
                        "publisher": "Prentice Hall",
                        "industryIdentifiers": [
                            {"type": "ISBN_13", "identifier": "9780132350884"},
                            {"type": "ISBN_10", "identifier": "0132350882"},
                        ],
                        "language": "en",
                        "description": "A great book",
                        "averageRating": 4.5,
                        "ratingsCount": 100,
                        "imageLinks": {"thumbnail": "https://example.com/thumb.jpg"},
                        "infoLink": "https://books.google.com/books?id=abc123",
                    },
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with (
            patch.dict(os.environ, {"GOOGLE_BOOKS_API_KEY": "test_key"}),
            patch("requests.get", return_value=mock_response),
        ):
            results = search_books("Clean Code", max_results=5)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["title"] == "Clean Code"

    def test_search_returns_empty_when_no_api_key(self):
        from tools.booksearch.googlebooks.client import search_books

        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_BOOKS_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            results = search_books("query")

        assert results == []

    def test_search_returns_empty_when_no_items(self):
        from tools.booksearch.googlebooks.client import search_books

        mock_data = {}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with (
            patch.dict(os.environ, {"GOOGLE_BOOKS_API_KEY": "test_key"}),
            patch("requests.get", return_value=mock_response),
        ):
            results = search_books("nonexistent book xyz")

        assert results == []

    def test_search_with_language_filter(self):
        from tools.booksearch.googlebooks.client import search_books

        mock_data = {"items": []}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with (
            patch.dict(os.environ, {"GOOGLE_BOOKS_API_KEY": "test_key"}),
            patch("requests.get", return_value=mock_response) as mock_get,
        ):
            search_books("Python", language="es")

        mock_get.assert_called_once()
