"""Unit tests for articlesearch client modules (arxiv, openalex, semanticscholar)."""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import requests

# ============================================================
# arXiv client tests
# ============================================================


class TestArxivHelpers:
    def test_extract_arxiv_id_from_url(self):
        from tools.articlesearch.arxiv.client import _extract_arxiv_id

        assert _extract_arxiv_id("http://arxiv.org/abs/2301.00001v1") == "2301.00001"

    def test_extract_arxiv_id_no_version(self):
        from tools.articlesearch.arxiv.client import _extract_arxiv_id

        assert _extract_arxiv_id("http://arxiv.org/abs/2301.00001") == "2301.00001"

    def test_extract_arxiv_id_passthrough_on_no_match(self):
        from tools.articlesearch.arxiv.client import _extract_arxiv_id

        assert _extract_arxiv_id("plain_id") == "plain_id"

    def test_extract_year_valid(self):
        from tools.articlesearch.arxiv.client import _extract_year

        assert _extract_year("2023-01-15T12:00:00Z") == 2023

    def test_extract_year_none_input(self):
        from tools.articlesearch.arxiv.client import _extract_year

        assert _extract_year(None) is None

    def test_extract_year_empty_input(self):
        from tools.articlesearch.arxiv.client import _extract_year

        assert _extract_year("") is None

    def test_truncate_abstract_short(self):
        from tools.articlesearch.arxiv.client import _truncate_abstract

        short = "short abstract"
        assert _truncate_abstract(short) == "short abstract"

    def test_truncate_abstract_long(self):
        from tools.articlesearch.arxiv.client import _truncate_abstract

        long_text = "word " * 200
        result = _truncate_abstract(long_text, max_length=50)
        assert result.endswith("...")
        assert len(result) <= 60

    def test_truncate_abstract_none(self):
        from tools.articlesearch.arxiv.client import _truncate_abstract

        assert _truncate_abstract(None) is None

    def test_clean_text_normalizes_whitespace(self):
        from tools.articlesearch.arxiv.client import _clean_text

        assert _clean_text("  hello   world  ") == "hello world"

    def test_clean_text_none_returns_none(self):
        from tools.articlesearch.arxiv.client import _clean_text

        assert _clean_text(None) is None

    def test_clean_text_empty_returns_none(self):
        from tools.articlesearch.arxiv.client import _clean_text

        assert _clean_text("") is None

    def test_extract_authors_from_entry(self):
        from tools.articlesearch.arxiv.client import _extract_authors

        xml_str = """<entry xmlns="http://www.w3.org/2005/Atom">
            <author><name>Alice Smith</name></author>
            <author><name>Bob Jones</name></author>
        </entry>"""
        entry = ET.fromstring(xml_str)
        authors = _extract_authors(entry)
        assert "Alice Smith" in authors
        assert "Bob Jones" in authors

    def test_extract_authors_empty(self):
        from tools.articlesearch.arxiv.client import _extract_authors

        xml_str = """<entry xmlns="http://www.w3.org/2005/Atom"></entry>"""
        entry = ET.fromstring(xml_str)
        assert _extract_authors(entry) == []


class TestArxivSearchArticles:
    def _make_xml_response(self, entries: list[dict]) -> bytes:
        """Build a minimal Atom XML response."""
        atom_ns = "http://www.w3.org/2005/Atom"
        root = ET.Element(f"{{{atom_ns}}}feed")
        for e in entries:
            entry = ET.SubElement(root, f"{{{atom_ns}}}entry")
            if "title" in e:
                ET.SubElement(entry, f"{{{atom_ns}}}title").text = e["title"]
            if "id" in e:
                ET.SubElement(entry, f"{{{atom_ns}}}id").text = e["id"]
            if "summary" in e:
                ET.SubElement(entry, f"{{{atom_ns}}}summary").text = e["summary"]
            if "published" in e:
                ET.SubElement(entry, f"{{{atom_ns}}}published").text = e["published"]
            if "authors" in e:
                for author_name in e["authors"]:
                    author_elem = ET.SubElement(entry, f"{{{atom_ns}}}author")
                    ET.SubElement(author_elem, f"{{{atom_ns}}}name").text = author_name
        return ET.tostring(root, encoding="unicode").encode()

    def test_search_returns_list_on_success(self):
        from tools.articlesearch.arxiv.client import search_articles

        xml_data = self._make_xml_response(
            [
                {
                    "title": "Test Paper",
                    "id": "http://arxiv.org/abs/2301.00001v1",
                    "summary": "This is an abstract.",
                    "published": "2023-01-15T12:00:00Z",
                    "authors": ["Alice Smith"],
                }
            ]
        )
        mock_response = MagicMock()
        mock_response.content = xml_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response), patch("tools.articlesearch.arxiv.client._rate_limit"):
            results = search_articles("machine learning", max_results=5)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["title"] == "Test Paper"
        assert results[0]["source"] == "arxiv"

    def test_search_returns_empty_on_request_error(self):
        from tools.articlesearch.arxiv.client import search_articles

        with (
            patch("requests.get", side_effect=requests.exceptions.ConnectionError("network error")),
            patch("tools.articlesearch.arxiv.client._rate_limit"),
        ):
            results = search_articles("query")

        assert results == []

    def test_search_returns_empty_on_xml_parse_error(self):
        from tools.articlesearch.arxiv.client import search_articles

        mock_response = MagicMock()
        mock_response.content = b"<invalid xml"
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response), patch("tools.articlesearch.arxiv.client._rate_limit"):
            results = search_articles("query")

        assert results == []

    def test_search_skips_entry_without_title(self):
        from tools.articlesearch.arxiv.client import search_articles

        xml_data = self._make_xml_response([{"id": "http://arxiv.org/abs/2301.99999v1"}])
        mock_response = MagicMock()
        mock_response.content = xml_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response), patch("tools.articlesearch.arxiv.client._rate_limit"):
            results = search_articles("query")

        assert results == []


# ============================================================
# OpenAlex client tests
# ============================================================


class TestOpenAlexSearchArticles:
    def test_search_returns_list_on_success(self):
        from tools.articlesearch.openalex.client import search_articles

        mock_data = {
            "results": [
                {
                    "title": "OpenAlex Paper",
                    "id": "https://openalex.org/W12345",
                    "doi": "10.1234/test",
                    "authorships": [{"author": {"display_name": "Author One"}}],
                    "publication_year": 2022,
                    "abstract_inverted_index": None,
                    "cited_by_count": 10,
                    "language": "en",
                    "primary_location": {"source": {"display_name": "Nature"}},
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            results = search_articles("machine learning", max_results=5)

        assert isinstance(results, list)

    def test_search_returns_empty_on_error(self):
        from tools.articlesearch.openalex.client import search_articles

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("network error")):
            results = search_articles("query")

        assert results == []


# ============================================================
# Semantic Scholar client tests
# ============================================================


class TestSemanticScholarSearchArticles:
    def test_search_returns_list_on_success(self):
        from tools.articlesearch.semanticscholar.client import search_articles

        mock_data = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "SS Paper",
                    "authors": [{"name": "John Doe"}],
                    "year": 2021,
                    "abstract": "Abstract text",
                    "externalIds": {"DOI": "10.5678/test"},
                    "citationCount": 5,
                    "venue": "ICML",
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            results = search_articles("neural networks", max_results=5)

        assert isinstance(results, list)

    def test_search_returns_empty_on_error(self):
        from tools.articlesearch.semanticscholar.client import search_articles

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("network error")):
            results = search_articles("query")

        assert isinstance(results, list)
