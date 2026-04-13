"""Unit tests for peoplesearch modules (wikipedia client)."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

# Load the wikipedia client directly, bypassing tools/peoplesearch/__init__.py
# which chains: search.py → suggester.py → from langchain.output_parsers import ...
_CLIENT_PATH = Path(__file__).resolve().parent.parent / "tools" / "peoplesearch" / "wikipedia" / "client.py"
_spec = importlib.util.spec_from_file_location("ps_wiki_client", _CLIENT_PATH)
_ps_wiki_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps_wiki_client)
get_person_info = _ps_wiki_client.get_person_info


# ============================================================
# Wikipedia peoplesearch client tests
# ============================================================


class TestGetPersonInfo:
    def _make_wiki_response(self, page_id: str, has_image: bool = True) -> dict:
        page = {
            "title": "Albert Einstein",
            "extract": "Albert Einstein was a theoretical physicist.",
            "fullurl": "https://en.wikipedia.org/wiki/Albert_Einstein",
        }
        if has_image:
            page["thumbnail"] = {"source": "https://upload.wikimedia.org/thumb/einstein.jpg"}

        return {"query": {"pages": {page_id: page}}}

    def test_returns_person_info_on_success(self):

        data = self._make_wiki_response("12345")
        mock_response = MagicMock()
        mock_response.json.return_value = data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = get_person_info("Albert Einstein")

        assert result is not None
        assert result["name"] == "Albert Einstein"
        assert result["image"] is not None
        assert "wikipedia.org" in result["wikiUrl"]

    def test_returns_none_when_page_not_found(self):

        data = {"query": {"pages": {"-1": {"title": "UnknownPerson"}}}}
        mock_response = MagicMock()
        mock_response.json.return_value = data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = get_person_info("UnknownPerson12345xyz")

        assert result is None

    def test_returns_none_when_no_image(self):

        data = self._make_wiki_response("12345", has_image=False)
        mock_response = MagicMock()
        mock_response.json.return_value = data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = get_person_info("NoImagePerson")

        assert result is None

    def test_returns_none_on_request_error(self):

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("error")):
            result = get_person_info("Someone")

        assert result is None

    def test_returns_none_when_no_pages(self):

        data = {"query": {"pages": {}}}
        mock_response = MagicMock()
        mock_response.json.return_value = data
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = get_person_info("EmptyResult")

        assert result is None
