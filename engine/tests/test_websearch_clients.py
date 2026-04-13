"""Unit tests for websearch client modules (ddg, tavily, wikipedia)."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

# Stub langchain_community (not installed in test environment).
# patch() needs these modules to exist in sys.modules with stub attributes so
# that patch("langchain_community.utilities.DuckDuckGoSearchAPIWrapper", ...)
# can find and replace the attribute.
_lc = types.ModuleType("langchain_community")
_lc_utils = types.ModuleType("langchain_community.utilities")
_lc_tools = types.ModuleType("langchain_community.tools")
_lc_utils.DuckDuckGoSearchAPIWrapper = MagicMock()
_lc_utils.WikipediaAPIWrapper = MagicMock()
_lc_tools.WikipediaQueryRun = MagicMock()
_lc.utilities = _lc_utils
_lc.tools = _lc_tools
sys.modules.setdefault("langchain_community", _lc)
sys.modules.setdefault("langchain_community.utilities", _lc_utils)
sys.modules.setdefault("langchain_community.tools", _lc_tools)

# Stub langchain_tavily (not installed in test environment)
_lc_tavily = types.ModuleType("langchain_tavily")
_lc_tavily.TavilySearch = MagicMock()
sys.modules.setdefault("langchain_tavily", _lc_tavily)


# ============================================================
# DuckDuckGo client tests
# ============================================================


class TestDdgWebSearch:
    def test_returns_string_on_success(self):
        from tools.websearch.ddg.client import web_search

        mock_wrapper = MagicMock()
        mock_wrapper.run.return_value = "Search results for python"

        with patch("langchain_community.utilities.DuckDuckGoSearchAPIWrapper", return_value=mock_wrapper):
            result = web_search("python programming")

        assert isinstance(result, str)

    def test_returns_error_string_on_exception(self):
        from tools.websearch.ddg.client import web_search

        with patch("langchain_community.utilities.DuckDuckGoSearchAPIWrapper", side_effect=Exception("ddg error")):
            result = web_search("query")

        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()

    def test_passes_max_results(self):
        from tools.websearch.ddg.client import web_search

        mock_wrapper = MagicMock()
        mock_wrapper.run.return_value = "results"

        with patch("langchain_community.utilities.DuckDuckGoSearchAPIWrapper", return_value=mock_wrapper) as mock_cls:
            web_search("test", max_results=10, region="es-es")

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("max_results") == 10

    def test_passes_region(self):
        from tools.websearch.ddg.client import web_search

        mock_wrapper = MagicMock()
        mock_wrapper.run.return_value = "results"

        with patch("langchain_community.utilities.DuckDuckGoSearchAPIWrapper", return_value=mock_wrapper) as mock_cls:
            web_search("test", region="de-de")

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("region") == "de-de"


# ============================================================
# Tavily client tests
# ============================================================


class TestTavilyWebSearch:
    def test_returns_error_when_no_api_key(self):
        from tools.websearch.tavily.client import web_search

        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = web_search("query")

        assert "Error" in result or "not set" in result

    def test_returns_string_on_success(self):
        from tools.websearch.tavily.client import web_search

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Tavily results"

        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"}),
            patch("langchain_tavily.TavilySearch", return_value=mock_tool),
        ):
            result = web_search("python programming")

        assert isinstance(result, str)

    def test_returns_error_string_on_exception(self):
        from tools.websearch.tavily.client import web_search

        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"}),
            patch("langchain_tavily.TavilySearch", side_effect=Exception("tavily error")),
        ):
            result = web_search("query")

        assert isinstance(result, str)
        assert "failed" in result.lower() or "Search failed" in result


# ============================================================
# Wikipedia client tests
# ============================================================


class TestWikipediaWebSearch:
    def test_returns_string_on_success(self):
        from tools.websearch.wikipedia.client import web_search

        mock_tool = MagicMock()
        mock_tool.run.return_value = "Wikipedia article content"

        with (
            patch("langchain_community.tools.WikipediaQueryRun", return_value=mock_tool),
            patch("langchain_community.utilities.WikipediaAPIWrapper"),
        ):
            result = web_search("artificial intelligence")

        assert isinstance(result, str)

    def test_returns_error_string_on_exception(self):
        from tools.websearch.wikipedia.client import web_search

        with patch("langchain_community.tools.WikipediaQueryRun", side_effect=Exception("wiki error")):
            result = web_search("query")

        assert isinstance(result, str)
        assert "failed" in result.lower()
