"""Unit tests for imagesearch client modules (bing, ddg)."""

import json
import sys
import types
from unittest.mock import MagicMock, patch

# Stub bs4 (beautifulsoup4 not installed in test environment)
_bs4_stub = types.ModuleType("bs4")
_bs4_stub.BeautifulSoup = MagicMock()
sys.modules.setdefault("bs4", _bs4_stub)


# ============================================================
# Bing image search tests
# ============================================================


class TestBingImageSearch:
    def _make_soup_with_results(self, image_urls: list[str]):
        """Build a mock soup object with iusc anchor tags (no real bs4 needed)."""
        mock_tags = []
        for url in image_urls:
            m_data = json.dumps({"murl": url, "turl": url, "t": "Test image", "purl": "example.com"})
            tag = MagicMock()
            tag.get.return_value = m_data
            mock_tags.append(tag)

        mock_soup = MagicMock()
        mock_soup.find_all.return_value = mock_tags
        return mock_soup

    def test_search_returns_list_on_success(self):
        from tools.imagesearch.bing.client import search_images

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None

        soup = self._make_soup_with_results(["https://example.com/img1.jpg", "https://example.com/img2.jpg"])

        with patch("requests.get", return_value=mock_response), patch("bs4.BeautifulSoup", return_value=soup):
            results = search_images("python programming", max_results=5)

        assert isinstance(results, list)

    def test_search_returns_error_on_exception(self):
        from tools.imagesearch.bing.client import search_images

        with patch("requests.get", side_effect=Exception("network error")):
            results = search_images("test")

        assert isinstance(results, list)
        assert len(results) > 0
        assert "error" in results[0]

    def test_search_respects_max_results(self):
        from tools.imagesearch.bing.client import search_images

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None

        urls = [f"https://example.com/img{i}.jpg" for i in range(10)]
        soup = self._make_soup_with_results(urls)

        with patch("requests.get", return_value=mock_response), patch("bs4.BeautifulSoup", return_value=soup):
            results = search_images("test", max_results=3)

        assert len(results) <= 3

    def test_search_license_filters_constant(self):
        from tools.imagesearch.bing.client import LICENSE_FILTERS

        assert isinstance(LICENSE_FILTERS, dict)
        assert "all" in LICENSE_FILTERS
        assert "public_domain" in LICENSE_FILTERS


# ============================================================
# DuckDuckGo image search tests
# ============================================================


class TestDdgImageSearch:
    def _mock_ddgs_module(self, images_return_value):
        """Create a mock duckduckgo_search module with DDGS context manager."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.images.return_value = images_return_value

        mock_module = MagicMock()
        mock_module.DDGS = MagicMock(return_value=mock_ddgs_instance)
        return mock_module

    def test_search_returns_list_on_success(self):
        from tools.imagesearch.ddg.client import search_images

        mock_module = self._mock_ddgs_module(
            [
                {
                    "image": "https://example.com/img1.jpg",
                    "thumbnail": "https://thumb1.jpg",
                    "title": "Image 1",
                    "source": "example.com",
                },
                {
                    "image": "https://example.com/img2.jpg",
                    "thumbnail": "https://thumb2.jpg",
                    "title": "Image 2",
                    "source": "example.com",
                },
            ]
        )

        with patch.dict(sys.modules, {"duckduckgo_search": mock_module}):
            results = search_images("cats", max_results=5)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/img1.jpg"

    def test_search_returns_error_on_exception(self):
        from tools.imagesearch.ddg.client import search_images

        mock_module = MagicMock()
        mock_module.DDGS.side_effect = Exception("ddg error")

        with patch.dict(sys.modules, {"duckduckgo_search": mock_module}):
            results = search_images("test")

        assert isinstance(results, list)
        assert len(results) > 0
        assert "error" in results[0]

    def test_search_maps_fields_correctly(self):
        from tools.imagesearch.ddg.client import search_images

        mock_module = self._mock_ddgs_module(
            [
                {
                    "image": "https://img.example.com/photo.jpg",
                    "thumbnail": "https://thumb.jpg",
                    "title": "Photo",
                    "source": "flickr.com",
                },
            ]
        )

        with patch.dict(sys.modules, {"duckduckgo_search": mock_module}):
            results = search_images("nature", max_results=1)

        assert results[0]["url"] == "https://img.example.com/photo.jpg"
        assert results[0]["thumbnail_url"] == "https://thumb.jpg"
        assert results[0]["description"] == "Photo"
        assert results[0]["author"] == "flickr.com"
