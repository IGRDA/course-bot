"""
arXiv API client for preprint paper search.

arXiv is an open-access repository for scientific preprints,
primarily in physics, mathematics, computer science, and related fields.
No API key required, but rate limiting applies (1 request per 3 seconds recommended).

API Documentation: https://info.arxiv.org/help/api/basics.html
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import requests

if TYPE_CHECKING:
    from ..factory import ArticleResult

logger = logging.getLogger(__name__)

# API endpoints
SEARCH_URL = "http://export.arxiv.org/api/query"

# XML namespaces used by arXiv API
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Rate limiting - arXiv recommends max 1 request per 3 seconds
_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 3.0


def _rate_limit() -> None:
    """Enforce rate limiting for arXiv API."""
    global _last_request_time
    
    current_time = time.time()
    elapsed = current_time - _last_request_time
    
    if elapsed < _MIN_REQUEST_INTERVAL:
        sleep_time = _MIN_REQUEST_INTERVAL - elapsed
        logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    
    _last_request_time = time.time()


def _extract_arxiv_id(id_url: str) -> str:
    """
    Extract arXiv ID from full URL.
    
    Args:
        id_url: Full arXiv URL (e.g., http://arxiv.org/abs/2301.00001v1)
        
    Returns:
        arXiv ID (e.g., 2301.00001)
    """
    # Remove version suffix and extract ID
    match = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", id_url)
    if match:
        return match.group(1)
    return id_url


def _extract_doi(entry: ET.Element) -> str | None:
    """
    Extract DOI from arXiv entry if available.
    
    Args:
        entry: XML entry element
        
    Returns:
        DOI string or None
    """
    doi_elem = entry.find("arxiv:doi", NAMESPACES)
    if doi_elem is not None and doi_elem.text:
        return doi_elem.text
    return None


def _extract_authors(entry: ET.Element) -> list[str]:
    """
    Extract author names from arXiv entry.
    
    Args:
        entry: XML entry element
        
    Returns:
        List of author names
    """
    authors = []
    for author in entry.findall("atom:author", NAMESPACES):
        name_elem = author.find("atom:name", NAMESPACES)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())
    return authors


def _extract_year(published: str | None) -> int | None:
    """
    Extract year from published date string.
    
    Args:
        published: ISO date string (e.g., 2023-01-15T12:00:00Z)
        
    Returns:
        Year as integer or None
    """
    if not published:
        return None
    try:
        return int(published[:4])
    except (ValueError, IndexError):
        return None


def _extract_categories(entry: ET.Element) -> str | None:
    """
    Extract primary category as venue-like field.
    
    Args:
        entry: XML entry element
        
    Returns:
        Primary category (e.g., "cs.LG") or None
    """
    primary = entry.find("arxiv:primary_category", NAMESPACES)
    if primary is not None:
        return primary.get("term")
    return None


def _truncate_abstract(abstract: str | None, max_length: int = 300) -> str | None:
    """
    Truncate abstract for snippet display.
    
    Args:
        abstract: Full abstract text
        max_length: Maximum length for snippet
        
    Returns:
        Truncated abstract or None
    """
    if not abstract:
        return None
    # Clean up whitespace
    abstract = " ".join(abstract.split())
    if len(abstract) <= max_length:
        return abstract
    return abstract[:max_length].rsplit(" ", 1)[0] + "..."


def _clean_text(text: str | None) -> str | None:
    """Clean up text by normalizing whitespace."""
    if not text:
        return None
    return " ".join(text.split())


def search_articles(
    query: str,
    max_results: int = 10,
    language: str | None = None,
) -> list["ArticleResult"]:
    """
    Search arXiv for preprint papers.
    
    Note: arXiv content is primarily in English. The language parameter
    is accepted for API compatibility but ignored.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (max 100 recommended)
        language: Not supported - arXiv is English-only
        
    Returns:
        List of ArticleResult dictionaries with paper metadata
    """
    from ..factory import ArticleResult
    
    if language and language.lower() != "en":
        logger.debug(
            "arXiv primarily contains English papers. "
            "Consider using OpenAlex for other languages."
        )
    
    # Enforce rate limiting
    _rate_limit()
    
    # Build query parameters
    # arXiv uses a specific query syntax
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 100),  # Keep reasonable for API
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    
    try:
        response = requests.get(
            SEARCH_URL,
            params=params,
            timeout=30,  # arXiv can be slow
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"arXiv search failed: {e}")
        return []
    
    # Parse XML response
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.error(f"Failed to parse arXiv response: {e}")
        return []
    
    results: list[ArticleResult] = []
    
    for entry in root.findall("atom:entry", NAMESPACES):
        # Extract title
        title_elem = entry.find("atom:title", NAMESPACES)
        title = _clean_text(title_elem.text) if title_elem is not None else None
        if not title:
            continue
        
        # Extract other fields
        id_elem = entry.find("atom:id", NAMESPACES)
        id_url = id_elem.text if id_elem is not None else ""
        arxiv_id = _extract_arxiv_id(id_url)
        
        summary_elem = entry.find("atom:summary", NAMESPACES)
        abstract = _clean_text(summary_elem.text) if summary_elem is not None else None
        
        published_elem = entry.find("atom:published", NAMESPACES)
        published = published_elem.text if published_elem is not None else None
        
        doi = _extract_doi(entry)
        authors = _extract_authors(entry)
        year = _extract_year(published)
        category = _extract_categories(entry)
        
        # Build URL (prefer DOI if available)
        if doi:
            url = f"https://doi.org/{doi}"
        else:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        
        result: ArticleResult = {
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "url": url,
            "citation_count": None,  # arXiv doesn't provide citation counts
            "source": "arxiv",
            "language": "en",  # arXiv is primarily English
            "venue": f"arXiv:{category}" if category else "arXiv",
            "snippet": _truncate_abstract(abstract),
        }
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Quick test
    query = "neural network"
    print(f"🔍 Searching arXiv for: '{query}'")
    print("-" * 60)
    
    results = search_articles(query, max_results=3)
    
    for i, article in enumerate(results, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Authors: {', '.join(article['authors'][:3])}")
        print(f"   Year: {article['year']}")
        print(f"   Venue: {article['venue']}")
        print(f"   URL: {article['url']}")

