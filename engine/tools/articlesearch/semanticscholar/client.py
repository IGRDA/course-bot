"""
Semantic Scholar API client for academic paper search.

Semantic Scholar provides free access to academic paper metadata,
abstracts, and citation counts. No API key required for basic usage,
but rate limits apply (100 requests per 5 minutes without key).

API Documentation: https://api.semanticscholar.org/api-docs/
"""

import logging
import os
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from ..factory import ArticleResult

logger = logging.getLogger(__name__)

# API endpoints
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Fields to request from the API
SEARCH_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "citationCount",
    "venue",
    "externalIds",
    "url",
]


def _format_author_name(author: dict) -> str:
    """
    Format author dict to simple name string.
    
    Args:
        author: Author dict with 'name' field
        
    Returns:
        Author name string
    """
    return author.get("name", "Unknown Author")


def _extract_doi(external_ids: dict | None) -> str | None:
    """
    Extract DOI from external IDs dict.
    
    Args:
        external_ids: Dict of external identifiers
        
    Returns:
        DOI string or None
    """
    if not external_ids:
        return None
    return external_ids.get("DOI")


def _build_url(paper_id: str, doi: str | None) -> str:
    """
    Build URL to the paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        doi: Optional DOI
        
    Returns:
        URL to the paper
    """
    if doi:
        return f"https://doi.org/{doi}"
    return f"https://www.semanticscholar.org/paper/{paper_id}"


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
    if len(abstract) <= max_length:
        return abstract
    return abstract[:max_length].rsplit(" ", 1)[0] + "..."


def search_articles(
    query: str,
    max_results: int = 10,
    language: str | None = None,
) -> list["ArticleResult"]:
    """
    Search Semantic Scholar for academic papers.
    
    Note: Semantic Scholar doesn't support native language filtering.
    The language parameter is accepted for API compatibility but ignored.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (max 100)
        language: Not supported - included for API compatibility
        
    Returns:
        List of ArticleResult dictionaries with paper metadata
    """
    from ..factory import ArticleResult
    
    if language:
        logger.debug(
            "Semantic Scholar doesn't support language filtering. "
            "Consider using OpenAlex for language-specific searches."
        )
    
    # Build request parameters
    params = {
        "query": query,
        "limit": min(max_results, 100),  # API max is 100
        "fields": ",".join(SEARCH_FIELDS),
    }
    
    # Add API key if available (for higher rate limits)
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        response = requests.get(
            SEARCH_URL,
            params=params,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return []
    
    results: list[ArticleResult] = []
    
    for paper in data.get("data", [])[:max_results]:
        if not paper.get("title"):
            continue
        
        # Extract fields
        paper_id = paper.get("paperId", "")
        external_ids = paper.get("externalIds", {})
        doi = _extract_doi(external_ids)
        authors = paper.get("authors", [])
        abstract = paper.get("abstract")
        
        result: ArticleResult = {
            "title": paper.get("title", "Unknown Title"),
            "authors": [_format_author_name(a) for a in authors],
            "year": paper.get("year"),
            "abstract": abstract,
            "doi": doi,
            "url": _build_url(paper_id, doi),
            "citation_count": paper.get("citationCount"),
            "source": "semanticscholar",
            "language": None,  # Not provided by Semantic Scholar
            "venue": paper.get("venue") or None,
            "snippet": _truncate_abstract(abstract),
        }
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Quick test
    import json
    
    query = "transformer neural networks"
    print(f"🔍 Searching Semantic Scholar for: '{query}'")
    print("-" * 60)
    
    results = search_articles(query, max_results=3)
    
    for i, article in enumerate(results, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Authors: {', '.join(article['authors'][:3])}")
        print(f"   Year: {article['year']}")
        print(f"   Citations: {article['citation_count']}")
        print(f"   URL: {article['url']}")

