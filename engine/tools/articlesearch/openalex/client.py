"""
OpenAlex API client for academic paper search.

OpenAlex is a fully open catalog of the global research system.
No API key required, completely free. Supports native language filtering.

API Documentation: https://docs.openalex.org/
"""

import logging
import os
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from ..factory import ArticleResult

logger = logging.getLogger(__name__)

# API endpoints
SEARCH_URL = "https://api.openalex.org/works"

# ISO 639-1 to OpenAlex language code mapping
# OpenAlex uses ISO 639-1 codes directly
LANGUAGE_CODES = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
    "ru": "ru",
    "it": "it",
    "nl": "nl",
    "pl": "pl",
    "ar": "ar",
}


def _format_author_name(authorship: dict) -> str:
    """
    Format authorship dict to author name string.
    
    Args:
        authorship: Authorship dict with nested author info
        
    Returns:
        Author name string
    """
    author = authorship.get("author", {})
    return author.get("display_name", "Unknown Author")


def _extract_doi(doi_url: str | None) -> str | None:
    """
    Extract DOI from full DOI URL.
    
    Args:
        doi_url: Full DOI URL (e.g., https://doi.org/10.1234/example)
        
    Returns:
        DOI string without URL prefix
    """
    if not doi_url:
        return None
    # Remove https://doi.org/ prefix
    if doi_url.startswith("https://doi.org/"):
        return doi_url[16:]
    return doi_url


def _extract_abstract(abstract_inverted_index: dict | None) -> str | None:
    """
    Reconstruct abstract from OpenAlex inverted index format.
    
    OpenAlex stores abstracts as inverted indexes for efficiency.
    Format: {"word": [position1, position2, ...], ...}
    
    Args:
        abstract_inverted_index: Inverted index dict
        
    Returns:
        Reconstructed abstract text
    """
    if not abstract_inverted_index:
        return None
    
    try:
        # Reconstruct from inverted index
        words_with_positions = []
        for word, positions in abstract_inverted_index.items():
            for pos in positions:
                words_with_positions.append((pos, word))
        
        # Sort by position and join
        words_with_positions.sort(key=lambda x: x[0])
        abstract = " ".join(word for _, word in words_with_positions)
        return abstract
    except Exception as e:
        logger.debug(f"Failed to reconstruct abstract: {e}")
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
    if len(abstract) <= max_length:
        return abstract
    return abstract[:max_length].rsplit(" ", 1)[0] + "..."


def _get_venue_name(source: dict | None) -> str | None:
    """
    Extract venue/journal name from source dict.
    
    Args:
        source: Primary location source dict
        
    Returns:
        Venue name or None
    """
    if not source:
        return None
    return source.get("display_name")


def search_articles(
    query: str,
    max_results: int = 10,
    language: str | None = None,
) -> list["ArticleResult"]:
    """
    Search OpenAlex for academic papers.
    
    Supports native language filtering using ISO 639-1 codes.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (max 200)
        language: ISO 639-1 language code (e.g., "en", "es", "fr")
        
    Returns:
        List of ArticleResult dictionaries with paper metadata
    """
    from ..factory import ArticleResult
    
    # Build filter string
    filters = []
    if language:
        lang_code = LANGUAGE_CODES.get(language.lower(), language.lower())
        filters.append(f"language:{lang_code}")
    
    # Build request parameters
    params = {
        "search": query,
        "per_page": min(max_results, 200),  # API max is 200
        "select": "id,title,authorships,publication_year,cited_by_count,doi,primary_location,abstract_inverted_index,language",
    }
    
    if filters:
        params["filter"] = ",".join(filters)
    
    # Add polite pool email if available (recommended by OpenAlex)
    headers = {}
    email = os.getenv("OPENALEX_EMAIL")
    if email:
        params["mailto"] = email
    
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
        logger.error(f"OpenAlex search failed: {e}")
        return []
    
    results: list[ArticleResult] = []
    
    for work in data.get("results", [])[:max_results]:
        if not work.get("title"):
            continue
        
        # Extract fields
        doi = _extract_doi(work.get("doi"))
        authorships = work.get("authorships", [])
        abstract = _extract_abstract(work.get("abstract_inverted_index"))
        primary_location = work.get("primary_location", {}) or {}
        source = primary_location.get("source")
        
        # Build URL (prefer DOI, fall back to OpenAlex ID)
        work_id = work.get("id", "")
        if doi:
            url = f"https://doi.org/{doi}"
        elif work_id:
            url = work_id  # OpenAlex IDs are URLs
        else:
            url = ""
        
        result: ArticleResult = {
            "title": work.get("title", "Unknown Title"),
            "authors": [_format_author_name(a) for a in authorships[:10]],  # Limit authors
            "year": work.get("publication_year"),
            "abstract": abstract,
            "doi": doi,
            "url": url,
            "citation_count": work.get("cited_by_count"),
            "source": "openalex",
            "language": work.get("language"),
            "venue": _get_venue_name(source),
            "snippet": _truncate_abstract(abstract),
        }
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Quick test
    import json
    
    query = "machine learning"
    language = "es"
    print(f"🔍 Searching OpenAlex for: '{query}' (language={language})")
    print("-" * 60)
    
    results = search_articles(query, max_results=3, language=language)
    
    for i, article in enumerate(results, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Authors: {', '.join(article['authors'][:3])}")
        print(f"   Year: {article['year']}")
        print(f"   Citations: {article['citation_count']}")
        print(f"   Language: {article['language']}")
        print(f"   URL: {article['url']}")

