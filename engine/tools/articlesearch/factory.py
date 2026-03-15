"""
Factory for creating article search client instances.

This module provides a unified interface for academic article/paper search
across different providers (Semantic Scholar, OpenAlex, arXiv).
"""

from typing import Callable, TypedDict


class ArticleResult(TypedDict):
    """Structured article result from academic search APIs.
    
    Attributes:
        title: Article/paper title (required)
        authors: List of author names (required)
        year: Publication year
        abstract: Article abstract/summary
        doi: Digital Object Identifier
        url: URL to the article page (required)
        citation_count: Number of citations
        source: Provider name (e.g., "semanticscholar", "openalex", "arxiv")
        language: ISO 639-1 language code (e.g., "en", "es")
        venue: Journal or conference name
        snippet: Short preview text for search results
    """
    title: str
    authors: list[str]
    year: int | None
    abstract: str | None
    doi: str | None
    url: str
    citation_count: int | None
    source: str
    language: str | None
    venue: str | None
    snippet: str | None


ArticleSearchFunc = Callable[[str, int, str | None], list[ArticleResult]]

# Provider registry - will be populated as clients are implemented
ARTICLE_SEARCH_PROVIDERS: dict[str, ArticleSearchFunc] = {}


def _register_providers() -> None:
    """Lazily register all available providers."""
    global ARTICLE_SEARCH_PROVIDERS
    
    if ARTICLE_SEARCH_PROVIDERS:
        return  # Already registered
    
    # Import and register providers
    from .semanticscholar.client import search_articles as semanticscholar_search
    from .openalex.client import search_articles as openalex_search
    from .arxiv.client import search_articles as arxiv_search
    
    ARTICLE_SEARCH_PROVIDERS.update({
        "semanticscholar": semanticscholar_search,
        "openalex": openalex_search,
        "arxiv": arxiv_search,
    })


def available_article_search_providers() -> list[str]:
    """Return the list of registered article search providers."""
    _register_providers()
    return sorted(ARTICLE_SEARCH_PROVIDERS.keys())


def create_article_search(provider: str = "semanticscholar") -> ArticleSearchFunc:
    """
    Get article search function for the specified provider.
    
    Args:
        provider: Article search provider name (semanticscholar | openalex | arxiv).
        
    Returns:
        An article search function that accepts (query: str, max_results: int, language: str | None).
        
    Raises:
        ValueError: If provider is not supported.
        
    Example:
        >>> search = create_article_search("semanticscholar")
        >>> results = search("machine learning", max_results=5, language=None)
    """
    _register_providers()
    
    if not provider:
        provider = "semanticscholar"
    
    key = provider.lower()
    try:
        return ARTICLE_SEARCH_PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(available_article_search_providers())
        raise ValueError(
            f"Unsupported article search provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc

