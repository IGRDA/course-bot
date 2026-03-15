"""
Factory for creating book search client instances.

This module provides a unified interface for book search across
different providers (Open Library, Google Books).

Uses lazy imports so that provider-specific packages are only loaded
when the caller requests a specific provider.
"""

from typing import Callable

# Provider names registered (no eager imports)
_PROVIDER_NAMES: list[str] = ["googlebooks", "openlibrary"]


def available_book_search_providers() -> list[str]:
    """Return the list of registered book search providers."""
    return sorted(_PROVIDER_NAMES)


def _get_search_func(provider: str):
    """Lazily import and return the search function for *provider*."""
    if provider == "openlibrary":
        from .openlibrary.client import search_books
        return search_books
    elif provider == "googlebooks":
        from .googlebooks.client import search_books
        return search_books
    else:
        return None


def create_book_search(provider: str = "openlibrary"):
    """
    Get book search function for the specified provider.
    
    Args:
        provider: Book search provider name (openlibrary | googlebooks).
        
    Returns:
        A book search function that accepts (query: str, max_results: int).
        
    Raises:
        ValueError: If provider is not supported.
    """
    if not provider:
        provider = "openlibrary"
    
    key = provider.lower()
    func = _get_search_func(key)
    if func is None:
        available = ", ".join(available_book_search_providers())
        raise ValueError(
            f"Unsupported book search provider '{provider}'. "
            f"Available providers: {available}"
        )
    return func
