"""
Factory for creating image search client instances.

This module provides a unified interface for image search across
different providers (DuckDuckGo, Bing, Freepik, Google).

Uses lazy imports so that heavy optional dependencies (playwright,
playwright_stealth, etc.) are only loaded when the caller requests
a provider that needs them.
"""

from typing import Callable

ImageSearchFunc = Callable[[str, int], list[dict]]

# Provider names registered (no eager imports)
_PROVIDER_NAMES: list[str] = ["bing", "ddg", "freepik", "google"]


def available_image_search_providers() -> list[str]:
    """Return the list of registered image search providers."""
    return sorted(_PROVIDER_NAMES)


def _get_search_func(provider: str) -> ImageSearchFunc | None:
    """Lazily import and return the search function for *provider*."""
    if provider == "ddg":
        from .ddg.client import search_images
        return search_images
    elif provider == "bing":
        from .bing.client import search_images
        return search_images
    elif provider == "freepik":
        from .freepik.client import search_images
        return search_images
    elif provider == "google":
        from .google.client import search_images
        return search_images
    else:
        return None


def create_image_search(provider: str) -> ImageSearchFunc:
    """
    Get image search function for the specified provider.
    
    Args:
        provider: Image search provider name (ddg | bing | freepik | google).
        
    Returns:
        An image search function that accepts (query: str, max_results: int).
        
    Raises:
        ValueError: If provider is empty or not supported.
    """
    if not provider:
        available = ", ".join(available_image_search_providers())
        raise ValueError(f"Provider is required. Available providers: {available}")
    
    key = provider.lower()
    func = _get_search_func(key)
    if func is None:
        available = ", ".join(available_image_search_providers())
        raise ValueError(
            f"Unsupported image search provider '{provider}'. "
            f"Available providers: {available}"
        )
    return func
