"""
Factory for creating web search client instances.

This module provides a unified interface for web search across
different providers (DuckDuckGo, Tavily, Wikipedia).

Uses lazy imports so that provider-specific packages (langchain_community,
langchain_tavily, etc.) are only loaded when the caller requests them.
"""

from typing import Callable

WebSearchFunc = Callable[[str, int], str]

# Provider names registered (no eager imports)
_PROVIDER_NAMES: list[str] = ["ddg", "tavily", "wikipedia"]


def available_search_providers() -> list[str]:
    """Return the list of registered web search providers."""
    return sorted(_PROVIDER_NAMES)


def _get_search_func(provider: str) -> WebSearchFunc | None:
    """Lazily import and return the search function for *provider*."""
    if provider == "ddg":
        from .ddg.client import web_search
        return web_search
    elif provider == "tavily":
        from .tavily.client import web_search
        return web_search
    elif provider == "wikipedia":
        from .wikipedia.client import web_search
        return web_search
    else:
        return None


def create_web_search(provider: str) -> WebSearchFunc:
    """
    Get web search function for the specified provider.
    
    Args:
        provider: Web search provider name (ddg | tavily | wikipedia).
        
    Returns:
        A web search function that accepts (query: str, max_results: int).
        
    Raises:
        ValueError: If provider is empty or not supported.
    """
    if not provider:
        raise ValueError("Provider is required. Must be one of: ddg, tavily, wikipedia")
    
    key = provider.lower()
    func = _get_search_func(key)
    if func is None:
        available = ", ".join(available_search_providers())
        raise ValueError(
            f"Unsupported web search provider '{provider}'. "
            f"Available providers: {available}"
        )
    return func
