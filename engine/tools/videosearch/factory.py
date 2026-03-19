"""Video search factory with provider abstraction.

Uses lazy imports so that heavy optional dependencies (yt-dlp, etc.)
are only loaded when the caller requests a specific provider.
"""

from typing import Callable, List
from pydantic import BaseModel, Field


class VideoResult(BaseModel):
    """Video search result with full metadata."""
    title: str = Field(..., description="Video title")
    url: str = Field(..., description="YouTube video URL")
    duration: int = Field(..., description="Video duration in seconds")
    published_at: int = Field(..., description="Publication timestamp in milliseconds")
    thumbnail: str = Field(..., description="Thumbnail URL")
    channel: str = Field(..., description="Channel name")
    views: int = Field(..., description="View count")
    likes: int = Field(..., description="Like count")


VideoSearchFunc = Callable[[str, int], List[dict]]

# Provider names registered (no eager imports)
_PROVIDER_NAMES: list[str] = ["bing", "youtube"]


def available_video_search_providers() -> list[str]:
    """Return the list of registered video search providers."""
    return sorted(_PROVIDER_NAMES)


def _get_search_func(provider: str) -> VideoSearchFunc | None:
    """Lazily import and return the search function for *provider*."""
    if provider == "youtube":
        from .youtube.client import search_videos
        return search_videos
    elif provider == "bing":
        from .bing.client import search_videos
        return search_videos
    else:
        return None


def create_video_search(provider: str = "youtube") -> VideoSearchFunc:
    """
    Get video search function for the specified provider.
    
    Args:
        provider: Video search provider name. Options:
            - "youtube" (default): Full metadata via yt-dlp (title, url, duration,
              published_at, thumbnail, channel, views, likes)
            - "bing": Deprecated, minimal metadata (url only)
        
    Returns:
        A video search function that accepts (query: str, max_results: int).
    """
    if not provider:
        provider = "youtube"
    
    key = provider.lower()
    func = _get_search_func(key)
    if func is None:
        available = ", ".join(available_video_search_providers())
        raise ValueError(
            f"Unsupported video search provider '{provider}'. "
            f"Available providers: {available}"
        )
    return func
