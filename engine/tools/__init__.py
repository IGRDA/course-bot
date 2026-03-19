"""Tools module for course generator utilities.

All sub-package imports are lazy so that heavy optional dependencies
(playwright, yt-dlp, etc.) are only loaded when the caller actually
requests a specific tool.
"""


def __getattr__(name: str):
    """Lazy module-level attribute access for tool exports."""
    if name in ("create_web_search", "available_search_providers"):
        from .websearch import create_web_search, available_search_providers
        return create_web_search if name == "create_web_search" else available_search_providers

    if name in ("create_image_search", "available_image_search_providers"):
        from .imagesearch import create_image_search, available_image_search_providers
        return create_image_search if name == "create_image_search" else available_image_search_providers

    if name in ("create_video_search", "available_video_search_providers", "VideoResult"):
        from .videosearch import create_video_search, available_video_search_providers, VideoResult
        _map = {
            "create_video_search": create_video_search,
            "available_video_search_providers": available_video_search_providers,
            "VideoResult": VideoResult,
        }
        return _map[name]

    if name in ("extract_images", "extract_images_from_urls"):
        from .web_image_extractor import extract_images, extract_images_from_urls
        return extract_images if name == "extract_images" else extract_images_from_urls

    if name == "scrape_website":
        from .web_scraper import scrape_website
        return scrape_website

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_web_search",
    "available_search_providers",
    "create_image_search",
    "available_image_search_providers",
    "create_video_search",
    "available_video_search_providers",
    "VideoResult",
    "extract_images",
    "extract_images_from_urls",
    "scrape_website",
]
