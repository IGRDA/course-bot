"""Article/paper search tools for academic research."""

from .factory import (
    ArticleResult,
    available_article_search_providers,
    create_article_search,
)

__all__ = [
    "ArticleResult",
    "available_article_search_providers",
    "create_article_search",
]
