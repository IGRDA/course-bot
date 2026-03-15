"""Bibliography generation configuration."""

from pydantic import BaseModel, Field


class BibliographyConfig(BaseModel):
    """Configuration for bibliography generation (books + articles)."""
    
    enabled: bool = Field(
        default=False,
        description="Generate bibliography for the course"
    )
    books_per_module: int = Field(
        default=5,
        description="Number of books to recommend per module"
    )
    articles_per_module: int = Field(
        default=5,
        description="Number of academic articles to recommend per module"
    )
    search_provider: str = Field(
        default="openlibrary",
        description="Book search provider for validation (openlibrary)"
    )
    article_search_provider: str = Field(
        default="openalex",
        description="Article search provider (semanticscholar | openalex | arxiv)"
    )

