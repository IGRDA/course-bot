"""Research phase configuration."""

from pydantic import BaseModel, Field


class ResearchConfig(BaseModel):
    """Configuration for the research phase before index generation."""
    
    enabled: bool = Field(
        default=True,
        description="Enable research phase before index generation"
    )
    max_queries: int = Field(
        default=5,
        description="Maximum number of search queries to generate"
    )
    max_results_per_query: int = Field(
        default=3,
        description="Maximum results per search query"
    )

