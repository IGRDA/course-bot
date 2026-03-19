"""Video generation configuration."""

from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    """Configuration for video recommendations generation."""
    
    enabled: bool = Field(
        default=False,
        description="Generate video recommendations for each module"
    )
    videos_per_module: int = Field(
        default=3,
        description="Number of videos to recommend per module"
    )
    search_provider: str = Field(
        default="youtube",
        description="Video search provider (youtube | bing)"
    )

