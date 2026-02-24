"""Image generation configuration."""

from pydantic import BaseModel, Field


class ImageConfig(BaseModel):
    """Configuration for image search and generation."""
    
    search_provider: str = Field(
        default="freepik",
        description="Image search provider (bing | freepik | ddg | google)"
    )
    use_vision_ranking: bool = Field(
        default=False,
        description="Use vision LLM (Pixtral) to rank images; if False, picks first result"
    )
    num_to_fetch: int = Field(
        default=5,
        description="Number of images to fetch for ranking (only used if use_vision_ranking=True)"
    )
    vision_llm_provider: str = Field(
        default="pixtral",
        description="Vision LLM provider for image ranking (pixtral)"
    )
    concurrency: int = Field(
        default=3,
        description="Number of image blocks to process in parallel"
    )
    imagetext2text_concurrency: int = Field(
        default=5,
        description="Number of Pixtral vision LLM calls in parallel for image scoring"
    )
    vision_ranking_batch_size: int = Field(
        default=8,
        description="Number of images per batch for Pixtral ranking calls"
    )

