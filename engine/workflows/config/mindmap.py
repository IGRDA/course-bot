"""Mind map generation configuration."""

from pydantic import BaseModel, Field


class MindmapConfig(BaseModel):
    """Configuration for mind map generation."""
    
    enabled: bool = Field(
        default=False,
        description="Generate mind map for each module"
    )
    max_nodes: int = Field(
        default=20,
        description="Maximum number of nodes in the mind map (including root)"
    )
    llm_provider: str = Field(
        default="",
        description="LLM provider for mind map generation (empty = use text_llm_provider)"
    )

