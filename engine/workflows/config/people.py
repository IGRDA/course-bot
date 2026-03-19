"""People generation configuration."""

from pydantic import BaseModel, Field


class PeopleConfig(BaseModel):
    """Configuration for relevant people generation."""
    
    enabled: bool = Field(
        default=False,
        description="Generate relevant people for each module"
    )
    people_per_module: int = Field(
        default=3,
        description="Number of relevant people to find per module"
    )
    llm_provider: str = Field(
        default="",
        description="LLM provider for people suggestion (empty = use text_llm_provider)"
    )
    concurrency: int = Field(
        default=5,
        description="Number of parallel Wikipedia API calls"
    )

