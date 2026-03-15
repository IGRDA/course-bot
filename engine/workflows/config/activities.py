"""Activities generation configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class ActivitiesConfig(BaseModel):
    """Configuration for section activities generation."""
    
    concurrency: int = Field(
        default=8,
        description="Number of concurrent section activity generations"
    )
    selection_mode: Literal["random", "deterministic"] = Field(
        default="deterministic",
        description="How to select activity types"
    )
    sections_per_activity: int = Field(
        default=1,
        description="Generate activities every N sections within each submodule (1 = every section)"
    )

