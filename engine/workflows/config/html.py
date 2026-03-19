"""HTML formatting configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class HtmlConfig(BaseModel):
    """Configuration for HTML structure generation."""
    
    concurrency: int = Field(
        default=8,
        description="Number of concurrent HTML structure generations"
    )
    select_mode: Literal["LLM", "random"] = Field(
        default="LLM",
        description="HTML format selection mode: LLM chooses or random selection"
    )
    formats: str = Field(
        default="paragraphs|accordion|tabs|carousel|flip|timeline|conversation",
        description="Available HTML formats (pipe-separated)"
    )
    random_seed: int = Field(
        default=42,
        description="Seed for deterministic random format selection"
    )
    include_quotes: bool = Field(
        default=False,
        description="Whether to include quote elements in HTML structure"
    )
    include_tables: bool = Field(
        default=False,
        description="Whether to include table elements in HTML structure"
    )

