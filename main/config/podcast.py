"""Podcast generation configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class PodcastConfig(BaseModel):
    """Configuration for podcast/audio generation."""
    
    target_words: int = Field(
        default=600,
        description="Target word count per module podcast"
    )
    tts_engine: Literal["edge", "coqui", "elevenlabs", "chatterbox", "openai_tts", "qwen_tts"] = Field(
        default="edge",
        description="TTS engine for podcast generation"
    )
    speaker_map: dict[str, str] | None = Field(
        default=None,
        description="Custom speaker mapping for podcast voices (e.g., {'host': 'es-ES-AlvaroNeural', 'guest': 'es-ES-XimenaNeural'})"
    )

