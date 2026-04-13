"""
Podcast Generator Agent.

Generates two-speaker educational dialogue conversations from course content
and synthesizes them into podcast audio using TTS.
"""

from .agent import (
    LANGUAGE_MAP,
    extract_module_context,
    generate_conversation,
    generate_module_podcast,
    get_tts_language,
)

__all__ = [
    "LANGUAGE_MAP",
    "extract_module_context",
    "generate_conversation",
    "generate_module_podcast",
    "get_tts_language",
]
