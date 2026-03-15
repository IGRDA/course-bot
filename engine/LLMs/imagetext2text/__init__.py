"""
Factory utilities for building image-to-text (vision) LLM clients.

This module exposes a simple factory so that agents can request a vision model
without knowing the underlying provider (Pixtral, etc.). Adding new
providers only requires registering another builder in ``factory.py``.
"""

from .factory import (
    available_vision_llms,
    create_vision_llm,
    resolve_vision_model_name,
)

__all__ = ["available_vision_llms", "create_vision_llm", "resolve_vision_model_name"]

