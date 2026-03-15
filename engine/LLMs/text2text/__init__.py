"""
Factory utilities for building text-to-text LLM clients.

This module exposes a simple factory so that agents can request a chat model
without knowing the underlying provider (Mistral, OpenAI, etc.). Adding new
providers only requires registering another builder in ``factory.py``.
"""

from .factory import (
    available_text_llms,
    create_text_llm,
    resolve_text_model_name,
)

__all__ = ["available_text_llms", "create_text_llm", "resolve_text_model_name"]

