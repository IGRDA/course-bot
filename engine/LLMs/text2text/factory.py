"""
Factory for creating text-to-text LLM instances.

This module provides a unified interface for instantiating LLMs from
different providers (Mistral, Gemini, Groq, OpenAI, DeepSeek).

Uses lazy imports so that provider-specific packages (langchain_mistralai,
langchain_google_genai, etc.) are only loaded when actually requested.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel

# Registered provider names (no eager imports of provider clients)
_PROVIDER_NAMES: list[str] = ["deepseek", "gemini", "groq", "mistral", "openai"]

MODEL_ENV_VARS: dict[str, str] = {
    "deepseek": "DEEPSEEK_MODEL_NAME",
    "gemini": "GEMINI_MODEL_NAME",
    "groq": "GROQ_MODEL_NAME",
    "mistral": "MISTRAL_MODEL_NAME",
    "openai": "OPENAI_MODEL_NAME",
}


def available_text_llms() -> list[str]:
    """Return the list of registered providers."""
    return sorted(_PROVIDER_NAMES)


def _get_builder(provider: str):
    """Lazily import and return the builder function for *provider*."""
    if provider == "deepseek":
        from .deepseek.client import build_deepseek_chat_model
        return build_deepseek_chat_model
    elif provider == "gemini":
        from .gemini.client import build_gemini_chat_model
        return build_gemini_chat_model
    elif provider == "groq":
        from .groq.client import build_groq_chat_model
        return build_groq_chat_model
    elif provider == "mistral":
        from .mistral.client import build_mistral_chat_model
        return build_mistral_chat_model
    elif provider == "openai":
        from .openai.client import build_openai_chat_model
        return build_openai_chat_model
    else:
        return None


def create_text_llm(provider: str, **kwargs) -> BaseChatModel:
    """
    Instantiate a text-to-text LLM using the registered factory builders.

    Args:
        provider: LLM provider name (mistral | gemini | groq | openai | deepseek).
        **kwargs: Extra keyword arguments forwarded to the underlying builder.

    Returns:
        A ``BaseChatModel`` ready for use in LangChain pipelines.
        
    Raises:
        ValueError: If provider is empty or not supported.
    """
    if not provider:
        raise ValueError(
            "Provider is required. Must be one of: deepseek, mistral, gemini, groq, openai"
        )

    key = provider.lower()
    builder = _get_builder(key)
    if builder is None:
        available = ", ".join(available_text_llms())
        raise ValueError(
            f"Unsupported text LLM provider '{provider}'. "
            f"Available providers: {available}"
        )

    return builder(**kwargs)


def resolve_text_model_name(provider: str) -> str | None:
    """
    Return the provider-specific model name from environment variable.
    
    Args:
        provider: LLM provider name (mistral | gemini | groq | openai | deepseek).
    
    Returns:
        Model name from environment variable, or None if not set.
    """
    if not provider:
        return None
    
    env_var = MODEL_ENV_VARS.get(provider.lower())
    return os.getenv(env_var) if env_var else None
