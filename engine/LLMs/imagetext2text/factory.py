"""
Factory for creating vision (image+text → text) LLM instances.

Uses lazy imports so that provider-specific packages are only loaded
when actually requested.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel

# Registered provider names (no eager imports)
_PROVIDER_NAMES: list[str] = ["pixtral"]

MODEL_ENV_VARS = {
    "pixtral": "PIXTRAL_MODEL_NAME",
}


def available_vision_llms() -> list[str]:
    """Return the list of registered vision LLM providers."""
    return sorted(_PROVIDER_NAMES)


def _get_builder(provider: str):
    """Lazily import and return the builder function for *provider*."""
    if provider == "pixtral":
        from .pixtral.client import build_pixtral_chat_model
        return build_pixtral_chat_model
    else:
        return None


def create_vision_llm(provider: str, **kwargs) -> BaseChatModel:
    """
    Instantiate a vision LLM using the registered factory builders.

    Args:
        provider: Vision LLM provider name (pixtral).
        **kwargs: Extra keyword arguments forwarded to the underlying builder.

    Returns:
        A ``BaseChatModel`` ready for use in LangChain pipelines with vision support.
    """
    if not provider:
        raise ValueError(
            "Provider is required. Must be one of: pixtral"
        )

    key = provider.lower()
    builder = _get_builder(key)
    if builder is None:
        available = ", ".join(available_vision_llms())
        raise ValueError(
            f"Unsupported vision LLM provider '{provider}'. "
            f"Available providers: {available}"
        )

    return builder(**kwargs)


def resolve_vision_model_name(provider: str) -> str | None:
    """
    Return the provider-specific model name from environment variable.
    
    Args:
        provider: Vision LLM provider name (pixtral).
    
    Returns:
        Model name from environment variable, or None if not set.
    """
    if not provider:
        return None
    
    env_var = MODEL_ENV_VARS.get(provider.lower())
    return os.getenv(env_var) if env_var else None

