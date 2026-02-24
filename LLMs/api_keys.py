"""Shared utilities for multi-API-key support across LLM providers.

Supports comma-separated API keys in environment variables.
A single key (no commas) works identically to a list of one.
"""

import random


def parse_api_keys(value: str | None) -> list[str]:
    """Parse a comma-separated string of API keys into a list.

    Args:
        value: Raw value from env var or kwarg, possibly comma-separated.

    Returns:
        List of stripped, non-empty API key strings.
        Empty list if value is None or empty.
    """
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]


def get_random_key(keys: list[str]) -> str:
    """Pick a random API key from the list.

    Args:
        keys: Non-empty list of API keys.

    Returns:
        A randomly selected key.
    """
    return random.choice(keys)


def mask_key(key: str) -> str:
    """Return a masked version of an API key for safe logging.

    Shows only the last 4 characters.

    Args:
        key: The full API key string.

    Returns:
        Masked string like ``...BvzA``.
    """
    if len(key) <= 4:
        return "***"
    return f"...{key[-4:]}"
