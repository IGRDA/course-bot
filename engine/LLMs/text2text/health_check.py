"""Pre-flight API key validation for all text LLM providers.

Reads ALL keys from ``env.secrets`` (the immutable source of truth),
tests each with a minimal request, and sets the environment variable
to only the healthy subset for the current process.

Supported providers:
    - **OpenAI-compatible** (Mistral, OpenAI, Groq, DeepSeek):
      ``POST /chat/completions`` with ``max_tokens=1``
    - **Gemini** (Google AI):
      ``POST /models/{model}:generateContent`` with ``maxOutputTokens=1``
"""

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from LLMs.api_keys import _load_all_keys_from_secrets, mask_key

PROVIDER_REGISTRY: dict[str, dict] = {
    "mistral": {
        "env_var": "MISTRAL_API_KEY",
        "model_env_var": "MISTRAL_MODEL_NAME",
        "base_url": "https://api.mistral.ai/v1",
        "api_format": "openai",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "model_env_var": "OPENAI_MODEL_NAME",
        "base_url": "https://api.openai.com/v1",
        "api_format": "openai",
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "model_env_var": "GROQ_MODEL_NAME",
        "base_url": "https://api.groq.com/openai/v1",
        "api_format": "openai",
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "model_env_var": "DEEPSEEK_MODEL_NAME",
        "base_url": "https://api.deepseek.com",
        "api_format": "openai",
        "default_model": "deepseek-chat",
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "model_env_var": "GEMINI_MODEL_NAME",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_format": "gemini",
    },
}


# ---------------------------------------------------------------------------
# Per-format health check probes
# ---------------------------------------------------------------------------


def _check_key_openai(key: str, base_url: str, model: str, timeout: float) -> int:
    """POST /chat/completions with max_tokens=1 (Mistral, OpenAI, Groq, DeepSeek)."""
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=timeout,
        )
        return resp.status_code
    except requests.Timeout:
        return 408
    except requests.RequestException:
        return 0


def _check_key_gemini(key: str, base_url: str, model: str, timeout: float) -> int:
    """POST /models/{model}:generateContent?key={key} (Google AI / Gemini)."""
    try:
        resp = requests.post(
            f"{base_url}/models/{model}:generateContent",
            params={"key": key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": 1},
            },
            timeout=timeout,
        )
        return resp.status_code
    except requests.Timeout:
        return 408
    except requests.RequestException:
        return 0


_CHECK_DISPATCH = {
    "openai": _check_key_openai,
    "gemini": _check_key_gemini,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_provider_keys(
    provider: str,
    timeout: float = 15.0,
) -> list[str]:
    """Test ALL keys for *provider* from env.secrets; keep only healthy ones.

    Reads the full key list from ``env.secrets`` every time so that keys
    temporarily rate-limited in a previous run are always retested.

    Args:
        provider: Provider name (mistral, openai, groq, deepseek, gemini).
        timeout:  Per-request timeout in seconds.

    Returns:
        List of healthy API keys.

    Raises:
        RuntimeError: When zero keys pass the health check.
    """
    provider = provider.lower()
    config = PROVIDER_REGISTRY.get(provider)

    if config is None:
        print(f"[APICheck] Provider '{provider}' has no health-check config — skipping.")
        return []

    env_var = config["env_var"]
    model_env_var = config["model_env_var"]
    base_url = config["base_url"]
    api_format = config["api_format"]
    default_model = config.get("default_model")

    all_keys = _load_all_keys_from_secrets(env_var)
    if not all_keys:
        raise RuntimeError(f"No API keys found for {env_var}")

    model = os.getenv(model_env_var) or default_model
    if not model:
        raise RuntimeError(f"{model_env_var} environment variable must be set for health check")

    check_fn = _CHECK_DISPATCH[api_format]

    print(f"\n[APICheck] Testing {len(all_keys)} key(s) from {env_var} (provider: {provider}, model: {model})...")

    status_counts: Counter = Counter()
    healthy: list[str] = []
    results: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=len(all_keys)) as pool:
        futures = {pool.submit(check_fn, k, base_url, model, timeout): k for k in all_keys}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    for key in all_keys:
        status = results[key]
        status_counts[status] += 1
        label = "  OK " if status == 200 else " SKIP"
        print(f"  [{label}] Key {mask_key(key)} -> {status}")
        if status == 200:
            healthy.append(key)

    print(f"[APICheck] Status summary: {dict(sorted(status_counts.items()))}")
    print(f"[APICheck] Healthy keys: {len(healthy)}/{len(all_keys)}")

    if not healthy:
        raise RuntimeError(
            f"All {len(all_keys)} {provider} API keys failed health check. Distribution: {dict(status_counts)}"
        )

    os.environ[env_var] = ",".join(healthy)
    return healthy
