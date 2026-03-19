import os

from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


def _is_retryable_error(exception: BaseException) -> bool:
    """
    Determine if an exception should trigger a retry.
    
    Returns True for:
    - 429 Rate Limit errors (temporary, should retry with backoff)
    - 5xx server errors (temporary)
    - Timeouts and connection issues (transient)
    
    Returns False for:
    - 4xx client errors EXCEPT 429 (permanent errors like bad URL, invalid request)
    """
    import re
    exc_str = str(exception)
    
    # Check for 429 Rate Limit - this MUST be retried (it's temporary)
    if re.search(r'(?:response|status|error)[:\s]+429', exc_str, re.IGNORECASE) or '429' in exc_str:
        print(f"[Pixtral] Rate limit (429), will retry with backoff...")
        return True
    
    # Check for 5xx errors - these should ALWAYS be retried
    # Match patterns like "Error response 502", "status 503", "500 Internal"
    if re.search(r'(?:response|status|error)[:\s]+5\d{2}', exc_str, re.IGNORECASE):
        print(f"[Pixtral] Server error (5xx), will retry: {exc_str[:100]}...")
        return True
    
    # Check for common httpx/requests exception attributes
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        status = exception.response.status_code
        if status == 429:
            print(f"[Pixtral] Rate limit (429), will retry with backoff...")
            return True
        if status >= 500:
            print(f"[Pixtral] Server error ({status}), will retry")
            return True
        if 400 <= status < 500:
            print(f"[Pixtral] Client error ({status}), skipping retry: {exc_str[:150]}")
            return False
    
    # Check for Mistral-specific error attributes
    if hasattr(exception, 'status_code'):
        status = exception.status_code
        if status == 429:
            print(f"[Pixtral] Rate limit (429), will retry with backoff...")
            return True
        if status >= 500:
            print(f"[Pixtral] Server error ({status}), will retry")
            return True
        if 400 <= status < 500:
            print(f"[Pixtral] Client error ({status}), skipping retry: {exc_str[:150]}")
            return False
    
    # Check for HTTP 4xx status codes in exception message (more precise matching)
    # Match patterns like "Error response 400", "status 403", "400 Bad Request"
    # But NOT 429 which we already handled above
    if re.search(r'(?:response|status|error)[:\s]+4(?!29)\d{2}', exc_str, re.IGNORECASE):
        print(f"[Pixtral] Client error (4xx), skipping retry: {exc_str[:150]}")
        return False
    
    # Retry for all other errors (timeouts, connection issues, etc.)
    print(f"[Pixtral] Transient error, will retry: {exc_str[:100]}...")
    return True


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[Pixtral] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatPixtral(ChatMistralAI):
    """ChatMistralAI for Pixtral vision model with exponential backoff retry (2s-20s, max 3 attempts)."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),  # Reduced from 5 to fail faster on persistent issues
            wait=wait_exponential(multiplier=8, min=8, max=500),  # Reduced max wait from 60s to 20s
            retry=retry_if_exception(_is_retryable_error),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            return super(ChatPixtral, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),  # Reduced from 5 to fail faster on persistent issues
            wait=wait_exponential(multiplier=8, min=8, max=500),  # Reduced max wait from 60s to 20s
            retry=retry_if_exception(_is_retryable_error),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            return await super(ChatPixtral, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_pixtral_chat_model(
    model_name: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 5,
    **kwargs,
) -> ChatMistralAI:
    """
    Build a ChatMistralAI client configured for Pixtral vision model.
    
    Pixtral is Mistral's vision model that can analyze images.
    Uses PIXTRAL_API_KEY or falls back to MISTRAL_API_KEY.
    
    Args:
        model_name: Model name override. Defaults to PIXTRAL_MODEL_NAME env var
                   or 'pixtral-large-latest'.
        temperature: Sampling temperature (default 0.1 for consistent scoring).
        max_retries: Number of retries (handled by custom exponential backoff).
        **kwargs: Additional arguments passed to ChatMistralAI.
    
    Returns:
        ChatPixtral instance configured for vision tasks.
    """
    model = model_name or os.getenv("PIXTRAL_MODEL_NAME", "pixtral-large-latest")
    
    # Try PIXTRAL_API_KEY first, then fall back to MISTRAL_API_KEY
    api_key = (
        kwargs.pop("pixtral_api_key", None)
        or kwargs.pop("mistral_api_key", None)
        or kwargs.pop("api_key", None)
        or os.getenv("PIXTRAL_API_KEY")
        or os.getenv("MISTRAL_API_KEY")
    )
    
    if not api_key:
        raise ValueError(
            "PIXTRAL_API_KEY or MISTRAL_API_KEY environment variable must be set"
        )

    return ChatPixtral(
        model=model,
        temperature=temperature,
        max_retries=0,  # Disable internal retries, we use custom exponential backoff
        mistral_api_key=api_key,
        timeout=60,  # 30-second request timeout to fail fast on hanging requests
        **kwargs,
    )

