import os

from langchain_deepseek import ChatDeepSeek
from pydantic import PrivateAttr, SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential

from LLMs.api_keys import get_random_key, mask_key, parse_api_keys


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[DeepSeek] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatDeepSeekWithRetry(ChatDeepSeek):
    """ChatDeepSeek with exponential backoff retry and multi-key rotation."""

    _api_keys: list[str] = PrivateAttr(default_factory=list)

    def _rotate_api_key(self):
        if self._api_keys:
            key = get_random_key(self._api_keys)
            self.api_key = SecretStr(key)

    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=10, min=10, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            self._rotate_api_key()
            return super(ChatDeepSeekWithRetry, self)._generate(*args, **kwargs)
        return _call()

    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=10, min=10, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            self._rotate_api_key()
            return await super(ChatDeepSeekWithRetry, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_deepseek_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatDeepSeek:
    """
    Build a ChatDeepSeek client using the official langchain-deepseek package.
    Supports structured output with deepseek-chat (DeepSeek-V3).
    """
    model = model_name or os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    if not model:
        raise ValueError("DEEPSEEK_MODEL_NAME environment variable must be set")

    raw_key = kwargs.pop("api_key", None) or os.getenv("DEEPSEEK_API_KEY")
    api_keys = parse_api_keys(raw_key)

    client = ChatDeepSeekWithRetry(
        model=model,
        temperature=temperature,
        api_key=api_keys[0] if api_keys else None,
        max_retries=0,
        **kwargs,
    )
    client._api_keys = api_keys
    if len(api_keys) > 1:
        print(f"[DeepSeek] Loaded {len(api_keys)} API keys for rotation")
    return client
