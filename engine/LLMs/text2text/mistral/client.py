import os

from langchain_mistralai import ChatMistralAI
from LLMs.api_keys import get_random_key, parse_api_keys
from pydantic import PrivateAttr, SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(
        f"[Mistral] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}"
    )


class ChatMistral(ChatMistralAI):
    """ChatMistralAI with exponential backoff retry and multi-key rotation."""

    _api_keys: list[str] = PrivateAttr(default_factory=list)

    def _rotate_api_key(self):
        if self._api_keys:
            key = get_random_key(self._api_keys)
            self.mistral_api_key = SecretStr(key)

    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=10, min=10, max=500),
            reraise=True,
            before_sleep=_log_retry,
        )
        def _call():
            self._rotate_api_key()
            return super(ChatMistral, self)._generate(*args, **kwargs)

        return _call()

    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=10, min=10, max=500),
            reraise=True,
            before_sleep=_log_retry,
        )
        async def _acall():
            self._rotate_api_key()
            return await super(ChatMistral, self)._agenerate(*args, **kwargs)

        return await _acall()


def build_mistral_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 5,
    **kwargs,
) -> ChatMistralAI:
    """Build a ChatMistralAI client with custom exponential backoff."""
    model = model_name or os.getenv("MISTRAL_MODEL_NAME")
    if not model:
        raise ValueError("MISTRAL_MODEL_NAME environment variable must be set")
    raw_key = kwargs.pop("mistral_api_key", None) or kwargs.pop("api_key", None) or os.getenv("MISTRAL_API_KEY")
    api_keys = parse_api_keys(raw_key)

    client = ChatMistral(
        model=model,
        temperature=temperature,
        max_retries=0,
        mistral_api_key=api_keys[0] if api_keys else None,
        timeout=360,
        **kwargs,
    )
    client._api_keys = api_keys
    if len(api_keys) > 1:
        print(f"[Mistral] Loaded {len(api_keys)} API keys for rotation")
    return client
