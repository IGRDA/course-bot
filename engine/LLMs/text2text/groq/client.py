import os

from langchain_groq import ChatGroq
from LLMs.api_keys import get_random_key, parse_api_keys
from pydantic import PrivateAttr, SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[Groq] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatGroqWithRetry(ChatGroq):
    """ChatGroq with exponential backoff retry and multi-key rotation."""

    _api_keys: list[str] = PrivateAttr(default_factory=list)

    def _rotate_api_key(self):
        if self._api_keys:
            key = get_random_key(self._api_keys)
            self.groq_api_key = SecretStr(key)

    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry,
        )
        def _call():
            self._rotate_api_key()
            return super(ChatGroqWithRetry, self)._generate(*args, **kwargs)

        return _call()

    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry,
        )
        async def _acall():
            self._rotate_api_key()
            return await super(ChatGroqWithRetry, self)._agenerate(*args, **kwargs)

        return await _acall()


def build_groq_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatGroq:
    """
    Build a ``ChatGroqWithRetry`` client that requires the model name to be provided
    either as a parameter or via the GROQ_MODEL_NAME environment variable.
    """
    model = model_name or os.getenv("GROQ_MODEL_NAME")
    if not model:
        raise ValueError("GROQ_MODEL_NAME environment variable must be set")
    raw_key = kwargs.pop("api_key", None) or os.getenv("GROQ_API_KEY")
    api_keys = parse_api_keys(raw_key)

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_keys:
        client_kwargs["api_key"] = api_keys[0]

    client = ChatGroqWithRetry(**client_kwargs)
    client._api_keys = api_keys
    if len(api_keys) > 1:
        print(f"[Groq] Loaded {len(api_keys)} API keys for rotation")
    return client
