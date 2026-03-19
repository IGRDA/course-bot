"""OpenAI TTS engine sub-package.

Imports are lazy so that the openai SDK is only loaded when used.
"""


def __getattr__(name: str):
    if name in ("OPENAI_TTS_VOICE_MAP", "OpenAITTSEngine", "generate_podcast_openai_tts"):
        from .client import OPENAI_TTS_VOICE_MAP, OpenAITTSEngine, generate_podcast_openai_tts
        _map = {
            "OPENAI_TTS_VOICE_MAP": OPENAI_TTS_VOICE_MAP,
            "OpenAITTSEngine": OpenAITTSEngine,
            "generate_podcast_openai_tts": generate_podcast_openai_tts,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OPENAI_TTS_VOICE_MAP",
    "OpenAITTSEngine",
    "generate_podcast_openai_tts",
]
