"""Qwen TTS API engine sub-package.

Imports are lazy so that requests/pydub are only loaded when used.
"""


def __getattr__(name: str):
    if name in ("QWEN_TTS_API_VOICE_MAP", "QwenTTSApiEngine", "generate_podcast_qwen_tts_api"):
        from .client import QWEN_TTS_API_VOICE_MAP, QwenTTSApiEngine, generate_podcast_qwen_tts_api
        _map = {
            "QWEN_TTS_API_VOICE_MAP": QWEN_TTS_API_VOICE_MAP,
            "QwenTTSApiEngine": QwenTTSApiEngine,
            "generate_podcast_qwen_tts_api": generate_podcast_qwen_tts_api,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QWEN_TTS_API_VOICE_MAP",
    "QwenTTSApiEngine",
    "generate_podcast_qwen_tts_api",
]
