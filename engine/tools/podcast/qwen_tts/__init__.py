"""Qwen3-TTS engine sub-package.

Imports are lazy so that qwen_tts/torch are only loaded when used.
"""


def __getattr__(name: str):
    if name in ("QwenTTSEngine", "generate_podcast_qwen_tts"):
        from .client import QwenTTSEngine, generate_podcast_qwen_tts

        _map = {
            "QwenTTSEngine": QwenTTSEngine,
            "generate_podcast_qwen_tts": generate_podcast_qwen_tts,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QwenTTSEngine",
    "generate_podcast_qwen_tts",
]
