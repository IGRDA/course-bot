"""Coqui TTS engine sub-package.

Imports are lazy so that pydub/torch/TTS are only loaded when used.
"""


def __getattr__(name: str):
    if name in ("CoquiTTSEngine", "TTSEngine", "generate_podcast", "list_available_languages", "list_speakers"):
        from .client import (
            CoquiTTSEngine,
            TTSEngine,
            generate_podcast,
            list_available_languages,
            list_speakers,
        )
        _map = {
            "CoquiTTSEngine": CoquiTTSEngine,
            "TTSEngine": TTSEngine,
            "generate_podcast": generate_podcast,
            "list_available_languages": list_available_languages,
            "list_speakers": list_speakers,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CoquiTTSEngine",
    "TTSEngine",
    "generate_podcast",
    "list_available_languages",
    "list_speakers",
]
