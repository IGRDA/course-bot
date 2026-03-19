"""Chatterbox TTS engine sub-package.

Imports are lazy so that pydub/torch/chatterbox are only loaded when used.
"""


def __getattr__(name: str):
    if name in ("ChatterboxEngine", "generate_podcast_chatterbox"):
        from .client import ChatterboxEngine, generate_podcast_chatterbox
        _map = {
            "ChatterboxEngine": ChatterboxEngine,
            "generate_podcast_chatterbox": generate_podcast_chatterbox,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatterboxEngine",
    "generate_podcast_chatterbox",
]
