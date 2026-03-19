"""ElevenLabs TTS engine sub-package.

Imports are lazy so that the elevenlabs SDK is only loaded when used.
"""


def __getattr__(name: str):
    if name in ("ELEVENLABS_VOICE_MAP", "ElevenLabsTTSEngine", "generate_podcast_elevenlabs"):
        from .client import ELEVENLABS_VOICE_MAP, ElevenLabsTTSEngine, generate_podcast_elevenlabs
        _map = {
            "ELEVENLABS_VOICE_MAP": ELEVENLABS_VOICE_MAP,
            "ElevenLabsTTSEngine": ElevenLabsTTSEngine,
            "generate_podcast_elevenlabs": generate_podcast_elevenlabs,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ELEVENLABS_VOICE_MAP",
    "ElevenLabsTTSEngine",
    "generate_podcast_elevenlabs",
]
