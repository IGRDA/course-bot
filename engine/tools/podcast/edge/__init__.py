"""Edge TTS engine sub-package.

Imports are lazy so that pydub/edge-tts are only loaded when used.
"""


def __getattr__(name: str):
    if name in ("EDGE_VOICE_MAP", "EDGE_VOICES", "EdgeTTSEngine", "generate_podcast_edge"):
        from .client import EDGE_VOICE_MAP, EDGE_VOICES, EdgeTTSEngine, generate_podcast_edge
        _map = {
            "EDGE_VOICE_MAP": EDGE_VOICE_MAP,
            "EDGE_VOICES": EDGE_VOICES,
            "EdgeTTSEngine": EdgeTTSEngine,
            "generate_podcast_edge": generate_podcast_edge,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EDGE_VOICE_MAP",
    "EDGE_VOICES",
    "EdgeTTSEngine",
    "generate_podcast_edge",
]
