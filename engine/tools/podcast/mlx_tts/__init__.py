"""MLX-Audio TTS engine sub-package (Apple Silicon optimized).

Imports are lazy so that mlx/mlx_audio are only loaded when used.
"""


def __getattr__(name: str):
    if name in ("MLXTTSEngine", "generate_podcast_mlx_tts"):
        from .client import MLXTTSEngine, generate_podcast_mlx_tts
        _map = {
            "MLXTTSEngine": MLXTTSEngine,
            "generate_podcast_mlx_tts": generate_podcast_mlx_tts,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MLXTTSEngine",
    "generate_podcast_mlx_tts",
]
