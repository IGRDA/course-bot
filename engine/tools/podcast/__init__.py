"""
Podcast TTS Tool

Generate multi-speaker podcast audio from structured conversations.
Supports Edge TTS, Coqui TTS (offline), and Chatterbox TTS.

All engine-specific imports are lazy so that heavy optional dependencies
(pydub, torch, coqui-tts, edge-tts, chatterbox, mutagen) are only
loaded when the caller actually uses a specific engine or utility.
"""

# Lightweight imports that have no heavy deps
from .models import (
    Message,
    Conversation,
    LanguageConfig,
    LANGUAGE_CONFIGS,
    get_language_config,
)
from .base_engine import BaseTTSEngine
from .factory import create_tts_engine, get_engine_info, list_engines, EngineType


def __getattr__(name: str):
    """Lazy module-level attribute access for heavy engine exports."""
    # Edge TTS exports
    if name in ("EdgeTTSEngine", "EDGE_VOICE_MAP", "EDGE_VOICES", "generate_podcast_edge"):
        from .edge.client import EdgeTTSEngine, EDGE_VOICE_MAP, EDGE_VOICES, generate_podcast_edge
        _map = {
            "EdgeTTSEngine": EdgeTTSEngine,
            "EDGE_VOICE_MAP": EDGE_VOICE_MAP,
            "EDGE_VOICES": EDGE_VOICES,
            "generate_podcast_edge": generate_podcast_edge,
        }
        return _map[name]

    # Coqui TTS exports
    if name in ("TTSEngine", "CoquiTTSEngine", "generate_podcast", "list_available_languages", "list_speakers"):
        from .coqui.client import (
            CoquiTTSEngine,
            TTSEngine,
            generate_podcast,
            list_available_languages,
            list_speakers,
        )
        _map = {
            "TTSEngine": TTSEngine,
            "CoquiTTSEngine": CoquiTTSEngine,
            "generate_podcast": generate_podcast,
            "list_available_languages": list_available_languages,
            "list_speakers": list_speakers,
        }
        return _map[name]

    # Chatterbox TTS exports
    if name in ("ChatterboxEngine", "generate_podcast_chatterbox"):
        from .chatterbox.client import ChatterboxEngine, generate_podcast_chatterbox
        _map = {
            "ChatterboxEngine": ChatterboxEngine,
            "generate_podcast_chatterbox": generate_podcast_chatterbox,
        }
        return _map[name]

    # ElevenLabs TTS exports
    if name in ("ElevenLabsTTSEngine", "generate_podcast_elevenlabs", "ELEVENLABS_VOICE_MAP"):
        from .elevenlabs.client import ELEVENLABS_VOICE_MAP, ElevenLabsTTSEngine, generate_podcast_elevenlabs
        _map = {
            "ElevenLabsTTSEngine": ElevenLabsTTSEngine,
            "generate_podcast_elevenlabs": generate_podcast_elevenlabs,
            "ELEVENLABS_VOICE_MAP": ELEVENLABS_VOICE_MAP,
        }
        return _map[name]

    # OpenAI TTS exports
    if name in ("OpenAITTSEngine", "generate_podcast_openai_tts", "OPENAI_TTS_VOICE_MAP"):
        from .openai_tts.client import OPENAI_TTS_VOICE_MAP, OpenAITTSEngine, generate_podcast_openai_tts
        _map = {
            "OpenAITTSEngine": OpenAITTSEngine,
            "generate_podcast_openai_tts": generate_podcast_openai_tts,
            "OPENAI_TTS_VOICE_MAP": OPENAI_TTS_VOICE_MAP,
        }
        return _map[name]

    # Qwen TTS exports
    if name in ("QwenTTSEngine", "generate_podcast_qwen_tts"):
        from .qwen_tts.client import QwenTTSEngine, generate_podcast_qwen_tts
        _map = {
            "QwenTTSEngine": QwenTTSEngine,
            "generate_podcast_qwen_tts": generate_podcast_qwen_tts,
        }
        return _map[name]

    # Qwen TTS API exports
    if name in ("QwenTTSApiEngine", "generate_podcast_qwen_tts_api", "QWEN_TTS_API_VOICE_MAP"):
        from .qwen_tts_api.client import QWEN_TTS_API_VOICE_MAP, QwenTTSApiEngine, generate_podcast_qwen_tts_api
        _map = {
            "QwenTTSApiEngine": QwenTTSApiEngine,
            "generate_podcast_qwen_tts_api": generate_podcast_qwen_tts_api,
            "QWEN_TTS_API_VOICE_MAP": QWEN_TTS_API_VOICE_MAP,
        }
        return _map[name]

    # MLX TTS exports
    if name in ("MLXTTSEngine", "generate_podcast_mlx_tts"):
        from .mlx_tts.client import MLXTTSEngine, generate_podcast_mlx_tts
        _map = {
            "MLXTTSEngine": MLXTTSEngine,
            "generate_podcast_mlx_tts": generate_podcast_mlx_tts,
        }
        return _map[name]

    # Audio utility exports
    if name in ("add_metadata", "add_background_music", "get_default_music_path"):
        from .audio_utils import add_metadata, add_background_music, get_default_music_path
        _map = {
            "add_metadata": add_metadata,
            "add_background_music": add_background_music,
            "get_default_music_path": get_default_music_path,
        }
        return _map[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Models
    "Message",
    "Conversation",
    "LanguageConfig",
    "LANGUAGE_CONFIGS",
    "get_language_config",
    # Base Engine
    "BaseTTSEngine",
    # Edge TTS Engine
    "EdgeTTSEngine",
    "EDGE_VOICE_MAP",
    "EDGE_VOICES",
    "generate_podcast_edge",
    # Coqui TTS Engine
    "TTSEngine",
    "CoquiTTSEngine",
    "generate_podcast",
    "list_available_languages",
    "list_speakers",
    # Chatterbox TTS Engine
    "ChatterboxEngine",
    "generate_podcast_chatterbox",
    # ElevenLabs TTS Engine
    "ElevenLabsTTSEngine",
    "generate_podcast_elevenlabs",
    "ELEVENLABS_VOICE_MAP",
    # OpenAI TTS Engine
    "OpenAITTSEngine",
    "generate_podcast_openai_tts",
    "OPENAI_TTS_VOICE_MAP",
    # Qwen TTS Engine
    "QwenTTSEngine",
    "generate_podcast_qwen_tts",
    # Qwen TTS API Engine
    "QwenTTSApiEngine",
    "generate_podcast_qwen_tts_api",
    "QWEN_TTS_API_VOICE_MAP",
    # MLX TTS Engine
    "MLXTTSEngine",
    "generate_podcast_mlx_tts",
    # Factory
    "create_tts_engine",
    "get_engine_info",
    "list_engines",
    "EngineType",
    # Audio Utilities
    "add_metadata",
    "add_background_music",
    "get_default_music_path",
]
