"""
TTS Engine Factory.

Provides a unified interface for creating TTS engines by name.
"""

from typing import Optional, Literal

from .base_engine import BaseTTSEngine


# Available engine types
EngineType = Literal[
"edge", "coqui", "chatterbox", "elevenlabs", "openai_tts", "qwen_tts", "mlx_tts", "qwen_tts_api"]


def create_tts_engine(
    engine: EngineType = "qwen_tts_api",
    language: str = "en",
    speaker_map: Optional[dict[str, str]] = None,
    **kwargs,
) -> BaseTTSEngine:
    """Create a TTS engine by name.
    
    Args:
        engine: Engine type ("edge" or "coqui")
        language: Language code (e.g., "en", "es")
        speaker_map: Optional mapping of role names to speaker/voice IDs
        **kwargs: Additional engine-specific arguments
            - For Coqui: device (str) - "cpu" or "cuda"
    
    Returns:
        TTS engine instance
        
    Raises:
        ValueError: If engine type is unknown
        
    Example:
        >>> engine = create_tts_engine("edge", language="es")
        >>> engine.supports_ssml
        True
        
        >>> engine = create_tts_engine("coqui", language="en", device="cuda")
        >>> engine.supports_ssml
        False
    """
    if engine == "edge":
        from .edge.client import EdgeTTSEngine
        return EdgeTTSEngine(
            language=language,
            speaker_map=speaker_map,
        )
    elif engine == "coqui":
        from .coqui.client import CoquiTTSEngine
        return CoquiTTSEngine(
            language=language,
            speaker_map=speaker_map,
            device=kwargs.get("device", "cpu"),
        )
    elif engine == "chatterbox":
        from .chatterbox.client import ChatterboxEngine
        return ChatterboxEngine(
            language=language,
            speaker_map=speaker_map,
            device=kwargs.get("device", "cuda"),
            exaggeration=kwargs.get("exaggeration", 0.5),
            cfg_weight=kwargs.get("cfg_weight", 0.5),
        )
    elif engine == "elevenlabs":
        from .elevenlabs.client import ElevenLabsTTSEngine
        return ElevenLabsTTSEngine(
            language=language,
            speaker_map=speaker_map,
            model_id=kwargs.get("model_id", "eleven_multilingual_v2"),
            api_key=kwargs.get("api_key"),
        )
    elif engine == "openai_tts":
        from .openai_tts.client import OpenAITTSEngine
        return OpenAITTSEngine(
            language=language,
            speaker_map=speaker_map,
            model=kwargs.get("model", "gpt-4o-mini-tts"),
            api_key=kwargs.get("api_key"),
            instructions=kwargs.get("instructions"),
        )
    elif engine == "qwen_tts":
        from .qwen_tts.client import QwenTTSEngine
        return QwenTTSEngine(
            language=language,
            speaker_map=speaker_map,
            task_type=kwargs.get("task_type", "custom_voice"),
            device=kwargs.get("device"),
            instruct=kwargs.get("instruct"),
            model_name=kwargs.get("model_name"),
        )
    elif engine == "mlx_tts":
        from .mlx_tts.client import MLXTTSEngine
        return MLXTTSEngine(
            language=language,
            speaker_map=speaker_map,
            task_type=kwargs.get("task_type", "voice_clone"),
            instruct=kwargs.get("instruct"),
            model_name=kwargs.get("model_name"),
            temperature=kwargs.get("temperature", 0.7),
            speed=kwargs.get("speed", 1.0),
        )
    elif engine == "qwen_tts_api":
        from .qwen_tts_api.client import QwenTTSApiEngine
        return QwenTTSApiEngine(
            language=language,
            speaker_map=speaker_map,
            api_key=kwargs.get("api_key"),
            api_url=kwargs.get("api_url"),
            concurrency=kwargs.get("concurrency", 10),
        )
    else:
        available = [
            "edge", "coqui", "chatterbox", "elevenlabs", "openai_tts",
            "qwen_tts", "mlx_tts", "qwen_tts_api",
        ]
        raise ValueError(f"Unknown engine '{engine}'. Available: {available}")


def get_engine_info(engine: EngineType) -> dict:
    """Get information about a TTS engine.
    
    Args:
        engine: Engine type
        
    Returns:
        Dict with engine capabilities and info
    """
    info = {
        "edge": {
            "name": "Edge TTS",
            "description": "Microsoft Edge neural TTS - fast, high quality",
            "requires_internet": True,
            "languages": ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        },
        "coqui": {
            "name": "Coqui TTS",
            "description": "Open-source TTS with multi-speaker support - works offline",
            "requires_internet": False,
            "languages": ["en", "es", "multilingual"],
        },
        "chatterbox": {
            "name": "Chatterbox TTS",
            "description": "Resemble AI's zero-shot TTS with voice cloning - 23 languages",
            "requires_internet": False,
            "languages": [
                "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                "sw", "tr", "zh",
            ],
        },
        "elevenlabs": {
            "name": "ElevenLabs TTS",
            "description": "ElevenLabs cloud API - ultra-realistic voices, 29+ languages",
            "requires_internet": True,
            "languages": [
                "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                "tr", "zh",
            ],
        },
        "openai_tts": {
            "name": "OpenAI TTS",
            "description": "OpenAI gpt-4o-mini-tts - controllable accent/tone via instructions, 50+ languages",
            "requires_internet": True,
            "languages": [
                "af", "ar", "be", "bg", "bs", "ca", "cs", "da", "de", "el",
                "en", "es", "et", "fi", "fr", "gl", "he", "hi", "hr", "hu",
                "id", "is", "it", "ja", "kn", "ko", "lv", "lt", "mk", "ms",
                "mr", "ne", "nl", "no", "fa", "pl", "pt", "ro", "ru", "sk",
                "sl", "sr", "sv", "sw", "ta", "th", "tr", "uk", "ur", "vi",
                "zh",
            ],
        },
        "qwen_tts": {
            "name": "Qwen3-TTS",
            "description": "Alibaba Qwen3-TTS - local GPU, 9 built-in speakers + voice cloning, 10 languages",
            "requires_internet": False,
            "languages": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        },
        "mlx_tts": {
            "name": "MLX Qwen3-TTS",
            "description": "Qwen3-TTS via MLX-Audio - Apple Silicon optimized, offline, voice cloning, 10 languages",
            "requires_internet": False,
            "languages": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        },
        "qwen_tts_api": {
            "name": "Qwen TTS API",
            "description": "Remote Qwen TTS gateway API - voice cloning, parallel synthesis, requires GCLOUD_GATEWAY_API_KEY",
            "requires_internet": True,
            "languages": ["es", "en", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        },
    }
    
    if engine not in info:
        raise ValueError(f"Unknown engine '{engine}'")
    
    return info[engine]


def list_engines() -> list[dict]:
    """List all available TTS engines with their info.
    
    Returns:
        List of engine info dictionaries
    """
    return [
        {"engine": "edge", **get_engine_info("edge")},
        {"engine": "coqui", **get_engine_info("coqui")},
        {"engine": "chatterbox", **get_engine_info("chatterbox")},
        {"engine": "elevenlabs", **get_engine_info("elevenlabs")},
        {"engine": "openai_tts", **get_engine_info("openai_tts")},
        {"engine": "qwen_tts", **get_engine_info("qwen_tts")},
        {"engine": "mlx_tts", **get_engine_info("mlx_tts")},
        {"engine": "qwen_tts_api", **get_engine_info("qwen_tts_api")},
    ]

