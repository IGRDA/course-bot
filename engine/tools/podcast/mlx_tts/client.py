"""
MLX-Audio TTS Engine for podcast generation on Apple Silicon.

Uses the mlx-audio library with Qwen3-TTS models optimized for Apple's
MLX framework. Provides near-real-time inference on M1-M4 chips using
unified memory, without requiring CUDA.

Requires ``pip install -U mlx-audio``.

https://github.com/Blaizzy/mlx-audio
"""

import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from ..base_engine import BaseTTSEngine
from ..models import Conversation, Message

TaskType = Literal["custom_voice", "voice_clone"]

# Qwen3-TTS supported languages (code -> full name expected by the library)
MLX_LANGUAGE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# Default speaker assignments per language for CustomVoice mode.
# All 9 built-in speakers can speak all 10 languages, but native language
# gives the best quality.
MLX_VOICE_MAP: dict[str, dict[str, str]] = {
    "en": {"host": "Ryan", "guest": "Aiden"},
    "es": {"host": "Ryan", "guest": "Aiden"},
    "fr": {"host": "Ryan", "guest": "Aiden"},
    "de": {"host": "Ryan", "guest": "Aiden"},
    "it": {"host": "Ryan", "guest": "Aiden"},
    "pt": {"host": "Ryan", "guest": "Aiden"},
    "ru": {"host": "Ryan", "guest": "Aiden"},
    "zh": {"host": "Vivian", "guest": "Serena"},
    "ja": {"host": "Ono_Anna", "guest": "Vivian"},
    "ko": {"host": "Sohee", "guest": "Vivian"},
}

MLX_ALL_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

# HuggingFace MLX-optimized model IDs
MLX_MODEL_CUSTOM_VOICE = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
MLX_MODEL_BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

# Default reference audio for voice cloning (relative to project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MLX_DEFAULT_CLONE_MAP: dict[str, dict[str, str]] = {
    "es": {
        "host": str(_PROJECT_ROOT / "output" / "spanish_male_voice_v2.wav"),
        "guest": str(_PROJECT_ROOT / "output" / "spanish_female_voice_v2.wav"),
    },
}


class MLXTTSEngine(BaseTTSEngine):
    """Text-to-Speech engine using MLX-Audio with Qwen3-TTS on Apple Silicon.

    Supports two modes via ``task_type``:
      - ``"voice_clone"`` (default): Clone any voice from a short reference
        audio clip. The ``speaker_map`` should map roles to audio file paths.
        Uses the 1.7B Base model. When no speaker_map is provided and default
        reference voices exist for the language, they are used automatically.
      - ``"custom_voice"``: 9 built-in speakers with optional instruction-based
        tone/emotion control. Uses the 1.7B CustomVoice model (~3.5 GB).

    Optimized for Apple Silicon (M1-M4) via MLX unified memory.
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: dict[str, str] | None = None,
        task_type: TaskType = "voice_clone",
        instruct: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.7,
        speed: float = 1.0,
    ):
        """Initialize the MLX-Audio TTS engine.

        Args:
            language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ru)
            speaker_map: For custom_voice: role -> speaker name.
                         For voice_clone: role -> path to reference audio file.
                         If None and voice_clone, uses default reference voices
                         when available for the language.
            task_type: "voice_clone" (default) for cloning from reference audio
                       or "custom_voice" for built-in speakers.
            instruct: Optional instruction for tone/emotion control
                      (custom_voice mode only, e.g. "Speak warmly").
            model_name: Override the HuggingFace model ID.
            temperature: Sampling temperature for generation (default 0.7).
            speed: Playback speed multiplier (default 1.0).
        """
        super().__init__(language=language, speaker_map=speaker_map)

        if language not in MLX_LANGUAGE_MAP:
            available = ", ".join(sorted(MLX_LANGUAGE_MAP.keys()))
            raise ValueError(f"Unsupported language '{language}'. Available: {available}")

        self.task_type = task_type
        self.instruct = instruct
        self.language_full = MLX_LANGUAGE_MAP[language]
        self.temperature = temperature
        self.speed = speed

        if model_name:
            self.model_name = model_name
        elif task_type == "voice_clone":
            self.model_name = MLX_MODEL_BASE
        else:
            self.model_name = MLX_MODEL_CUSTOM_VOICE

        if not self.speaker_map:
            if task_type == "voice_clone":
                # Use default reference voices if available for this language
                default_clone = MLX_DEFAULT_CLONE_MAP.get(language)
                if default_clone and all(Path(p).exists() for p in default_clone.values()):
                    self.speaker_map = default_clone.copy()
                else:
                    # Fall back to custom_voice built-in speakers
                    self.task_type = "custom_voice"
                    self.model_name = model_name or MLX_MODEL_CUSTOM_VOICE
                    self.speaker_map = MLX_VOICE_MAP.get(language, MLX_VOICE_MAP["en"]).copy()
            else:
                self.speaker_map = MLX_VOICE_MAP.get(language, MLX_VOICE_MAP["en"]).copy()

        self._model = None
        self._ref_audio_cache: dict[str, object] = {}

    @property
    def model(self):
        """Lazy-load the MLX model on first access."""
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            print(f"Loading MLX model {self.model_name} …")
            self._model = load_model(self.model_name)
            print("MLX model loaded successfully.")
        return self._model

    def _load_ref_audio(self, ref_audio_path: str):
        """Load and cache a reference audio file as an MLX array.

        Args:
            ref_audio_path: Path to the reference audio file.

        Returns:
            Loaded audio data reusable across multiple generate calls.
        """
        if ref_audio_path not in self._ref_audio_cache:
            from mlx_audio.utils import load_audio

            self._ref_audio_cache[ref_audio_path] = load_audio(ref_audio_path, sample_rate=self.model.sample_rate)
        return self._ref_audio_cache[ref_audio_path]

    def get_speaker_for_role(self, role: str) -> str:
        """Get the speaker name or reference audio path for a given role.

        Args:
            role: Speaker role name (e.g. "host", "guest")

        Returns:
            Speaker name (custom_voice) or audio path (voice_clone).
        """
        if role in self.speaker_map:
            return self.speaker_map[role]

        if self.task_type == "custom_voice":
            fallback = MLX_VOICE_MAP.get(self.language, MLX_VOICE_MAP["en"])
            speaker = fallback.get("host", "Ryan")
            self.speaker_map[role] = speaker
            return speaker

        return role

    def _generate_to_file(
        self,
        text: str,
        output_path: str,
        role: str = "host",
        language_code: str | None = None,
    ) -> str:
        """Run model.generate() and write the result to a WAV file."""
        from mlx_audio.audio_io import write as audio_write

        lang = language_code or self.language_full
        if lang in MLX_LANGUAGE_MAP:
            lang = MLX_LANGUAGE_MAP[lang]

        gen_kwargs: dict = {
            "text": text,
            "lang_code": lang,
            "speed": self.speed,
            "temperature": self.temperature,
            "verbose": False,
        }

        if self.task_type == "voice_clone":
            ref_audio_path = self.get_speaker_for_role(role)
            if not Path(ref_audio_path).exists():
                raise FileNotFoundError(f"Reference audio not found for role '{role}': {ref_audio_path}")
            gen_kwargs["ref_audio"] = self._load_ref_audio(ref_audio_path)
            gen_kwargs["ref_text"] = ""
        else:
            gen_kwargs["voice"] = self.get_speaker_for_role(role)
            if self.instruct:
                gen_kwargs["instruct"] = self.instruct

        results = self.model.generate(**gen_kwargs)

        audio_chunks = []
        sample_rate = self.model.sample_rate
        for result in results:
            audio_chunks.append(np.array(result.audio))
            sample_rate = result.sample_rate

        audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        audio_write(output_path, audio, sample_rate, format="wav")
        return output_path

    def synthesize_message(
        self,
        message: Message,
        output_path: str,
        language_code: str | None = None,
    ) -> str:
        """Synthesize a single message to a WAV file.

        Args:
            message: Message to synthesize.
            output_path: Path to save the audio file.
            language_code: Optional language code override.

        Returns:
            Path to the generated audio file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return self._generate_to_file(
            text=message.content,
            output_path=output_path,
            role=message.role,
            language_code=language_code,
        )

    def synthesize_conversation(
        self,
        conversation: Conversation,
        output_path: str,
        language_code: str | None = None,
        silence_duration_ms: int = 500,
        progress_callback: callable | None = None,
    ) -> str:
        """Synthesize a full conversation to a single audio file.

        Args:
            conversation: Conversation to synthesize.
            output_path: Path to save the final audio file.
            language_code: Optional language code override.
            silence_duration_ms: Silence between messages in ms.
            progress_callback: Optional callback(current, total).

        Returns:
            Path to the generated audio file.
        """
        if len(conversation) == 0:
            raise ValueError("Conversation cannot be empty")

        from pydub import AudioSegment

        with tempfile.TemporaryDirectory(prefix="mlx_tts_") as temp_dir:
            audio_segments: list[AudioSegment] = []
            silence = AudioSegment.silent(duration=silence_duration_ms)
            self.segment_durations_ms = []

            total = len(conversation)
            for idx, message in enumerate(conversation):
                temp_path = os.path.join(temp_dir, f"segment_{idx:04d}.wav")
                self.synthesize_message(
                    message=message,
                    output_path=temp_path,
                    language_code=language_code,
                )
                segment = AudioSegment.from_wav(temp_path)
                audio_segments.append(segment)
                self.segment_durations_ms.append(len(segment))
                if progress_callback:
                    progress_callback(idx + 1, total)

            combined = audio_segments[0]
            for segment in audio_segments[1:]:
                combined = combined + silence + segment

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            output_format = out.suffix.lstrip(".").lower()
            if output_format not in ("mp3", "wav", "ogg", "flac"):
                output_format = "wav"
            combined.export(str(out), format=output_format)

        return str(output_path)

    @classmethod
    def list_available_voices(cls, language: str = "en") -> list[str]:
        """List available built-in speakers (CustomVoice mode).

        Args:
            language: Language code (unused -- all speakers support all langs).

        Returns:
            List of speaker name strings.
        """
        return list(MLX_ALL_SPEAKERS)


def generate_podcast_mlx_tts(
    conversation: list[dict],
    output_path: str,
    language: str = "en",
    speaker_map: dict[str, str] | None = None,
    silence_duration_ms: int = 500,
    progress_callback: callable | None = None,
    task_type: TaskType = "voice_clone",
    instruct: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.7,
    speed: float = 1.0,
    # Metadata options
    title: str = "Module",
    artist: str = "Adinhub",
    album: str = "Course",
    track_number: int | None = None,
    # Background music options
    music_path: str | None = None,
    intro_duration_ms: int = 5000,
    outro_duration_ms: int = 5000,
    intro_fade_ms: int = 3000,
    outro_fade_ms: int = 3000,
    music_volume_db: int = -6,
) -> dict:
    """Generate a podcast audio file from a conversation using MLX-Audio + Qwen3-TTS.

    Args:
        conversation: List of dicts with 'role' and 'content' keys.
        output_path: Path to save the output audio file.
        language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ru).
        speaker_map: Mapping of roles to speaker names (custom_voice) or
                     reference audio paths (voice_clone).
        silence_duration_ms: Silence between messages in milliseconds.
        progress_callback: Optional callback(current, total).
        task_type: "custom_voice" or "voice_clone".
        instruct: Optional instruction for tone/emotion (custom_voice only).
        model_name: Override HuggingFace model ID.
        temperature: Sampling temperature for generation.
        speed: Playback speed multiplier.
        title: Podcast title for metadata.
        artist: Artist name for metadata.
        album: Album name for metadata.
        track_number: Optional track number for metadata.
        music_path: Path to background music file (None to skip).
        intro_duration_ms: Duration of intro music in ms.
        outro_duration_ms: Duration of outro music in ms.
        intro_fade_ms: Fade-in duration for intro in ms.
        outro_fade_ms: Fade-out duration for outro in ms.
        music_volume_db: Volume adjustment for music in dB.

    Returns:
        Dict with 'path' and 'segment_durations_ms' keys.
    """
    from ..audio_utils import add_background_music, add_metadata
    from ..models import Conversation as ConvModel

    conv = ConvModel.from_dicts(conversation)

    engine = MLXTTSEngine(
        language=language,
        speaker_map=speaker_map,
        task_type=task_type,
        instruct=instruct,
        model_name=model_name,
        temperature=temperature,
        speed=speed,
    )

    if music_path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_voice_path = tmp.name
        try:
            engine.synthesize_conversation(
                conversation=conv,
                output_path=temp_voice_path,
                silence_duration_ms=silence_duration_ms,
                progress_callback=progress_callback,
            )
            add_background_music(
                voice_path=temp_voice_path,
                music_path=music_path,
                output_path=output_path,
                intro_duration_ms=intro_duration_ms,
                outro_duration_ms=outro_duration_ms,
                intro_fade_ms=intro_fade_ms,
                outro_fade_ms=outro_fade_ms,
                music_volume_db=music_volume_db,
            )
        finally:
            if os.path.exists(temp_voice_path):
                os.unlink(temp_voice_path)
    else:
        engine.synthesize_conversation(
            conversation=conv,
            output_path=output_path,
            silence_duration_ms=silence_duration_ms,
            progress_callback=progress_callback,
        )

    if output_path.lower().endswith(".mp3"):
        add_metadata(
            file_path=output_path,
            title=title,
            artist=artist,
            album=album,
            track_number=track_number,
        )

    return {
        "path": output_path,
        "segment_durations_ms": engine.segment_durations_ms,
    }
