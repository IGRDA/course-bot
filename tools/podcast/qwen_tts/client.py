"""
Qwen3-TTS Engine for podcast generation.

Uses Alibaba's Qwen3-TTS models for high-quality multilingual speech synthesis
with support for built-in custom voices and voice cloning from reference audio.

Requires ``pip install qwen-tts``. Works on CUDA (fastest), CPU (slow but
functional), and experimentally on MPS.

https://github.com/QwenLM/Qwen3-TTS
"""

import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

from ..base_engine import BaseTTSEngine
from ..models import Conversation, Message


TaskType = Literal["custom_voice", "voice_clone"]


def _auto_device() -> str:
    """Pick the best available device: cuda > cpu.

    MPS is excluded by default because Qwen3-TTS triggers LLVM errors on
    Apple Silicon MPS as of qwen-tts 0.1.1.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

# Qwen3-TTS supported languages (code -> full name expected by the library)
QWEN_LANGUAGE_MAP: dict[str, str] = {
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
QWEN_VOICE_MAP: dict[str, dict[str, str]] = {
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

QWEN_ALL_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

# HuggingFace model IDs
QWEN_MODEL_CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class QwenTTSEngine(BaseTTSEngine):
    """Text-to-Speech engine using Alibaba's Qwen3-TTS.

    Supports two modes via ``task_type``:
      - ``"custom_voice"`` (default): 9 built-in speakers with optional
        instruction-based tone/emotion control.
      - ``"voice_clone"``: Clone any voice from a short reference audio clip.
        The ``speaker_map`` should map roles to audio file paths.

    Requires a CUDA GPU (~4 GB VRAM for 1.7B model).
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: Optional[dict[str, str]] = None,
        task_type: TaskType = "custom_voice",
        device: Optional[str] = None,
        instruct: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize the Qwen3-TTS engine.

        Args:
            language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ru)
            speaker_map: For custom_voice: role -> speaker name.
                         For voice_clone: role -> path to reference audio file.
            task_type: "custom_voice" for built-in speakers or
                       "voice_clone" for cloning from reference audio.
            device: PyTorch device string (e.g. "cuda:0", "cpu").
                    Auto-detected if None.
            instruct: Optional instruction for tone/emotion control
                      (custom_voice mode only, e.g. "Speak warmly").
            model_name: Override the HuggingFace model ID.
        """
        super().__init__(language=language, speaker_map=speaker_map)

        if language not in QWEN_LANGUAGE_MAP:
            available = ", ".join(sorted(QWEN_LANGUAGE_MAP.keys()))
            raise ValueError(
                f"Unsupported language '{language}'. Available: {available}"
            )

        self.task_type = task_type
        self.device = device or _auto_device()
        self.instruct = instruct
        self.language_full = QWEN_LANGUAGE_MAP[language]

        if model_name:
            self.model_name = model_name
        elif task_type == "voice_clone":
            self.model_name = QWEN_MODEL_BASE
        else:
            self.model_name = QWEN_MODEL_CUSTOM_VOICE

        if not self.speaker_map and task_type == "custom_voice":
            self.speaker_map = QWEN_VOICE_MAP.get(
                language, QWEN_VOICE_MAP["en"]
            ).copy()

        self._model = None
        self._clone_prompts: dict[str, object] = {}

    @property
    def model(self):
        """Lazy-load the Qwen3-TTS model on first access."""
        if self._model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            dtype = torch.bfloat16 if "cuda" in self.device else torch.float32

            print(f"Loading Qwen3-TTS model {self.model_name} on {self.device}...")
            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    dtype=dtype,
                    attn_implementation="flash_attention_2",
                )
            except (ImportError, ValueError):
                self._model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    dtype=dtype,
                )
            print("Qwen3-TTS model loaded successfully.")
        return self._model

    def _get_clone_prompt(self, ref_audio_path: str):
        """Build or retrieve a cached voice-clone prompt for a reference audio.

        Args:
            ref_audio_path: Path to the reference audio file.

        Returns:
            Reusable voice_clone_prompt object.
        """
        if ref_audio_path not in self._clone_prompts:
            self._clone_prompts[ref_audio_path] = (
                self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text="",
                    x_vector_only_mode=True,
                )
            )
        return self._clone_prompts[ref_audio_path]

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
            fallback = QWEN_VOICE_MAP.get(self.language, QWEN_VOICE_MAP["en"])
            speaker = fallback.get("host", "Ryan")
            self.speaker_map[role] = speaker
            return speaker

        return role

    def synthesize_message(
        self,
        message: Message,
        output_path: str,
        language_code: Optional[str] = None,
    ) -> str:
        """Synthesize a single message to a WAV file.

        Args:
            message: Message to synthesize.
            output_path: Path to save the audio file.
            language_code: Optional language code override.

        Returns:
            Path to the generated audio file.
        """
        import soundfile as sf

        lang_full = QWEN_LANGUAGE_MAP.get(
            language_code, self.language_full
        ) if language_code else self.language_full

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.task_type == "voice_clone":
            ref_audio_path = self.get_speaker_for_role(message.role)
            if not Path(ref_audio_path).exists():
                raise FileNotFoundError(
                    f"Reference audio not found for role '{message.role}': "
                    f"{ref_audio_path}"
                )
            clone_prompt = self._get_clone_prompt(ref_audio_path)
            wavs, sr = self.model.generate_voice_clone(
                text=message.content,
                language=lang_full,
                voice_clone_prompt=clone_prompt,
            )
        else:
            speaker = self.get_speaker_for_role(message.role)
            gen_kwargs: dict = {
                "text": message.content,
                "language": lang_full,
                "speaker": speaker,
            }
            if self.instruct:
                gen_kwargs["instruct"] = self.instruct
            wavs, sr = self.model.generate_custom_voice(**gen_kwargs)

        sf.write(output_path, wavs[0], sr)
        return output_path

    def synthesize_conversation(
        self,
        conversation: Conversation,
        output_path: str,
        language_code: Optional[str] = None,
        silence_duration_ms: int = 500,
        progress_callback: Optional[callable] = None,
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

        with tempfile.TemporaryDirectory(prefix="qwen_tts_") as temp_dir:
            audio_segments: list[AudioSegment] = []
            silence = AudioSegment.silent(duration=silence_duration_ms)

            total = len(conversation)
            for idx, message in enumerate(conversation):
                temp_path = os.path.join(temp_dir, f"segment_{idx:04d}.wav")
                self.synthesize_message(
                    message=message,
                    output_path=temp_path,
                    language_code=language_code,
                )
                audio_segments.append(AudioSegment.from_wav(temp_path))
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
        return list(QWEN_ALL_SPEAKERS)


def generate_podcast_qwen_tts(
    conversation: list[dict],
    output_path: str,
    language: str = "en",
    speaker_map: Optional[dict[str, str]] = None,
    silence_duration_ms: int = 500,
    progress_callback: Optional[callable] = None,
    task_type: TaskType = "custom_voice",
    device: Optional[str] = None,
    instruct: Optional[str] = None,
    model_name: Optional[str] = None,
    # Metadata options
    title: str = "Module",
    artist: str = "Adinhub",
    album: str = "Course",
    track_number: Optional[int] = None,
    # Background music options
    music_path: Optional[str] = None,
    intro_duration_ms: int = 5000,
    outro_duration_ms: int = 5000,
    intro_fade_ms: int = 3000,
    outro_fade_ms: int = 3000,
    music_volume_db: int = -6,
) -> str:
    """Generate a podcast audio file from a conversation using Qwen3-TTS.

    Args:
        conversation: List of dicts with 'role' and 'content' keys.
        output_path: Path to save the output audio file.
        language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ru).
        speaker_map: Mapping of roles to speaker names (custom_voice) or
                     reference audio paths (voice_clone).
        silence_duration_ms: Silence between messages in milliseconds.
        progress_callback: Optional callback(current, total).
        task_type: "custom_voice" or "voice_clone".
        device: PyTorch device string.
        instruct: Optional instruction for tone/emotion (custom_voice only).
        model_name: Override HuggingFace model ID.
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
        Path to the generated audio file.
    """
    from ..audio_utils import add_background_music, add_metadata
    from ..models import Conversation as ConvModel

    conv = ConvModel.from_dicts(conversation)

    engine = QwenTTSEngine(
        language=language,
        speaker_map=speaker_map,
        task_type=task_type,
        device=device,
        instruct=instruct,
        model_name=model_name,
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

    return output_path
