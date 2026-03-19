"""
Chatterbox TTS Engine for podcast generation.

Uses Resemble AI's Chatterbox Multilingual TTS for high-quality
zero-shot voice synthesis in 23 languages.

https://huggingface.co/ResembleAI/chatterbox
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from ..base_engine import BaseTTSEngine
from ..models import Conversation, Message


# =============================================================================
# Editable defaults for emotion control
# =============================================================================
# Emotion intensity (0.0-1.0). Higher values = more expressive speech.
# Default 0.5 works well for most use cases.
DEFAULT_EXAGGERATION = 0.5

# Classifier-free guidance weight (0.0-1.0).
# Lower values = slower, more deliberate pacing.
# Higher values = faster, more natural pacing.
# Default 0.5 is recommended. Use ~0.3 for expressive/dramatic speech.
DEFAULT_CFG_WEIGHT = 0.5


# Supported languages with their Chatterbox language IDs
# Chatterbox Multilingual supports 23 languages
CHATTERBOX_LANGUAGES = {
    "ar": "ar",  # Arabic
    "da": "da",  # Danish
    "de": "de",  # German
    "el": "el",  # Greek
    "en": "en",  # English
    "es": "es",  # Spanish
    "fi": "fi",  # Finnish
    "fr": "fr",  # French
    "he": "he",  # Hebrew
    "hi": "hi",  # Hindi
    "it": "it",  # Italian
    "ja": "ja",  # Japanese
    "ko": "ko",  # Korean
    "ms": "ms",  # Malay
    "nl": "nl",  # Dutch
    "no": "no",  # Norwegian
    "pl": "pl",  # Polish
    "pt": "pt",  # Portuguese
    "ru": "ru",  # Russian
    "sv": "sv",  # Swedish
    "sw": "sw",  # Swahili
    "tr": "tr",  # Turkish
    "zh": "zh",  # Chinese
}


class ChatterboxEngine(BaseTTSEngine):
    """Text-to-Speech engine using Resemble AI's Chatterbox Multilingual.

    Provides high-quality zero-shot TTS in 23 languages.
    Requires GPU for reasonable performance (runs on CPU but slower).

    Features:
        - 23 language support
        - Emotion exaggeration control
        - 0.5B Llama backbone
        - Built-in watermarking for responsible AI
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: Optional[dict[str, str]] = None,
        device: str = "cuda",
        exaggeration: float = DEFAULT_EXAGGERATION,
        cfg_weight: float = DEFAULT_CFG_WEIGHT,
    ):
        """Initialize the Chatterbox TTS engine.

        Args:
            language: Language code (see CHATTERBOX_LANGUAGES for supported codes)
            speaker_map: Optional mapping of roles to speaker IDs (reserved for future use)
            device: Device to run model on ("cuda" or "cpu")
            exaggeration: Emotion intensity (0.0-1.0), default 0.5
            cfg_weight: Classifier-free guidance weight (0.0-1.0), default 0.5
        """
        super().__init__(language=language, speaker_map=speaker_map)

        # Validate language
        if language not in CHATTERBOX_LANGUAGES:
            available = ", ".join(sorted(CHATTERBOX_LANGUAGES.keys()))
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Available: {available}"
            )

        self.device = device
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.language_id = CHATTERBOX_LANGUAGES[language]

        # Lazy load the model
        self._model = None

    @property
    def model(self):
        """Lazy-load the Chatterbox model on first access."""
        if self._model is None:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            # Patch torch.load to handle CUDA checkpoints on non-CUDA devices
            # The chatterbox library doesn't pass map_location, causing failures
            # on CPU/MPS when loading models saved on CUDA
            original_torch_load = torch.load

            def patched_torch_load(f, *args, **kwargs):
                # Force map_location to the target device if not specified
                if "map_location" not in kwargs:
                    kwargs["map_location"] = self.device
                return original_torch_load(f, *args, **kwargs)

            torch.load = patched_torch_load

            try:
                print(f"Loading Chatterbox Multilingual TTS model on {self.device}...")
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                print("Chatterbox model loaded successfully.")
            finally:
                # Restore original torch.load
                torch.load = original_torch_load

        return self._model

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self.model.sr

    def get_speaker_for_role(self, role: str) -> str:
        """Get the speaker ID for a given role.

        Note: Chatterbox uses audio prompts for voice cloning, not speaker IDs.
        This method is implemented for interface compatibility but currently
        returns a placeholder. Future versions may support audio prompt paths.

        Args:
            role: Speaker role name (e.g., "host", "guest")

        Returns:
            Placeholder speaker identifier
        """
        # For now, return role name as placeholder
        # Future: could map roles to audio prompt file paths
        if role in self.speaker_map:
            return self.speaker_map[role]
        return f"default_{role}"

    def synthesize_message(
        self,
        message: Message,
        output_path: str,
        language_code: Optional[str] = None,
    ) -> str:
        """Synthesize a single message to audio.

        Args:
            message: Message to synthesize
            output_path: Path to save the audio file
            language_code: Optional language code override

        Returns:
            Path to the generated audio file
        """
        import torchaudio as ta

        # Use provided language or fall back to instance default
        lang_id = language_code or self.language_id
        if lang_id not in CHATTERBOX_LANGUAGES.values():
            # Try to map from our language codes
            lang_id = CHATTERBOX_LANGUAGES.get(lang_id, self.language_id)

        # Check if we have a voice reference audio for this role
        audio_prompt_path = None
        if message.role in self.speaker_map:
            voice_ref = self.speaker_map[message.role]
            # If the value is a path to an audio file, use it for voice cloning
            if voice_ref and Path(voice_ref).exists():
                audio_prompt_path = voice_ref

        # Build generation kwargs
        gen_kwargs = {
            "language_id": lang_id,
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
        }

        # Add audio prompt if available (for voice cloning)
        if audio_prompt_path:
            gen_kwargs["audio_prompt_path"] = audio_prompt_path

        # Generate audio
        wav = self.model.generate(message.content, **gen_kwargs)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Chatterbox outputs tensor, save as wav
        ta.save(str(output_path), wav, self.sample_rate)

        return str(output_path)

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
            conversation: Conversation to synthesize
            output_path: Path to save the final audio file
            language_code: Optional language code override
            silence_duration_ms: Duration of silence between messages (ms)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Path to the generated audio file
        """
        if len(conversation) == 0:
            raise ValueError("Conversation cannot be empty")

        from pydub import AudioSegment

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory(prefix="chatterbox_tts_") as temp_dir:
            audio_segments = []
            silence = AudioSegment.silent(duration=silence_duration_ms)
            self.segment_durations_ms = []

            total = len(conversation)
            for idx, message in enumerate(conversation):
                temp_audio_path = os.path.join(temp_dir, f"segment_{idx:04d}.wav")

                # Synthesize this message
                self.synthesize_message(
                    message=message,
                    output_path=temp_audio_path,
                    language_code=language_code,
                )

                # Load the generated audio
                segment = AudioSegment.from_wav(temp_audio_path)
                audio_segments.append(segment)
                self.segment_durations_ms.append(len(segment))

                if progress_callback:
                    progress_callback(idx + 1, total)

            # Concatenate all segments with silence between them
            combined = audio_segments[0]
            for segment in audio_segments[1:]:
                combined = combined + silence + segment

            # Export to output path
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine format from extension
            output_format = output_path.suffix.lstrip(".").lower()
            if output_format not in ("mp3", "wav", "ogg", "flac"):
                output_format = "wav"  # Default to wav for Chatterbox

            combined.export(str(output_path), format=output_format)

        return str(output_path)

    @classmethod
    def list_available_voices(cls, language: str = "en") -> list[str]:
        """List available voices for a language.

        Note: Chatterbox uses a single default voice per synthesis.
        Voice cloning via audio prompts may be supported in future versions.

        Args:
            language: Language code

        Returns:
            List with single "default" voice
        """
        if language not in CHATTERBOX_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        return ["default"]

    @classmethod
    def list_supported_languages(cls) -> list[str]:
        """List all supported language codes.

        Returns:
            List of language codes (e.g., ["ar", "da", "de", ...])
        """
        return sorted(CHATTERBOX_LANGUAGES.keys())


def generate_podcast_chatterbox(
    conversation: list[dict],
    output_path: str,
    language: str = "es",
    speaker_map: Optional[dict[str, str]] = None,
    silence_duration_ms: int = 500,
    device: str = "mps",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    progress_callback: Optional[callable] = None,
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
    """Generate a podcast audio file from a conversation using Chatterbox TTS.

    Supports voice cloning via speaker_map paths to audio files.

    Args:
        conversation: List of dicts with 'role' and 'content' keys
        output_path: Path to save the output audio file
        language: Language code (23 languages supported)
        speaker_map: Mapping of roles to audio file paths for voice cloning
        silence_duration_ms: Silence duration between messages in milliseconds
        device: Device to run TTS on (cuda, mps, cpu)
        exaggeration: Emotion intensity (0.0-1.0), default 0.5
        cfg_weight: Classifier-free guidance (0.0-1.0), default 0.5
        progress_callback: Optional callback(current, total) for progress updates
        title: Podcast title for metadata (default: "Module")
        artist: Artist name for metadata (default: "Adinhub")
        album: Album name for metadata (default: "Course")
        track_number: Optional track number for metadata
        music_path: Path to background music file (None to skip music)
        intro_duration_ms: Duration of intro music in ms (default: 5000)
        outro_duration_ms: Duration of outro music in ms (default: 5000)
        intro_fade_ms: Fade-in duration for intro in ms (default: 3000)
        outro_fade_ms: Fade-out duration for outro in ms (default: 3000)
        music_volume_db: Volume adjustment for music in dB (default: -6)

    Returns:
        Path to the generated audio file
    """
    from ..audio_utils import add_metadata, add_background_music
    from ..models import Conversation as ConvModel

    # Convert dict list to Conversation object
    conv = ConvModel.from_dicts(conversation)

    # Create engine with voice cloning support
    engine = ChatterboxEngine(
        language=language,
        speaker_map=speaker_map,
        device=device,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    # Determine if we need a temp file for voice (when adding music)
    if music_path:
        # Generate voice to temp file, then add music
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_voice:
            temp_voice_path = temp_voice.name

        try:
            # Synthesize voice to temp file
            engine.synthesize_conversation(
                conversation=conv,
                output_path=temp_voice_path,
                language_code=language,
                silence_duration_ms=silence_duration_ms,
                progress_callback=progress_callback,
            )

            # Add background music
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
            # Clean up temp file
            if os.path.exists(temp_voice_path):
                os.unlink(temp_voice_path)
    else:
        # No music, generate directly to output
        engine.synthesize_conversation(
            conversation=conv,
            output_path=output_path,
            language_code=language,
            silence_duration_ms=silence_duration_ms,
            progress_callback=progress_callback,
        )

    # Add metadata if output is MP3
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
