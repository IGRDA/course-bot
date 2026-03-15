"""
Coqui TTS Engine for podcast generation.

Provides functionality to synthesize multi-speaker podcast audio from conversations.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from ..base_engine import BaseTTSEngine
from ..models import (
    Conversation,
    Message,
    LanguageConfig,
    get_language_config,
    LANGUAGE_CONFIGS,
)


def _patch_torch_load_for_tts():
    """Patch TTS library to use weights_only=False for model loading.

    PyTorch 2.6+ changed torch.load() to use weights_only=True by default.
    TTS model checkpoints require weights_only=False to load properly.
    This patches the TTS io module to pass the correct argument.
    """
    try:
        import torch
        import TTS.utils.io as tts_io

        # Store original function
        original_load_fsspec = tts_io.load_fsspec

        def patched_load_fsspec(path, map_location=None, **kwargs):
            # Force weights_only=False for TTS model loading
            kwargs["weights_only"] = False
            return original_load_fsspec(path, map_location=map_location, **kwargs)

        # Apply patch
        tts_io.load_fsspec = patched_load_fsspec
    except (ImportError, AttributeError):
        pass


class CoquiTTSEngine(BaseTTSEngine):
    """Text-to-Speech engine using Coqui TTS for multi-speaker podcast generation.

    Works offline without internet connection. Does not support SSML.
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: Optional[dict[str, str | int]] = None,
        device: str = "cpu",
    ):
        """Initialize the TTS engine.

        Args:
            language: Language code (en, es, multilingual)
            speaker_map: Mapping of role names to speaker IDs
            device: Device to run TTS on (cpu, cuda)
        """
        super().__init__(language=language, speaker_map=speaker_map)
        self.config = get_language_config(language)
        self.device = device

        # Use default speaker map if not provided
        if not self.speaker_map:
            self.speaker_map = self.config.default_speaker_map.copy()

        # Lazy load the TTS model
        self._tts = None

    @property
    def tts(self):
        """Lazy-load the TTS model on first access."""
        if self._tts is None:
            # Patch TTS for PyTorch 2.6+ compatibility
            if self.config.is_multilingual:
                _patch_torch_load_for_tts()

            from TTS.api import TTS
            print(f"Loading TTS model: {self.config.model_name}")
            self._tts = TTS(model_name=self.config.model_name).to(self.device)
        return self._tts

    def get_speaker_for_role(self, role: str) -> str | int:
        """Get the speaker ID for a given role.

        Args:
            role: Speaker role name

        Returns:
            Speaker ID to use for TTS
        """
        if role in self.speaker_map:
            return self.speaker_map[role]

        # Assign a default speaker for unknown roles
        available_speakers = self.config.speakers
        role_index = hash(role) % len(available_speakers)
        speaker = available_speakers[role_index]
        self.speaker_map[role] = speaker
        return speaker

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
            language_code: Optional language code for multilingual models (overrides config)

        Returns:
            Path to the generated audio file
        """
        speaker = self.get_speaker_for_role(message.role)

        # Build TTS arguments
        tts_kwargs = {
            "text": message.content,
            "file_path": output_path,
        }

        # Add speaker for multi-speaker models
        if self.config.is_multilingual:
            # XTTS multilingual model requires speaker and language
            tts_kwargs["speaker"] = speaker
            # Use provided language_code, or fall back to config default
            tts_kwargs["language"] = language_code or self.config.language_code or "en"
        elif len(self.config.speakers) > 1:
            # Multi-speaker model (like VCTK) - just needs speaker
            tts_kwargs["speaker"] = speaker
        # For single-speaker models, no speaker argument needed

        self.tts.tts_to_file(**tts_kwargs)
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
            conversation: Conversation to synthesize
            output_path: Path to save the final audio file
            language_code: Optional language code for multilingual models
            silence_duration_ms: Duration of silence between messages (ms)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Path to the generated audio file
        """
        if len(conversation) == 0:
            raise ValueError("Conversation cannot be empty")

        from pydub import AudioSegment

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory(prefix="podcast_tts_") as temp_dir:
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
                output_format = "mp3"  # Default to mp3

            combined.export(str(output_path), format=output_format)

        return str(output_path)

    @classmethod
    def list_available_voices(cls, language: str = "en") -> list[str | int]:
        """List available speakers for a language.

        Args:
            language: Language code

        Returns:
            List of speaker IDs
        """
        config = get_language_config(language)
        return config.speakers


# Backward compatibility alias
TTSEngine = CoquiTTSEngine


def generate_podcast(
    conversation: list[dict],
    output_path: str,
    language: str = "en",
    speaker_map: Optional[dict[str, str | int]] = None,
    language_code: Optional[str] = None,
    silence_duration_ms: int = 500,
    device: str = "cpu",
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
    """Generate a podcast audio file from a conversation using Coqui TTS.

    This is a convenience function that uses the Coqui TTS engine.
    For Edge TTS, use generate_podcast_edge instead.

    Args:
        conversation: List of dicts with 'role' and 'content' keys
        output_path: Path to save the output audio file
        language: TTS language/model to use (en, es, multilingual)
        speaker_map: Optional mapping of roles to speaker IDs
        language_code: Language code for multilingual model (e.g., "es", "en", "fr")
        silence_duration_ms: Silence duration between messages in milliseconds
        device: Device to run TTS on (cpu, cuda)
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

    # Create engine and synthesize
    engine = CoquiTTSEngine(
        language=language,
        speaker_map=speaker_map,
        device=device,
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
                language_code=language_code,
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
            language_code=language_code,
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


def list_available_languages() -> list[str]:
    """Get list of available language codes for Coqui TTS."""
    return list(LANGUAGE_CONFIGS.keys())


def list_speakers(language: str) -> list[str | int]:
    """Get list of available speakers for a Coqui TTS language.

    Args:
        language: Language code

    Returns:
        List of speaker IDs
    """
    config = get_language_config(language)
    return config.speakers
