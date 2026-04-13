"""
Edge TTS Engine for podcast generation.

Uses Microsoft Edge's TTS API for high-quality neural voices.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from ..base_engine import BaseTTSEngine
from ..models import Conversation, Message

# Voice configurations for Edge TTS
# Format: {language_code: {role: voice_name}}
EDGE_VOICE_MAP = {
    "en": {
        "host": "en-US-GuyNeural",
        "guest": "en-US-JennyNeural",
    },
    "es": {
        "host": "es-ES-AlvaroNeural",
        "guest": "es-ES-XimenaNeural",
    },
    "fr": {
        "host": "fr-FR-HenriNeural",
        "guest": "fr-FR-DeniseNeural",
    },
    "de": {
        "host": "de-DE-ConradNeural",
        "guest": "de-DE-KatjaNeural",
    },
    "it": {
        "host": "it-IT-DiegoNeural",
        "guest": "it-IT-ElsaNeural",
    },
    "pt": {
        "host": "pt-BR-AntonioNeural",
        "guest": "pt-BR-FranciscaNeural",
    },
    "zh": {
        "host": "zh-CN-YunxiNeural",
        "guest": "zh-CN-XiaoxiaoNeural",
    },
    "ja": {
        "host": "ja-JP-KeitaNeural",
        "guest": "ja-JP-NanamiNeural",
    },
    "ko": {
        "host": "ko-KR-InJoonNeural",
        "guest": "ko-KR-SunHiNeural",
    },
}

# All available Edge TTS voices (subset of most common ones)
EDGE_VOICES = {
    "en": [
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-AmberNeural",
        "en-US-AnaNeural",
        "en-US-AndrewNeural",
        "en-US-EmmaNeural",
        "en-US-BrianNeural",
        "en-GB-RyanNeural",
        "en-GB-SoniaNeural",
        "en-GB-LibbyNeural",
        "en-AU-WilliamNeural",
        "en-AU-NatashaNeural",
    ],
    "es": [
        "es-ES-AlvaroNeural",
        "es-ES-ElviraNeural",
        "es-ES-XimenaNeural",
        "es-MX-DaliaNeural",
        "es-MX-JorgeNeural",
        "es-AR-ElenaNeural",
        "es-AR-TomasNeural",
    ],
    "fr": [
        "fr-FR-HenriNeural",
        "fr-FR-DeniseNeural",
        "fr-FR-AlainNeural",
        "fr-CA-AntoineNeural",
        "fr-CA-SylvieNeural",
    ],
    "de": [
        "de-DE-ConradNeural",
        "de-DE-KatjaNeural",
        "de-DE-AmalaNeural",
        "de-AT-JonasNeural",
        "de-AT-IngridNeural",
    ],
    "it": [
        "it-IT-DiegoNeural",
        "it-IT-ElsaNeural",
        "it-IT-IsabellaNeural",
    ],
    "pt": [
        "pt-BR-AntonioNeural",
        "pt-BR-FranciscaNeural",
        "pt-BR-ThalitaNeural",
        "pt-PT-DuarteNeural",
        "pt-PT-RaquelNeural",
    ],
}


class EdgeTTSEngine(BaseTTSEngine):
    """Text-to-Speech engine using Microsoft Edge TTS.

    Provides high-quality neural voices.
    Requires internet connection.
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: dict[str, str] | None = None,
    ):
        """Initialize the Edge TTS engine.

        Args:
            language: Language code (en, es, fr, de, it, pt, zh, ja, ko)
            speaker_map: Mapping of role names to voice IDs
        """
        super().__init__(language=language, speaker_map=speaker_map)

        # Use default voice map if not provided
        if not self.speaker_map:
            self.speaker_map = EDGE_VOICE_MAP.get(language, EDGE_VOICE_MAP["en"]).copy()

    def get_speaker_for_role(self, role: str) -> str:
        """Get the voice ID for a given role.

        Args:
            role: Speaker role name (e.g., "host", "guest")

        Returns:
            Voice ID to use for Edge TTS
        """
        if role in self.speaker_map:
            return self.speaker_map[role]

        # Assign a default voice for unknown roles
        available_voices = EDGE_VOICES.get(self.language, EDGE_VOICES["en"])
        role_index = hash(role) % len(available_voices)
        voice = available_voices[role_index]
        self.speaker_map[role] = voice
        return voice

    async def _synthesize_async(
        self,
        text: str,
        voice: str,
        output_path: str,
    ) -> str:
        """Async synthesis using edge-tts.

        Args:
            text: Text to synthesize
            voice: Voice ID
            output_path: Path to save the audio file

        Returns:
            Path to the generated audio file
        """
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return output_path

    def synthesize_message(
        self,
        message: Message,
        output_path: str,
        language_code: str | None = None,
    ) -> str:
        """Synthesize a single message to audio.

        Args:
            message: Message to synthesize
            output_path: Path to save the audio file
            language_code: Optional language code (updates voice selection)

        Returns:
            Path to the generated audio file
        """
        voice = self.get_speaker_for_role(message.role)

        # Run async synthesis
        asyncio.run(
            self._synthesize_async(
                text=message.content,
                voice=voice,
                output_path=output_path,
            )
        )

        return output_path

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
        with tempfile.TemporaryDirectory(prefix="edge_tts_") as temp_dir:
            audio_segments = []
            silence = AudioSegment.silent(duration=silence_duration_ms)
            self.segment_durations_ms = []

            total = len(conversation)
            for idx, message in enumerate(conversation):
                temp_audio_path = os.path.join(temp_dir, f"segment_{idx:04d}.mp3")

                # Synthesize this message
                self.synthesize_message(
                    message=message,
                    output_path=temp_audio_path,
                    language_code=language_code,
                )

                # Load the generated audio
                segment = AudioSegment.from_mp3(temp_audio_path)
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
    def list_available_voices(cls, language: str = "en") -> list[str]:
        """List available voices for a language.

        Args:
            language: Language code

        Returns:
            List of voice IDs
        """
        return EDGE_VOICES.get(language, EDGE_VOICES["en"])

    @classmethod
    async def list_all_voices_async(cls) -> list[dict]:
        """List all available Edge TTS voices (async).

        Returns:
            List of voice dictionaries with name, language, gender info
        """
        import edge_tts

        voices = await edge_tts.list_voices()
        return voices

    @classmethod
    def list_all_voices(cls) -> list[dict]:
        """List all available Edge TTS voices.

        Returns:
            List of voice dictionaries with name, language, gender info
        """
        return asyncio.run(cls.list_all_voices_async())


def generate_podcast_edge(
    conversation: list[dict],
    output_path: str,
    language: str = "en",
    speaker_map: dict[str, str] | None = None,
    silence_duration_ms: int = 500,
    progress_callback: callable | None = None,
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
) -> str:
    """Generate a podcast audio file from a conversation using Edge TTS.

    Requires internet connection.

    Args:
        conversation: List of dicts with 'role' and 'content' keys
        output_path: Path to save the output audio file
        language: Language code (en, es, fr, de, it, pt, zh, ja, ko)
        speaker_map: Optional mapping of roles to voice IDs
        silence_duration_ms: Silence duration between messages in milliseconds
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
    from ..audio_utils import add_background_music, add_metadata
    from ..models import Conversation as ConvModel

    # Convert dict list to Conversation object
    conv = ConvModel.from_dicts(conversation)

    # Create engine and synthesize
    engine = EdgeTTSEngine(
        language=language,
        speaker_map=speaker_map,
    )

    # Determine if we need a temp file for voice (when adding music)
    if music_path:
        # Generate voice to temp file, then add music
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_voice:
            temp_voice_path = temp_voice.name

        try:
            # Synthesize voice to temp file
            engine.synthesize_conversation(
                conversation=conv,
                output_path=temp_voice_path,
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
