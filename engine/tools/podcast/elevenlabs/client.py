"""
ElevenLabs TTS Engine for podcast generation.

Uses the ElevenLabs API for high-quality, multilingual neural voices.
Requires an API key (ELEVENLABS_API_KEY env var) and internet connection.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from ..base_engine import BaseTTSEngine
from ..models import Conversation, Message


# Premade ElevenLabs voice IDs.
# These voices support 29+ languages via the language_code parameter.
ELEVENLABS_VOICE_MAP = {
    "en": {
        "host": "pNInz6obpgDQGcFmaJgB",      # Adam (male)
        "guest": "21m00Tcm4TlvDq8ikWAM",      # Rachel (female)
    },
    "es": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
    "fr": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
    "de": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
    "it": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
    "pt": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
    "zh": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
    "ja": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
    "ko": {
        "host": "pNInz6obpgDQGcFmaJgB",
        "guest": "21m00Tcm4TlvDq8ikWAM",
    },
}

# ISO 639-1 language codes accepted by ElevenLabs multilingual models
ELEVENLABS_LANGUAGE_CODES = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
    "ar": "ar",
    "hi": "hi",
    "pl": "pl",
    "ru": "ru",
    "nl": "nl",
    "sv": "sv",
    "tr": "tr",
}


class ElevenLabsTTSEngine(BaseTTSEngine):
    """Text-to-Speech engine using the ElevenLabs API.

    Provides high-quality multilingual neural voices via cloud API.
    Requires an API key and internet connection.
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: Optional[dict[str, str]] = None,
        model_id: str = "eleven_multilingual_v2",
        api_key: Optional[str] = None,
    ):
        """Initialize the ElevenLabs TTS engine.

        Args:
            language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ...)
            speaker_map: Mapping of role names to ElevenLabs voice IDs
            model_id: ElevenLabs model identifier
            api_key: API key (falls back to ELEVENLABS_API_KEY env var)
        """
        super().__init__(language=language, speaker_map=speaker_map)

        self.model_id = model_id
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY env var "
                "or pass api_key parameter."
            )

        if not self.speaker_map:
            self.speaker_map = ELEVENLABS_VOICE_MAP.get(
                language, ELEVENLABS_VOICE_MAP["en"]
            ).copy()

        self._client = None

    @property
    def client(self):
        """Lazy-initialise the ElevenLabs client."""
        if self._client is None:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    def get_speaker_for_role(self, role: str) -> str:
        """Get the ElevenLabs voice ID for a given role.

        Args:
            role: Speaker role name (e.g., "host", "guest")

        Returns:
            ElevenLabs voice ID string
        """
        if role in self.speaker_map:
            return self.speaker_map[role]

        # Fall back to the host voice for unknown roles
        fallback = ELEVENLABS_VOICE_MAP.get(self.language, ELEVENLABS_VOICE_MAP["en"])
        voice = fallback.get("host", "pNInz6obpgDQGcFmaJgB")
        self.speaker_map[role] = voice
        return voice

    def synthesize_message(
        self,
        message: Message,
        output_path: str,
        language_code: Optional[str] = None,
    ) -> str:
        """Synthesize a single message to an MP3 file via ElevenLabs API.

        Args:
            message: Message to synthesize
            output_path: Path to save the audio file
            language_code: Optional ISO 639-1 language code override

        Returns:
            Path to the generated audio file
        """
        voice_id = self.get_speaker_for_role(message.role)
        lang_code = language_code or ELEVENLABS_LANGUAGE_CODES.get(self.language)

        kwargs = dict(
            voice_id=voice_id,
            text=message.content,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        if lang_code:
            kwargs["language_code"] = lang_code

        audio_iter = self.client.text_to_speech.convert(**kwargs)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio_iter:
                f.write(chunk)

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

        Each message is synthesised individually then concatenated with
        silence gaps, matching the Edge TTS engine pattern.

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

        with tempfile.TemporaryDirectory(prefix="elevenlabs_tts_") as temp_dir:
            audio_segments = []
            silence = AudioSegment.silent(duration=silence_duration_ms)
            self.segment_durations_ms = []

            total = len(conversation)
            for idx, message in enumerate(conversation):
                temp_audio_path = os.path.join(temp_dir, f"segment_{idx:04d}.mp3")

                self.synthesize_message(
                    message=message,
                    output_path=temp_audio_path,
                    language_code=language_code,
                )

                segment = AudioSegment.from_mp3(temp_audio_path)
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
                output_format = "mp3"

            combined.export(str(out), format=output_format)

        return str(output_path)

    @classmethod
    def list_available_voices(cls, language: str = "en") -> list[str]:
        """List default voice IDs for a language.

        Args:
            language: Language code

        Returns:
            List of voice IDs
        """
        voice_map = ELEVENLABS_VOICE_MAP.get(language, ELEVENLABS_VOICE_MAP["en"])
        return list(voice_map.values())


def generate_podcast_elevenlabs(
    conversation: list[dict],
    output_path: str,
    language: str = "en",
    speaker_map: Optional[dict[str, str]] = None,
    silence_duration_ms: int = 500,
    progress_callback: Optional[callable] = None,
    # ElevenLabs-specific
    model_id: str = "eleven_multilingual_v2",
    api_key: Optional[str] = None,
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
    """Generate a podcast audio file from a conversation using ElevenLabs TTS.

    Requires internet connection and a valid ElevenLabs API key.

    Args:
        conversation: List of dicts with 'role' and 'content' keys
        output_path: Path to save the output audio file
        language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ...)
        speaker_map: Optional mapping of roles to ElevenLabs voice IDs
        silence_duration_ms: Silence between messages in milliseconds
        progress_callback: Optional callback(current, total)
        model_id: ElevenLabs model identifier
        api_key: API key (falls back to ELEVENLABS_API_KEY env var)
        title: Podcast title for metadata
        artist: Artist name for metadata
        album: Album name for metadata
        track_number: Optional track number for metadata
        music_path: Path to background music file (None to skip)
        intro_duration_ms: Duration of intro music in ms
        outro_duration_ms: Duration of outro music in ms
        intro_fade_ms: Fade-in duration for intro in ms
        outro_fade_ms: Fade-out duration for outro in ms
        music_volume_db: Volume adjustment for music in dB

    Returns:
        Path to the generated audio file
    """
    from ..audio_utils import add_background_music, add_metadata
    from ..models import Conversation as ConvModel

    conv = ConvModel.from_dicts(conversation)

    engine = ElevenLabsTTSEngine(
        language=language,
        speaker_map=speaker_map,
        model_id=model_id,
        api_key=api_key,
    )

    if music_path:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_voice:
            temp_voice_path = temp_voice.name

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
