"""
OpenAI TTS Engine for podcast generation.

Uses the OpenAI gpt-4o-mini-tts model for high-quality, multilingual speech
with controllable accent, tone, and style via the ``instructions`` parameter.
Requires an API key (OPENAI_API_KEY env var) and internet connection.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from ..base_engine import BaseTTSEngine
from ..models import Conversation, Message


# Default voice assignments per language.
# gpt-4o-mini-tts supports all 13 built-in voices for every language.
OPENAI_TTS_VOICE_MAP: dict[str, dict[str, str]] = {
    "es": {"host": "cedar", "guest": "marin"},
    "en": {"host": "cedar", "guest": "marin"},
    "fr": {"host": "cedar", "guest": "marin"},
    "de": {"host": "cedar", "guest": "marin"},
    "it": {"host": "cedar", "guest": "marin"},
    "pt": {"host": "cedar", "guest": "marin"},
    "zh": {"host": "cedar", "guest": "marin"},
    "ja": {"host": "cedar", "guest": "marin"},
    "ko": {"host": "cedar", "guest": "marin"},
}

# Per-language instructions that control accent and delivery style.
OPENAI_TTS_INSTRUCTIONS: dict[str, str] = {
    "es": (
        "Eres un presentador de podcast educativo en español de España. "
        "Habla con un tono natural, cercano y fácil de entender, como en "
        "una conversación entre amigos. "
        "Pronunciación OBLIGATORIA de España (castellano peninsular): "
        "pronuncia la 'z' y la 'c' (ante e, i) como /θ/ (como la 'th' "
        "inglesa en 'think'), NUNCA como /s/. "
        "Pronuncia la 'd' final de palabra suavemente (como /θ/ en 'Madrid'). "
        "Usa entonación y ritmo propios de Madrid/centro de España."
    ),
    "en": "Speak in a natural, conversational educational podcast tone.",
    "fr": "Parle en français avec un ton naturel et conversationnel de podcast éducatif.",
    "de": "Sprich in natürlichem, gesprächigem Podcast-Ton auf Deutsch.",
    "it": "Parla in italiano con un tono naturale e colloquiale da podcast educativo.",
    "pt": "Fale em português com um tom natural e conversacional de podcast educativo.",
    "zh": "用自然、对话式的教育播客语气说中文。",
    "ja": "自然で会話的な教育ポッドキャストのトーンで日本語を話してください。",
    "ko": "자연스럽고 대화하듯 교육 팟캐스트 톤으로 한국어를 말해주세요.",
}

# All 13 built-in voices available in gpt-4o-mini-tts
OPENAI_TTS_ALL_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar",
]


class OpenAITTSEngine(BaseTTSEngine):
    """Text-to-Speech engine using the OpenAI gpt-4o-mini-tts API.

    Provides high-quality multilingual speech with fine-grained control
    over accent and delivery style via the ``instructions`` parameter.
    Requires an API key and internet connection.
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: Optional[dict[str, str]] = None,
        model: str = "gpt-4o-mini-tts",
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
    ):
        """Initialize the OpenAI TTS engine.

        Args:
            language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ...)
            speaker_map: Mapping of role names to OpenAI voice names
            model: OpenAI TTS model identifier
            api_key: API key (falls back to OPENAI_API_KEY env var)
            instructions: Custom instructions for voice style/accent.
                          If None, uses the default for the chosen language.
        """
        super().__init__(language=language, speaker_map=speaker_map)

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var "
                "or pass api_key parameter."
            )

        self.instructions = instructions or OPENAI_TTS_INSTRUCTIONS.get(
            language, OPENAI_TTS_INSTRUCTIONS["en"]
        )

        if not self.speaker_map:
            self.speaker_map = OPENAI_TTS_VOICE_MAP.get(
                language, OPENAI_TTS_VOICE_MAP["en"]
            ).copy()

        self._client = None

    @property
    def client(self):
        """Lazy-initialise the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def get_speaker_for_role(self, role: str) -> str:
        """Get the OpenAI voice name for a given role.

        Args:
            role: Speaker role name (e.g., "host", "guest")

        Returns:
            OpenAI voice name string (e.g., "onyx", "nova")
        """
        if role in self.speaker_map:
            return self.speaker_map[role]

        fallback = OPENAI_TTS_VOICE_MAP.get(self.language, OPENAI_TTS_VOICE_MAP["en"])
        voice = fallback.get("host", "onyx")
        self.speaker_map[role] = voice
        return voice

    def synthesize_message(
        self,
        message: Message,
        output_path: str,
        language_code: Optional[str] = None,
    ) -> str:
        """Synthesize a single message to an MP3 file via OpenAI TTS API.

        Args:
            message: Message to synthesize
            output_path: Path to save the audio file
            language_code: Unused (accent is controlled via instructions).
                           Kept for interface compatibility.

        Returns:
            Path to the generated audio file
        """
        voice = self.get_speaker_for_role(message.role)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=voice,
            input=message.content,
            instructions=self.instructions,
        ) as response:
            response.stream_to_file(output_path)

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
        silence gaps, matching the ElevenLabs/Edge TTS engine pattern.

        Args:
            conversation: Conversation to synthesize
            output_path: Path to save the final audio file
            language_code: Unused (kept for interface compatibility)
            silence_duration_ms: Duration of silence between messages (ms)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Path to the generated audio file
        """
        if len(conversation) == 0:
            raise ValueError("Conversation cannot be empty")

        from pydub import AudioSegment

        with tempfile.TemporaryDirectory(prefix="openai_tts_") as temp_dir:
            audio_segments: list[AudioSegment] = []
            silence = AudioSegment.silent(duration=silence_duration_ms)
            self.segment_durations_ms = []

            total = len(conversation)
            for idx, message in enumerate(conversation):
                temp_audio_path = os.path.join(temp_dir, f"segment_{idx:04d}.mp3")

                self.synthesize_message(
                    message=message,
                    output_path=temp_audio_path,
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
        """List available voices (all 13 built-in voices are language-agnostic).

        Args:
            language: Language code (unused -- all voices work for all languages)

        Returns:
            List of voice name strings
        """
        return list(OPENAI_TTS_ALL_VOICES)


def generate_podcast_openai_tts(
    conversation: list[dict],
    output_path: str,
    language: str = "en",
    speaker_map: Optional[dict[str, str]] = None,
    silence_duration_ms: int = 500,
    progress_callback: Optional[callable] = None,
    # OpenAI-specific
    model: str = "gpt-4o-mini-tts",
    api_key: Optional[str] = None,
    instructions: Optional[str] = None,
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
    """Generate a podcast audio file from a conversation using OpenAI TTS.

    Requires internet connection and a valid OpenAI API key.

    Args:
        conversation: List of dicts with 'role' and 'content' keys
        output_path: Path to save the output audio file
        language: Language code (en, es, fr, de, it, pt, zh, ja, ko, ...)
        speaker_map: Optional mapping of roles to OpenAI voice names
        silence_duration_ms: Silence between messages in milliseconds
        progress_callback: Optional callback(current, total)
        model: OpenAI TTS model identifier
        api_key: API key (falls back to OPENAI_API_KEY env var)
        instructions: Custom instructions for voice style/accent
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

    engine = OpenAITTSEngine(
        language=language,
        speaker_map=speaker_map,
        model=model,
        api_key=api_key,
        instructions=instructions,
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
