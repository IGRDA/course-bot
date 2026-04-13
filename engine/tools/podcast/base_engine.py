"""
Abstract base class for TTS engines.

Defines the common interface that all TTS engines must implement.
"""

from abc import ABC, abstractmethod

from .models import Conversation, Message


class BaseTTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines.

    All TTS engines must implement this interface to be used with the
    podcast generator factory.
    """

    def __init__(
        self,
        language: str = "en",
        speaker_map: dict[str, str] | None = None,
    ):
        """Initialize the TTS engine.

        Args:
            language: Language code (e.g., "en", "es")
            speaker_map: Mapping of role names to speaker/voice IDs
        """
        self.language = language
        self.speaker_map = speaker_map or {}
        self.segment_durations_ms: list[int] = []

    @abstractmethod
    def get_speaker_for_role(self, role: str) -> str:
        """Get the speaker/voice ID for a given role.

        Args:
            role: Speaker role name (e.g., "host", "guest")

        Returns:
            Speaker/voice ID to use for TTS
        """
        pass

    @abstractmethod
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
            language_code: Optional language code override

        Returns:
            Path to the generated audio file
        """
        pass

    @abstractmethod
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
        pass

    @classmethod
    @abstractmethod
    def list_available_voices(cls, language: str = "en") -> list[str]:
        """List available voices for a language.

        Args:
            language: Language code

        Returns:
            List of voice IDs
        """
        pass
