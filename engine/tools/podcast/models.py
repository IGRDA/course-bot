"""
Data models for podcast generation.

Defines the structure for conversation messages used in TTS synthesis.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """A single message in a podcast conversation.
    
    Attributes:
        role: Speaker role identifier (e.g., "host", "guest")
        content: The text content to be spoken
    """
    role: str
    content: str
    
    def __post_init__(self):
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
        if not self.role.strip():
            raise ValueError("Message role cannot be empty")


@dataclass
class Conversation:
    """A complete podcast conversation consisting of multiple messages.
    
    Attributes:
        messages: List of Message objects representing the dialogue
        title: Optional title for the podcast episode
    """
    messages: list[Message] = field(default_factory=list)
    title: str = "Podcast"
    
    @classmethod
    def from_dicts(cls, data: list[dict], title: str = "Podcast") -> "Conversation":
        """Create a Conversation from a list of dictionaries.
        
        Args:
            data: List of dicts with 'role' and 'content' keys
            title: Optional title for the podcast
            
        Returns:
            Conversation instance
            
        Example:
            >>> conv = Conversation.from_dicts([
            ...     {"role": "host", "content": "Welcome!"},
            ...     {"role": "guest", "content": "Thanks for having me."}
            ... ])
        """
        messages = [Message(role=m["role"], content=m["content"]) for m in data]
        return cls(messages=messages, title=title)
    
    def get_roles(self) -> set[str]:
        """Get all unique speaker roles in the conversation."""
        return {m.role for m in self.messages}
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __iter__(self):
        return iter(self.messages)


# Language configuration for TTS models
@dataclass
class LanguageConfig:
    """Configuration for a specific language's TTS model.
    
    Attributes:
        model_name: Coqui TTS model identifier
        speakers: Available speaker IDs for multi-speaker models
        default_speaker_map: Default mapping of roles to speaker IDs
        language_code: Language code for multilingual models (e.g., "es", "en")
        is_multilingual: Whether this config uses the multilingual XTTS model
    """
    model_name: str
    speakers: list[str | int]
    default_speaker_map: dict[str, str | int]
    language_code: str | None = None
    is_multilingual: bool = False


# XTTS speakers list (shared between es and multilingual configs)
# Verified against actual XTTS v2 model speaker names
XTTS_SPEAKERS = [
    # Female voices
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", 
    "Tammie Ema", "Alison Dietlinde", "Ana Florence",
    "Annmarie Nele", "Asya Anara", "Brenda Stern",
    "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
    "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie",
    "Nova Hogarth", "Maja Ruoho", "Uta Obando",
    "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger",
    "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick",
    "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa",
    "Alma María", "Rosemary Okafor",
    # Male voices
    "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler",
    "Royston Min", "Viktor Eka", "Abrahan Mack",
    "Adde Michal", "Baldur Sanjin", "Craig Gutsy",
    "Damien Black", "Gilberto Mathias", "Ilkin Urbano",
    "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim",
    "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios",
    "Ige Behringer", "Filip Traverse", "Damjan Chapman", 
    "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", 
    "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka", 
    "Luis Moray", "Marcos Rudaski",
]

# Pre-configured language settings
LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        # Use XTTS for most natural English voices
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        speakers=XTTS_SPEAKERS,
        default_speaker_map={"host": "Damien Black", "guest": "Nova Hogarth"},
        language_code="en",
        is_multilingual=True,
    ),
    "en_fast": LanguageConfig(
        # VCTK model - faster but less natural (legacy option)
        model_name="tts_models/en/vctk/vits",
        speakers=["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232"],
        default_speaker_map={"host": "p225", "guest": "p226"},
        language_code=None,
        is_multilingual=False,
    ),
    "es": LanguageConfig(
        # XTTS multilingual model for Spain Spanish voices
        # Uses Spanish-sounding speakers for more natural pronunciation
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        speakers=XTTS_SPEAKERS,
        default_speaker_map={"host": "Luis Moray", "guest": "Alma María"},
        language_code="es",
        is_multilingual=True,
    ),
    "multilingual": LanguageConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        speakers=XTTS_SPEAKERS,
        default_speaker_map={"host": "Damien Black", "guest": "Nova Hogarth"},
        language_code="en",  # Default to English, can be overridden
        is_multilingual=True,
    ),
}


def get_language_config(language: str) -> LanguageConfig:
    """Get language configuration by language code.
    
    Args:
        language: Language code (en, es, multilingual)
        
    Returns:
        LanguageConfig for the specified language
        
    Raises:
        ValueError: If language is not supported
    """
    if language not in LANGUAGE_CONFIGS:
        available = ", ".join(LANGUAGE_CONFIGS.keys())
        raise ValueError(f"Unsupported language '{language}'. Available: {available}")
    return LANGUAGE_CONFIGS[language]

