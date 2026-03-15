"""
Audio utilities for podcast generation.

Provides functionality for:
- Adding ID3 metadata to MP3 files
- Adding background music (intro/outro) with fade effects

Heavy dependencies (pydub, mutagen) are imported lazily inside the
functions that need them.
"""

from pathlib import Path
from typing import Optional


def add_metadata(
    file_path: str,
    title: str = "Module",
    artist: str = "Adinhub",
    album: str = "Course",
    track_number: Optional[int] = None,
) -> None:
    """Add ID3 metadata to an MP3 file.
    
    Args:
        file_path: Path to the MP3 file
        title: Track title (default: "Module")
        artist: Artist name (default: "Adinhub")
        album: Album name (default: "Course")
        track_number: Optional track number
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not an MP3
    """
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import ID3NoHeaderError
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() != ".mp3":
        raise ValueError(f"File must be an MP3: {file_path}")
    
    try:
        audio = EasyID3(file_path)
    except ID3NoHeaderError:
        # Create ID3 header if it doesn't exist
        from mutagen.id3 import ID3
        audio_file = ID3()
        audio_file.save(file_path)
        audio = EasyID3(file_path)
    
    audio["title"] = title
    audio["artist"] = artist
    audio["album"] = album
    
    if track_number is not None:
        audio["tracknumber"] = str(track_number)
    
    audio.save()


def add_background_music(
    voice_path: str,
    music_path: str,
    output_path: str,
    intro_duration_ms: int = 5000,
    outro_duration_ms: int = 5000,
    intro_fade_ms: int = 3000,
    outro_fade_ms: int = 3000,
    music_volume_db: int = -6,
) -> str:
    """Add background music (intro/outro) to voice audio.
    
    Creates: [Intro → fade out] → [Voice] → [Outro fade in → fade out]
    
    The intro music plays then gradually reduces (fades out) before the voice.
    The outro music gradually comes in (fades in) after the voice, then fades out.
    
    Args:
        voice_path: Path to the voice audio file
        music_path: Path to the background music file
        output_path: Path to save the final audio
        intro_duration_ms: Duration of intro music in milliseconds (default: 5000)
        outro_duration_ms: Duration of outro music in milliseconds (default: 5000)
        intro_fade_ms: Fade-out duration at end of intro in milliseconds (default: 3000)
        outro_fade_ms: Fade-out duration at end of outro in milliseconds (default: 3000)
        music_volume_db: Volume adjustment for music in dB (default: -6)
        
    Returns:
        Path to the output file
        
    Raises:
        FileNotFoundError: If voice or music file doesn't exist
    """
    voice_file = Path(voice_path)
    music_file = Path(music_path)
    
    if not voice_file.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    if not music_file.exists():
        raise FileNotFoundError(f"Music file not found: {music_path}")
    
    from pydub import AudioSegment

    # Load audio files
    voice = AudioSegment.from_file(str(voice_file))
    music = AudioSegment.from_file(str(music_file))
    
    # Adjust music volume
    music = music + music_volume_db
    
    # Get music duration
    music_duration = len(music)
    
    # Extract intro segment (from beginning of music)
    # Ensure we don't exceed music length
    intro_duration = min(intro_duration_ms, music_duration)
    intro = music[:intro_duration]
    
    # Extract outro segment (from end of music for variety, or beginning if short)
    outro_duration = min(outro_duration_ms, music_duration)
    if music_duration > outro_duration_ms * 2:
        # Use end of music for outro (different from intro)
        outro = music[-outro_duration:]
    else:
        # Music is short, use beginning
        outro = music[:outro_duration]
    
    # Apply fades
    # Ensure fade duration doesn't exceed segment duration
    intro_fade = min(intro_fade_ms, intro_duration)
    outro_fade = min(outro_fade_ms, outro_duration)
    
    # Intro: music plays then fades OUT (reduces) before voice starts
    intro = intro.fade_out(intro_fade)
    
    # Outro: music fades IN (comes in gradually) then fades OUT at the end
    # Use half the fade time for fade-in, full time for fade-out
    outro_fade_in = min(outro_fade // 2, outro_duration // 2)
    outro = outro.fade_in(outro_fade_in).fade_out(outro_fade)
    
    # Apply gentle fades to voice to prevent clicks/pops
    voice_fade_ms = 100  # 100ms fade for smooth transitions
    voice = voice.fade_in(voice_fade_ms).fade_out(voice_fade_ms)
    
    # Concatenate: intro + voice + outro
    final = intro + voice + outro
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format
    output_format = output_file.suffix.lstrip(".").lower()
    if output_format not in ("mp3", "wav", "ogg", "flac"):
        output_format = "mp3"
    
    # Export
    final.export(str(output_file), format=output_format)
    
    return str(output_file)


def get_default_music_path() -> Optional[str]:
    """Get the path to the default background music file.
    
    Returns:
        Path to background_music.mp3 if it exists, None otherwise
    """
    # Look for music file in the same directory as this module
    module_dir = Path(__file__).parent
    default_music = module_dir / "background_music.mp3"
    
    if default_music.exists():
        return str(default_music)
    
    return None

