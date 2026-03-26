#!/usr/bin/env python3
"""
Podcast TTS Tool - Command Line Interface

Supports both Coqui TTS (offline) and Edge TTS.

Usage:
    python -m tools.podcast                           # Run with Edge TTS (default)
    python -m tools.podcast --engine coqui            # Use Coqui TTS (offline)
    python -m tools.podcast --language es             # Spanish
    python -m tools.podcast --output my_podcast.mp3   # Custom output path
    python -m tools.podcast --list-voices en          # List available voices
    python -m tools.podcast --list-engines            # List available TTS engines
    python -m tools.podcast --music                   # Add background music (intro/outro)
    python -m tools.podcast --title "Episode 1"       # Set metadata
"""

import argparse
import sys
from pathlib import Path


# Sample conversation for testing
SAMPLE_CONVERSATION_EN = [
    {
        "role": "host",
        "content": "Welcome to our podcast about artificial intelligence! I'm your host, and today we have a special guest joining us."
    },
    {
        "role": "guest", 
        "content": "Thank you for having me! I'm excited to discuss the latest developments in AI technology."
    },
    {
        "role": "host",
        "content": "Let's start with the basics. Can you explain what machine learning is in simple terms?"
    },
    {
        "role": "guest",
        "content": "Of course! Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed. Instead of writing rules, we show the computer examples and it learns patterns."
    },
    {
        "role": "host",
        "content": "That's a great explanation. And how does this relate to deep learning?"
    },
    {
        "role": "guest",
        "content": "Deep learning is a subset of machine learning that uses neural networks with many layers. These deep networks can learn very complex patterns and are behind many recent breakthroughs like image recognition and natural language processing."
    },
]

SAMPLE_CONVERSATION_ES = [
    {
        "role": "host",
        "content": "¡Bienvenidos a nuestro podcast sobre inteligencia artificial! Soy su anfitrión y hoy tenemos un invitado especial."
    },
    {
        "role": "guest",
        "content": "¡Gracias por invitarme! Estoy emocionado de hablar sobre los últimos avances en tecnología de IA."
    },
    {
        "role": "host",
        "content": "Comencemos con lo básico. ¿Puedes explicar qué es el aprendizaje automático en términos simples?"
    },
    {
        "role": "guest",
        "content": "¡Por supuesto! El aprendizaje automático es un tipo de inteligencia artificial que permite a las computadoras aprender de los datos sin ser programadas explícitamente."
    },
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a podcast using TTS with multiple speakers. Supports Edge TTS and Coqui TTS (offline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tools.podcast                           # Edge TTS (default)
    python -m tools.podcast --engine coqui            # Coqui TTS (offline)
    python -m tools.podcast --language es             # Spanish
    python -m tools.podcast --list-voices en          # List Edge voices
    python -m tools.podcast --list-engines            # Show engine info
        """
    )
    
    # Engine selection
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default="qwen_tts_api",
        choices=["edge", "coqui", "chatterbox", "elevenlabs", "openai_tts", "qwen_tts", "mlx_tts", "qwen_tts_api"],
        help="TTS engine to use. 'qwen_tts_api' (default) uses the remote voice-cloning API, 'edge' is fast and online, 'coqui' works offline."
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code (default: en). For Edge: en, es, fr, de, it, pt, zh, ja, ko. For Coqui: en, es, multilingual."
    )
    
    parser.add_argument(
        "--language-code",
        type=str,
        default=None,
        help="Language code for Coqui multilingual model (e.g., 'es', 'en', 'fr'). Only used with Coqui."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="podcast_sample.mp3",
        help="Output audio file path (default: podcast_sample.mp3)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for Coqui/Chatterbox TTS (default: cpu). Ignored for Edge TTS."
    )
    
    parser.add_argument(
        "--silence", "-s",
        type=int,
        default=500,
        help="Silence duration between messages in ms (default: 500)"
    )
    
    parser.add_argument(
        "--list-voices",
        type=str,
        metavar="LANGUAGE",
        help="List available voices for a language and exit"
    )
    
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available TTS engines and exit"
    )
    
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available languages for selected engine and exit"
    )
    
    parser.add_argument(
        "--host-voice",
        type=str,
        default=None,
        help="Voice/speaker ID for the host role"
    )
    
    parser.add_argument(
        "--guest-voice",
        type=str,
        default=None,
        help="Voice/speaker ID for the guest role"
    )
    
    # Metadata arguments
    metadata_group = parser.add_argument_group("Metadata", "ID3 metadata for the output MP3")
    metadata_group.add_argument(
        "--title",
        type=str,
        default="Module",
        help="Track title (default: Module)"
    )
    metadata_group.add_argument(
        "--artist",
        type=str,
        default="Adinhub",
        help="Artist name (default: Adinhub)"
    )
    metadata_group.add_argument(
        "--album",
        type=str,
        default="Course",
        help="Album name (default: Course)"
    )
    metadata_group.add_argument(
        "--track-number",
        type=int,
        default=None,
        help="Track number (optional)"
    )
    
    # Background music arguments
    music_group = parser.add_argument_group("Background Music", "Intro/outro music settings")
    music_group.add_argument(
        "--music",
        type=str,
        nargs="?",
        const="default",
        default=None,
        metavar="PATH",
        help="Path to background music file. Use --music without a path to use default music."
    )
    music_group.add_argument(
        "--intro-duration",
        type=int,
        default=5000,
        metavar="MS",
        help="Intro music duration in ms (default: 5000)"
    )
    music_group.add_argument(
        "--outro-duration",
        type=int,
        default=5000,
        metavar="MS",
        help="Outro music duration in ms (default: 5000)"
    )
    music_group.add_argument(
        "--intro-fade",
        type=int,
        default=3000,
        metavar="MS",
        help="Intro fade-in duration in ms (default: 3000)"
    )
    music_group.add_argument(
        "--outro-fade",
        type=int,
        default=3000,
        metavar="MS",
        help="Outro fade-out duration in ms (default: 3000)"
    )
    music_group.add_argument(
        "--music-volume",
        type=int,
        default=-6,
        metavar="DB",
        help="Music volume adjustment in dB (default: -6)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle list engines command first (doesn't need imports)
    if args.list_engines:
        from .factory import list_engines
        print("Available TTS Engines:\n")
        for engine_info in list_engines():
            print(f"  {engine_info['engine']}: {engine_info['name']}")
            print(f"    {engine_info['description']}")
            print(f"    Requires Internet: {'Yes' if engine_info['requires_internet'] else 'No'}")
            print(f"    Languages: {', '.join(engine_info['languages'])}")
            print()
        return 0
    
    # Handle list voices command
    if args.list_voices:
        if args.engine == "edge":
            from .edge import EdgeTTSEngine, EDGE_VOICE_MAP
            voices = EdgeTTSEngine.list_available_voices(args.list_voices)
            default_map = EDGE_VOICE_MAP.get(args.list_voices, {})
            print(f"Available Edge TTS voices for '{args.list_voices}':")
            for voice in voices:
                role = ""
                if default_map.get("host") == voice:
                    role = " (default host)"
                elif default_map.get("guest") == voice:
                    role = " (default guest)"
                print(f"  - {voice}{role}")
        elif args.engine == "coqui":
            from .coqui import list_speakers
            from .models import get_language_config
            try:
                speakers = list_speakers(args.list_voices)
                config = get_language_config(args.list_voices)
                print(f"Available Coqui speakers for '{args.list_voices}' ({config.model_name}):")
                for speaker in speakers:
                    print(f"  - {speaker}")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        else:
            from .chatterbox import ChatterboxEngine
            try:
                voices = ChatterboxEngine.list_available_voices(args.list_voices)
                print(f"Available Chatterbox voices for '{args.list_voices}':")
                for voice in voices:
                    print(f"  - {voice}")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        return 0
    
    # Handle list languages
    if args.list_languages:
        from .factory import get_engine_info
        info = get_engine_info(args.engine)
        print(f"Available languages for {info['name']}:")
        for lang in info['languages']:
            print(f"  - {lang}")
        return 0
    
    # Import generation functions
    from .edge import generate_podcast_edge
    from .coqui import generate_podcast
    from .chatterbox import generate_podcast_chatterbox
    from .audio_utils import get_default_music_path
    
    # Select sample conversation based on language
    is_spanish = args.language == "es" or (args.language == "multilingual" and args.language_code == "es")
    conversation = SAMPLE_CONVERSATION_ES if is_spanish else SAMPLE_CONVERSATION_EN
    
    # Build voice/speaker map if custom voices specified
    speaker_map = None
    if args.host_voice or args.guest_voice:
        speaker_map = {}
        if args.host_voice:
            speaker_map["host"] = args.host_voice
        if args.guest_voice:
            speaker_map["guest"] = args.guest_voice
    
    # Resolve music path
    music_path = None
    if args.music:
        if args.music == "default":
            music_path = get_default_music_path()
            if not music_path:
                print("⚠️  No default background music found. Skipping music.", file=sys.stderr)
        else:
            music_path = args.music
            if not Path(music_path).exists():
                print(f"❌ Music file not found: {music_path}", file=sys.stderr)
                return 1
    
    # Print configuration
    engine_names = {"edge": "Edge TTS", "coqui": "Coqui TTS", "chatterbox": "Chatterbox TTS"}
    engine_name = engine_names.get(args.engine, args.engine)
    print(f"🎙️  Podcast TTS Generator")
    print(f"   Engine: {engine_name}")
    print(f"   Language: {args.language}")
    if args.engine == "coqui" and args.language_code:
        print(f"   Language code: {args.language_code}")
    print(f"   Output: {args.output}")
    if args.engine in ("coqui", "chatterbox"):
        print(f"   Device: {args.device}")
    print(f"   Messages: {len(conversation)}")
    print(f"   Metadata: {args.title} by {args.artist} ({args.album})")
    if music_path:
        print(f"   Music: {music_path}")
        print(f"   Intro: {args.intro_duration}ms (fade: {args.intro_fade}ms)")
        print(f"   Outro: {args.outro_duration}ms (fade: {args.outro_fade}ms)")
    print()
    
    def progress_callback(current, total):
        print(f"   Synthesizing message {current}/{total}...", flush=True)
    
    try:
        if args.engine == "edge":
            output_path = generate_podcast_edge(
                conversation=conversation,
                output_path=args.output,
                language=args.language,
                speaker_map=speaker_map,
                silence_duration_ms=args.silence,
                progress_callback=progress_callback,
                # Metadata
                title=args.title,
                artist=args.artist,
                album=args.album,
                track_number=args.track_number,
                # Background music
                music_path=music_path,
                intro_duration_ms=args.intro_duration,
                outro_duration_ms=args.outro_duration,
                intro_fade_ms=args.intro_fade,
                outro_fade_ms=args.outro_fade,
                music_volume_db=args.music_volume,
            )
        elif args.engine == "chatterbox":
            output_path = generate_podcast_chatterbox(
                conversation=conversation,
                output_path=args.output,
                language=args.language,
                speaker_map=speaker_map,
                silence_duration_ms=args.silence,
                device=args.device,
                progress_callback=progress_callback,
                # Metadata
                title=args.title,
                artist=args.artist,
                album=args.album,
                track_number=args.track_number,
                # Background music
                music_path=music_path,
                intro_duration_ms=args.intro_duration,
                outro_duration_ms=args.outro_duration,
                intro_fade_ms=args.intro_fade,
                outro_fade_ms=args.outro_fade,
                music_volume_db=args.music_volume,
            )
        elif args.engine == "mlx_tts":
            from .mlx_tts.client import generate_podcast_mlx_tts
            output_path = generate_podcast_mlx_tts(
                conversation=conversation,
                output_path=args.output,
                language=args.language,
                speaker_map=speaker_map,
                silence_duration_ms=args.silence,
                progress_callback=progress_callback,
                title=args.title,
                artist=args.artist,
                album=args.album,
                track_number=args.track_number,
                music_path=music_path,
                intro_duration_ms=args.intro_duration,
                outro_duration_ms=args.outro_duration,
                intro_fade_ms=args.intro_fade,
                outro_fade_ms=args.outro_fade,
                music_volume_db=args.music_volume,
            )
        elif args.engine == "qwen_tts":
            from .qwen_tts.client import generate_podcast_qwen_tts
            output_path = generate_podcast_qwen_tts(
                conversation=conversation,
                output_path=args.output,
                language=args.language,
                speaker_map=speaker_map,
                silence_duration_ms=args.silence,
                progress_callback=progress_callback,
                title=args.title,
                artist=args.artist,
                album=args.album,
                track_number=args.track_number,
                music_path=music_path,
                intro_duration_ms=args.intro_duration,
                outro_duration_ms=args.outro_duration,
                intro_fade_ms=args.intro_fade,
                outro_fade_ms=args.outro_fade,
                music_volume_db=args.music_volume,
            )
        else:
            output_path = generate_podcast(
                conversation=conversation,
                output_path=args.output,
                language=args.language,
                speaker_map=speaker_map,
                language_code=args.language_code,
                silence_duration_ms=args.silence,
                device=args.device,
                progress_callback=progress_callback,
                # Metadata
                title=args.title,
                artist=args.artist,
                album=args.album,
                track_number=args.track_number,
                # Background music
                music_path=music_path,
                intro_duration_ms=args.intro_duration,
                outro_duration_ms=args.outro_duration,
                intro_fade_ms=args.intro_fade,
                outro_fade_ms=args.outro_fade,
                music_volume_db=args.music_volume,
            )
        
        print()
        print(f"✅ Podcast generated successfully: {output_path}")
        
        # Show file size
        size = Path(output_path).stat().st_size
        if size > 1024 * 1024:
            print(f"   Size: {size / (1024 * 1024):.2f} MB")
        else:
            print(f"   Size: {size / 1024:.2f} KB")
        
        # Show metadata info
        print(f"   Title: {args.title}")
        print(f"   Artist: {args.artist}")
        print(f"   Album: {args.album}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
