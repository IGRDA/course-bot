#!/usr/bin/env python3
"""Quick smoke test for TTS engines.

Usage:
    # Test Qwen3-TTS (CustomVoice mode, auto-detects device)
    python scripts/test_tts_engines.py --engine qwen_tts

    # Test Qwen3-TTS voice clone using an existing audio file as reference
    python scripts/test_tts_engines.py --engine qwen_tts --task-type voice_clone \
        --ref-audio output/some_audio.mp3

    # Test Edge TTS (fast, cloud-based, no GPU needed)
    python scripts/test_tts_engines.py --engine edge

    # Force a specific device
    python scripts/test_tts_engines.py --engine qwen_tts --device cpu

    # List all available engines
    python scripts/test_tts_engines.py --list
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


SAMPLE_CONVERSATION = [
    {
        "role": "host",
        "content": (
            "¡Bienvenidos a otro episodio de Cuidados del Bebé! Hoy vamos a "
            "explorar un tema fascinante: ¿cómo funciona el cuerpo de un recién "
            "nacido? Desde su anatomía hasta los reflejos que los protegen y las "
            "diferencias entre bebés prematuros, a término y postérmino. Para "
            "hablar de esto, tenemos con nosotros a la pediatra especializada en "
            "neonatología, Doctora Laura Mendoza. Laura, gracias por acompañarnos."
        ),
    },
    {
        "role": "guest",
        "content": (
            "¡Muchas gracias por la invitación! Es un placer compartir "
            "información tan importante para los padres y cuidadores. Los recién "
            "nacidos son increíbles, y entender su cuerpo nos ayuda a brindarles "
            "los mejores cuidados."
        ),
    },
]


def ensure_wav(audio_path: str, output_dir: str = "output") -> str:
    """Convert an audio file to WAV if it isn't already.

    Supports mp3, ogg, flac, m4a via pydub/ffmpeg.

    Args:
        audio_path: Path to the source audio file.
        output_dir: Directory for the converted WAV (if conversion is needed).

    Returns:
        Path to a WAV file (original path if already WAV, otherwise a new file).
    """
    if audio_path.lower().endswith(".wav"):
        return audio_path

    from pydub import AudioSegment

    ext = os.path.splitext(audio_path)[1].lstrip(".").lower()
    print(f"  Converting {ext} -> wav ...")
    segment = AudioSegment.from_file(audio_path, format=ext)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    wav_path = os.path.join(output_dir, f"{base}.wav")
    segment.export(wav_path, format="wav")
    print(f"  Saved converted WAV: {wav_path}")
    return wav_path


def list_engines():
    from tools.podcast.factory import list_engines as _list
    print("Available TTS engines:\n")
    for info in _list():
        print(f"  {info['engine']:15s}  {info['name']}")
        print(f"  {'':15s}  {info['description']}")
        print(f"  {'':15s}  Internet: {'yes' if info['requires_internet'] else 'no'}  "
              f"Languages: {', '.join(info['languages'][:8])}{'...' if len(info['languages']) > 8 else ''}")
        print()


def run_test(
    engine_name: str,
    language: str,
    task_type: str | None,
    speaker_map: dict[str, str] | None,
    device: str | None,
    ref_audio: str | None,
    output_path: str | None = None,
):
    from tools.podcast.factory import create_tts_engine

    if not output_path:
        output_path = f"output/test_tts_{engine_name}.wav"
    os.makedirs(os.path.dirname(output_path) or "output", exist_ok=True)

    # Build speaker_map for voice_clone from a single reference audio
    if ref_audio and task_type == "voice_clone" and speaker_map is None:
        wav_ref = ensure_wav(ref_audio)
        speaker_map = {"host": wav_ref, "guest": wav_ref}
        print(f"  Using reference audio for both roles: {wav_ref}")

    # Convert any non-WAV speaker_map paths (voice_clone with per-role audio)
    if speaker_map and task_type == "voice_clone":
        for role, path in speaker_map.items():
            if os.path.isfile(path) and not path.lower().endswith(".wav"):
                speaker_map[role] = ensure_wav(path)
        for role, path in speaker_map.items():
            print(f"  {role}: {path}")

    kwargs: dict = {}
    if task_type:
        kwargs["task_type"] = task_type
    if device:
        kwargs["device"] = device

    print(f"Creating engine: {engine_name} (language={language})")
    engine = create_tts_engine(
        engine=engine_name,
        language=language,
        speaker_map=speaker_map,
        **kwargs,
    )

    from tools.podcast.models import Conversation
    conv = Conversation.from_dicts(SAMPLE_CONVERSATION)

    print(f"Synthesizing {len(conv)} messages...")
    t0 = time.time()

    def progress(current, total):
        print(f"  [{current}/{total}] done")

    engine.synthesize_conversation(
        conversation=conv,
        output_path=output_path,
        silence_duration_ms=400,
        progress_callback=progress,
    )

    elapsed = time.time() - t0
    size_kb = os.path.getsize(output_path) / 1024

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {output_path} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Test TTS engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="Engine name (e.g. qwen_tts, edge, openai_tts)",
    )
    parser.add_argument("--language", default="es", help="Language code (default: es)")
    parser.add_argument("--task-type", help="For qwen_tts: custom_voice or voice_clone")
    parser.add_argument(
        "--speaker-map",
        type=str,
        help='JSON string mapping roles to speakers, e.g. \'{"host": "Ryan"}\'',
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        help="Reference audio file for voice_clone mode (mp3/wav/ogg). "
             "Used for both host and guest roles unless --speaker-map is given.",
    )
    parser.add_argument("--output", type=str, help="Output file path (default: output/test_tts_{engine}.wav)")
    parser.add_argument("--device", help="PyTorch device (e.g. cuda:0, cpu, mps)")
    parser.add_argument("--list", action="store_true", help="List available engines")
    args = parser.parse_args()

    if args.list:
        list_engines()
        return

    if not args.engine:
        parser.error("--engine is required (or use --list)")

    speaker_map = json.loads(args.speaker_map) if args.speaker_map else None

    run_test(
        engine_name=args.engine,
        language=args.language,
        task_type=args.task_type,
        speaker_map=speaker_map,
        device=args.device,
        ref_audio=args.ref_audio,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
