"""
Qwen TTS API Engine for podcast generation.

Calls a remote Qwen TTS gateway API (voice cloning) to synthesize speech.
Messages are grouped by speaker and sent in batch API calls to avoid Lambda
concurrency limits. The server processes all texts in a single GPU call.

Requires CLOUD_GATEWAY_API_KEY environment variable (or api_key parameter).
"""

import io
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..base_engine import BaseTTSEngine
from ..models import Conversation, Message


# Remote Lambda Function URL endpoint
QWEN_TTS_API_URL = "https://noyr5xagtj2ofkqq3eef7knha40lsogm.lambda-url.eu-west-1.on.aws"

# Default voice profile names per language (pre-computed voice clones)
QWEN_TTS_API_VOICE_MAP: dict[str, dict[str, str]] = {
    "es": {"host": "adrian", "guest": "teresa"},
    "en": {"host": "adrian", "guest": "teresa"},
    "fr": {"host": "adrian", "guest": "teresa"},
    "de": {"host": "adrian", "guest": "teresa"},
    "it": {"host": "adrian", "guest": "teresa"},
    "pt": {"host": "adrian", "guest": "teresa"},
    "zh": {"host": "adrian", "guest": "teresa"},
    "ja": {"host": "adrian", "guest": "teresa"},
    "ko": {"host": "adrian", "guest": "teresa"},
}

# Map internal language codes to the API's language string parameter
QWEN_TTS_API_LANGUAGES: dict[str, str] = {
    "es": "Spanish",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

DEFAULT_CONCURRENCY = 20  # max texts per batch API call (one Lambda invocation)
MAX_BATCH_SIZE = 20       # hard cap matching server-side MAX_BATCH_SIZE
MAX_RETRIES = 6
RETRY_BACKOFF_BASE = 5.0  # seconds — gives GPU queue time to drain on timeout


class QwenTTSApiEngine(BaseTTSEngine):
    """Text-to-Speech engine using the remote Qwen TTS gateway API.

    Synthesizes speech by calling a voice-cloning API endpoint.
    All messages in a conversation are sent to the API in parallel
    (up to ``concurrency`` requests at once) to minimize total latency.

    Requires the CLOUD_GATEWAY_API_KEY environment variable or the
    ``api_key`` constructor parameter.
    """

    def __init__(
        self,
        language: str = "es",
        speaker_map: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        concurrency: int = DEFAULT_CONCURRENCY,
    ):
        """Initialize the Qwen TTS API engine.

        Args:
            language: Language code (es, en, fr, de, it, pt, zh, ja, ko)
            speaker_map: Mapping of role names to API profile_name values
            api_key: GCloud gateway API key (falls back to CLOUD_GATEWAY_API_KEY env var)
            api_url: Override the default API endpoint URL
            concurrency: Maximum number of parallel API requests (default: 10)
        """
        super().__init__(language=language, speaker_map=speaker_map)

        self.api_key = api_key or os.environ.get("CLOUD_GATEWAY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Qwen TTS API key required. Set CLOUD_GATEWAY_API_KEY env var "
                "or pass api_key parameter."
            )

        self.api_url = api_url or QWEN_TTS_API_URL
        self.concurrency = concurrency
        self.language_str = QWEN_TTS_API_LANGUAGES.get(language, "Spanish")

        if not self.speaker_map:
            self.speaker_map = QWEN_TTS_API_VOICE_MAP.get(
                language, QWEN_TTS_API_VOICE_MAP["es"]
            ).copy()

    def get_speaker_for_role(self, role: str) -> str:
        """Get the API profile_name for a given role.

        Args:
            role: Speaker role name (e.g., "host", "guest")

        Returns:
            API profile_name string (e.g., "adrian", "teresa")
        """
        if role in self.speaker_map:
            return self.speaker_map[role]

        fallback = QWEN_TTS_API_VOICE_MAP.get(self.language, QWEN_TTS_API_VOICE_MAP["es"])
        voice = fallback.get("host", "adrian")
        self.speaker_map[role] = voice
        return voice

    def synthesize_message(
        self,
        message: Message,
        output_path: str,
        language_code: Optional[str] = None,
    ) -> str:
        """Synthesize a single message to a WAV file via the gateway API.

        Retries up to MAX_RETRIES times with exponential backoff on failure.

        Args:
            message: Message to synthesize
            output_path: Path to save the WAV audio file
            language_code: Optional override for the API language string

        Returns:
            Path to the generated audio file
        """
        import requests

        profile_name = self.get_speaker_for_role(message.role)
        language_str = (
            QWEN_TTS_API_LANGUAGES.get(language_code, self.language_str)
            if language_code
            else self.language_str
        )

        payload = {
            "text": message.content,
            "language": language_str,
        }

        # The new API supports either profile_name (cloned) or speaker (built-in)
        if hasattr(message, "speaker") and message.speaker:
            payload["speaker"] = message.speaker
        else:
            payload["profile_name"] = profile_name
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.api_url.rstrip('/')}/generate-clone",
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=300,  # 300s: covers cold-start + full GPU batch
                )
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                return output_path

            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    # Detect server-side timeout / overload (500, 502, 503)
                    # and apply a longer wait to let the GPU queue drain.
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status in (500, 502, 503):
                        # Server timeout — wait longer so the queue drains
                        wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    elif status == 429:
                        # Lambda concurrency cap hit — back off hard
                        wait = RETRY_BACKOFF_BASE * (3 ** attempt)
                    else:
                        wait = RETRY_BACKOFF_BASE ** attempt
                    wait = min(wait, 120)  # cap at 2 minutes
                    print(
                        f"   ⚠️ API error (attempt {attempt + 1}/{MAX_RETRIES}, "
                        f"HTTP {status or 'N/A'}): {exc} — retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to synthesize message after {MAX_RETRIES} attempts: {last_exc}"
        )

    def _send_batch(self, texts: list[str], profile_name: str, lang: str) -> list[bytes]:
        """Send a batch of texts for one speaker and return WAV bytes in order."""
        import requests

        payload = {
            "text": texts if len(texts) > 1 else texts[0],
            "language": lang,
            "profile_name": profile_name,
        }
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}

        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.api_url.rstrip('/')}/generate-clone",
                    json=payload,
                    headers=headers,
                    timeout=300,
                )
                response.raise_for_status()

                if len(texts) == 1:
                    return [response.content]

                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    # Sort numerically: output_1.wav < output_10.wav
                    names = sorted(
                        zf.namelist(),
                        key=lambda n: int(n.split("_")[1].split(".")[0]),
                    )
                    return [zf.read(name) for name in names]

            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status in (500, 502, 503):
                        wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    elif status == 429:
                        wait = RETRY_BACKOFF_BASE * (3 ** attempt)
                    else:
                        wait = RETRY_BACKOFF_BASE ** attempt
                    
                    wait = min(wait, 120)
                    print(
                        f"   ⚠️ Batch API error (attempt {attempt + 1}/{MAX_RETRIES}, "
                        f"HTTP {status or 'N/A'}, {len(texts)} texts): {exc} — retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)

        raise RuntimeError(f"Batch failed after {MAX_RETRIES} attempts: {last_exc}")

    def synthesize_conversation(
        self,
        conversation: Conversation,
        output_path: str,
        language_code: Optional[str] = None,
        silence_duration_ms: int = 500,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Synthesize a full conversation to a single audio file.

        Groups messages by speaker and sends each group as ONE batch API call,
        bypassing Lambda concurrency limits. The server processes all texts in a
        single GPU call and returns a ZIP of WAV files.

        Args:
            conversation: Conversation to synthesize
            output_path: Path to save the final audio file
            language_code: Optional language override
            silence_duration_ms: Duration of silence between messages (ms)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Path to the generated audio file
        """
        if len(conversation) == 0:
            raise ValueError("Conversation cannot be empty")

        import wave
        import subprocess
        total = len(conversation)
        self.segment_durations_ms = [None] * total
        lang = QWEN_TTS_API_LANGUAGES.get(language_code, self.language_str) if language_code else self.language_str

        with tempfile.TemporaryDirectory(prefix="qwen_tts_api_") as temp_dir:
            temp_paths = [
                os.path.join(temp_dir, f"segment_{idx:04d}.wav")
                for idx in range(total)
            ]

            # Group message indices by speaker profile so each batch uses one voice.
            groups: dict[str, list[int]] = defaultdict(list)
            for idx, msg in enumerate(conversation.messages):
                groups[self.get_speaker_for_role(msg.role)].append(idx)

            completed_count = 0
            batch_size = min(self.concurrency, MAX_BATCH_SIZE)

            for profile_name, indices in groups.items():
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    texts = [conversation.messages[i].content for i in batch_indices]
                    wav_bytes_list = self._send_batch(texts, profile_name, lang)
                    for i, wav_bytes in enumerate(wav_bytes_list):
                        with open(temp_paths[batch_indices[i]], "wb") as f:
                            f.write(wav_bytes)
                    completed_count += len(batch_indices)
                    if progress_callback:
                        progress_callback(completed_count, total)

            # Read first WAV to get audio params
            with wave.open(temp_paths[0], "rb") as w:
                params = w.getparams()
                n_channels = params.nchannels
                sampwidth = params.sampwidth
                framerate = params.framerate

            # Calculate duration of each segment and build silence frames
            silence_frames = int(framerate * silence_duration_ms / 1000)
            silence_data = b"\x00" * silence_frames * n_channels * sampwidth

            # Concatenated WAV output path
            combined_wav = os.path.join(temp_dir, "combined.wav")

            with wave.open(combined_wav, "wb") as out_wav:
                out_wav.setparams(params)
                for idx in range(total):
                    with wave.open(temp_paths[idx], "rb") as seg:
                        frames = seg.readframes(seg.getnframes())
                        duration_ms = int(seg.getnframes() / framerate * 1000)
                        self.segment_durations_ms[idx] = duration_ms
                    out_wav.writeframes(frames)
                    if idx < total - 1:
                        out_wav.writeframes(silence_data)

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            output_format = out.suffix.lstrip(".").lower()
            if output_format not in ("mp3", "wav", "ogg", "flac"):
                output_format = "mp3"

            if output_format == "wav":
                import shutil
                shutil.copy2(combined_wav, str(out))
            else:
                # Convert using ffmpeg (via imageio-ffmpeg bundled binary)
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError:
                    ffmpeg_exe = "ffmpeg"

                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", combined_wav,
                    "-acodec", "libmp3lame" if output_format == "mp3" else output_format,
                    "-q:a", "2",
                    str(out),
                ]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg conversion failed: {result.stderr.decode()}"
                    )

        return str(output_path)

    @classmethod
    def list_available_voices(cls, language: str = "es", api_key: Optional[str] = None) -> list[str]:
        """List the available voice profiles from the API.

        Args:
            language: Language code (unused)
            api_key: Optional API key override

        Returns:
            List of profile_name strings
        """
        import requests

        key = api_key or os.environ.get("CLOUD_GATEWAY_API_KEY")
        try:
            response = requests.get(
                f"{QWEN_TTS_API_URL.rstrip('/')}/profiles",
                headers={"x-api-key": key},
                timeout=60,  # Increased to 60 seconds
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"   ⚠️ Failed to fetch dynamic profiles: {e} — falling back to defaults")
            return ["adrian", "teresa", "xuban-berasategui"]


def _get_ffmpeg_exe() -> str:
    """Return path to ffmpeg binary (bundled or system)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _add_background_music_ffmpeg(
    voice_path: str,
    music_path: str,
    output_path: str,
    intro_duration_ms: int = 5000,
    outro_duration_ms: int = 5000,
    intro_fade_ms: int = 3000,
    outro_fade_ms: int = 3000,
    music_volume_db: int = -6,
) -> str:
    """Mix voice audio with intro/outro background music using ffmpeg.

    Produces: [intro music fading out] + [voice] + [outro music fading in/out]

    Args:
        voice_path: Path to the synthesized voice audio
        music_path: Path to the background music file
        output_path: Path to save the final mixed output
        intro_duration_ms: Duration of intro music segment in ms
        outro_duration_ms: Duration of outro music segment in ms
        intro_fade_ms: Fade-out duration for intro in ms
        outro_fade_ms: Fade-out duration for outro in ms
        music_volume_db: Volume adjustment for music (negative = quieter)

    Returns:
        Path to the output file
    """
    import subprocess

    ffmpeg = _get_ffmpeg_exe()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    intro_s = intro_duration_ms / 1000
    outro_s = outro_duration_ms / 1000
    intro_fade_s = intro_fade_ms / 1000
    outro_fade_s = outro_fade_ms / 1000
    music_vol = 10 ** (music_volume_db / 20)  # dB to linear

    # ffmpeg filter graph:
    # 1. Trim intro from music, apply fade-out
    # 2. Trim outro from music (from end), apply fade-in + fade-out
    # 3. Concatenate: intro + voice + outro
    filter_complex = (
        f"[1:a]volume={music_vol:.4f},asplit=2[music_intro][music_outro];"
        f"[music_intro]atrim=0:{intro_s},asetpts=PTS-STARTPTS,"
        f"afade=t=out:st={max(0, intro_s - intro_fade_s):.3f}:d={intro_fade_s:.3f}[intro];"
        f"[music_outro]atrim=0:{outro_s},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={outro_fade_s / 2:.3f},"
        f"afade=t=out:st={max(0, outro_s - outro_fade_s):.3f}:d={outro_fade_s:.3f}[outro];"
        f"[intro][0:a][outro]concat=n=3:v=0:a=1[out]"
    )

    cmd = [
        ffmpeg, "-y",
        "-i", voice_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg background music mixing failed: {result.stderr.decode()}"
        )
    return output_path


def generate_podcast_qwen_tts_api(
    conversation: list[dict],
    output_path: str,
    language: str = "es",
    speaker_map: Optional[dict[str, str]] = None,
    silence_duration_ms: int = 500,
    progress_callback: Optional[callable] = None,
    # API-specific
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
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
) -> dict:
    """Generate a podcast audio file from a conversation using the Qwen TTS API.

    Messages are synthesized in parallel (up to ``concurrency`` at once) to
    reduce total generation time despite slow individual API calls.

    Args:
        conversation: List of dicts with 'role' and 'content' keys
        output_path: Path to save the output audio file
        language: Language code (es, en, fr, de, it, pt, zh, ja, ko)
        speaker_map: Optional mapping of roles to API profile_name values
        silence_duration_ms: Silence between messages in milliseconds
        progress_callback: Optional callback(current, total)
        api_key: Gateway API key (falls back to CLOUD_GATEWAY_API_KEY env var)
        api_url: Override the default gateway endpoint URL
        concurrency: Max parallel API requests (default: 10)
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
        Dict with 'path' and 'segment_durations_ms' keys
    """
    from ..models import Conversation as ConvModel

    conv = ConvModel.from_dicts(conversation)

    engine = QwenTTSApiEngine(
        language=language,
        speaker_map=speaker_map,
        api_key=api_key,
        api_url=api_url,
        concurrency=concurrency,
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

            _add_background_music_ffmpeg(
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
        try:
            from ..audio_utils import add_metadata
            add_metadata(
                file_path=output_path,
                title=title,
                artist=artist,
                album=album,
                track_number=track_number,
            )
        except ImportError:
            pass  # mutagen not available, skip metadata

    return {
        "path": output_path,
        "segment_durations_ms": engine.segment_durations_ms,
    }
