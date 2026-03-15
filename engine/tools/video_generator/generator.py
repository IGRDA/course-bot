#!/usr/bin/env python3
"""
Course Video Generator

Generates a video from a course by:
1. Navigating through slides using Playwright
2. Extracting text content from each slide
3. Generating audio using edge-tts (Microsoft Edge TTS)
4. Creating video with screenshots + audio using FFmpeg
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page


# Default configuration
DEFAULT_URL = "https://www.demo.adinhub.com/"
DEFAULT_OUTPUT = "course_video.mp4"
DEFAULT_VOICE = "en-US-JennyNeural"  # Spanish male voice
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720

# Selectors
NEXT_BUTTON_SELECTOR = "button.module-app-footer-next"
NEXT_ICON_SELECTOR = "i.mdi-arrow-right-thick"
START_BUTTON_SELECTOR = "button:has-text('Comenzar')"
MAIN_CONTENT_SELECTOR = "main"
PROGRESS_SELECTOR = ".module-app-footer-progress-position"

# Available voices (run `edge-tts --list-voices` to see all)
VOICES = {
    "es-male": "es-ES-AlvaroNeural",
    "es-female": "es-ES-ElviraNeural",
    "en-male": "en-US-GuyNeural",
    "en-female": "en-US-JennyNeural",
    "es-mx-male": "es-MX-JorgeNeural",
    "es-mx-female": "es-MX-DaliaNeural",
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a video from a course using TTS and screen capture"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"Course URL (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting page/slide number (default: 1)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending page/slide number (default: last slide)"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Number of pages to process from start (alternative to --end)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output video filename (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="es-male",
        choices=list(VOICES.keys()),
        help="TTS voice to use (default: es-male)"
    )
    parser.add_argument(
        "--custom-voice",
        type=str,
        default=None,
        help="Custom edge-tts voice name (overrides --voice)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: False)"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for temp files (default: auto-created temp dir)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Keep temporary files after completion"
    )
    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
        help="TTS speech rate adjustment (e.g., '+10%%', '-20%%')"
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Path to local module JSON file to load (e.g., output/Quantum_Theory_.../module_0.json)"
    )
    return parser.parse_args()


def get_total_slides(page: Page) -> int:
    """Extract total number of slides from the progress indicator."""
    try:
        # Try to get from .total-slides-number first
        total_elem = page.locator(".total-slides-number")
        if total_elem.count() > 0:
            return int(total_elem.inner_text().strip())
        
        progress_text = page.locator(PROGRESS_SELECTOR).inner_text()
        if " de " in progress_text:
            total = int(progress_text.split(" de ")[-1].strip())
        elif " of " in progress_text:
            total = int(progress_text.split(" of ")[-1].strip())
        else:
            total = 80
        return total
    except Exception:
        return 80


def get_current_slide(page: Page) -> int:
    """Extract current slide number from the progress indicator."""
    try:
        current_elem = page.locator(".current-slide-number")
        if current_elem.count() > 0:
            return int(current_elem.inner_text().strip())
        
        progress_text = page.locator(PROGRESS_SELECTOR).inner_text()
        if " de " in progress_text:
            current = int(progress_text.split(" de ")[0].strip())
        elif " of " in progress_text:
            current = int(progress_text.split(" of ")[0].strip())
        else:
            import re
            match = re.search(r'(\d+)', progress_text)
            if match:
                return int(match.group(1))
            current = 1
        return current
    except Exception:
        return 1


def get_slide_content(page: Page) -> str:
    """Extract text content from the current slide."""
    try:
        main_content = page.locator(MAIN_CONTENT_SELECTOR)
        if main_content.count() > 0:
            text = main_content.inner_text()
            # Clean up the text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Remove common UI elements
            filtered = []
            skip_patterns = ['©', 'AdinHub', 'Comenzar', 'Contenido del módulo']
            for line in lines:
                if not any(pattern in line for pattern in skip_patterns):
                    filtered.append(line)
            return ' '.join(filtered)
        return ""
    except Exception:
        return ""


def click_next_slide(page: Page) -> bool:
    """Click the next slide button. Returns True if successful."""
    try:
        next_btn = page.locator(NEXT_BUTTON_SELECTOR)
        if next_btn.count() > 0 and next_btn.is_enabled():
            next_btn.click()
            return True
        
        next_icon = page.locator(NEXT_ICON_SELECTOR).first
        if next_icon.count() > 0:
            next_icon.locator("xpath=ancestor::button").first.click()
            return True
        
        return False
    except Exception as e:
        print(f"Error clicking next: {e}")
        return False


def click_start_button(page: Page) -> bool:
    """Click the start/comenzar button if present."""
    try:
        start_btn = page.locator(START_BUTTON_SELECTOR)
        if start_btn.count() > 0:
            start_btn.click()
            page.wait_for_timeout(500)
            return True
        return False
    except Exception:
        return False


def upload_local_module(page: Page, module_path: str) -> bool:
    """Upload a local JSON module file to the page.
    
    Args:
        page: Playwright page instance
        module_path: Path to the local JSON module file
        
    Returns:
        True if upload was successful, False otherwise
    """
    try:
        # Find the file input element (accepts JSON files)
        file_input = page.locator('input[type="file"][accept="json"]')
        
        if file_input.count() == 0:
            print("⚠️ File input not found on page")
            return False
        
        # Upload the file
        file_input.set_input_files(module_path)
        
        # Wait for the module to load (the page processes the JSON)
        page.wait_for_timeout(2000)
        
        # Wait for the loading indicator to disappear if present
        loading_indicator = page.locator("text=Comprobando credenciales")
        if loading_indicator.count() > 0:
            loading_indicator.wait_for(state="hidden", timeout=10000)
        
        # Additional wait for content to render
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)
        
        return True
    except Exception as e:
        print(f"⚠️ Error uploading module: {e}")
        return False


def generate_audio(text: str, output_path: str, voice: str, rate: str = "+0%") -> float:
    """Generate audio using edge-tts CLI. Returns duration in seconds."""
    if not text.strip():
        return 0.0
    
    # Use edge-tts CLI to generate audio
    cmd = [
        "edge-tts",
        "--voice", voice,
        "--rate", rate,
        "--text", text,
        "--write-media", output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"TTS error: {result.stderr}")
        return 0.0
    
    # Get audio duration using ffprobe
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", output_path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 3.0  # Default fallback


def create_slide_video(image_path: str, audio_path: str, output_path: str, duration: float = None) -> bool:
    """Create a video segment from an image and audio file."""
    if duration is None or duration <= 0:
        duration = 3.0  # Default duration if no audio
    
    if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
        # Video with audio
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-t", str(duration + 0.5),  # Add small buffer
            output_path
        ]
    else:
        # Video without audio (silent slide)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            output_path
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def concatenate_videos(video_list_file: str, output_path: str) -> bool:
    """Concatenate multiple videos into one."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", video_list_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg concat error: {result.stderr}")
    return result.returncode == 0


def generate_course_video(
    url: str = DEFAULT_URL,
    output: str = DEFAULT_OUTPUT,
    start: int = 1,
    end: Optional[int] = None,
    pages: Optional[int] = None,
    voice: str = "en-female",
    custom_voice: Optional[str] = None,
    headless: bool = False,
    work_dir: Optional[str] = None,
    keep_temp: bool = False,
    rate: str = "+0%",
    module: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a video from a course using TTS and screen capture.
    
    Args:
        url: Course URL to navigate to
        output: Output video filename
        start: Starting page/slide number
        end: Ending page/slide number (default: last slide)
        pages: Number of pages to process from start (alternative to end)
        voice: TTS voice preset to use
        custom_voice: Custom edge-tts voice name (overrides voice)
        headless: Run browser in headless mode
        work_dir: Working directory for temp files
        keep_temp: Keep temporary files after completion
        rate: TTS speech rate adjustment
        module: Path to local module JSON file to load
        
    Returns:
        Path to the generated video file, or None if failed
    """
    # Get voice
    selected_voice = custom_voice if custom_voice else VOICES.get(voice, DEFAULT_VOICE)
    
    # Determine URL based on whether a local module is provided
    if module:
        module_path = Path(module).resolve()
        if not module_path.exists():
            print(f"❌ Module file not found: {module_path}")
            return None
        if not module_path.suffix.lower() == '.json':
            print(f"❌ Module file must be a JSON file: {module_path}")
            return None
        # Use localFile URL when uploading a module
        actual_url = "https://www.demo.adinhub.com/?localFile=true"
    else:
        module_path = None
        actual_url = url
    
    # Setup working directory
    if work_dir:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir_path = Path(tempfile.mkdtemp(prefix="course_video_"))
        cleanup_work_dir = not keep_temp
    
    # Calculate page range
    start_page = start
    end_page = end
    
    print(f"🎥 Course Video Generator")
    print(f"   URL: {actual_url}")
    if module_path:
        print(f"   Module: {module_path}")
    print(f"   Start page: {start_page}")
    print(f"   End page: {end_page or 'last'}")
    if pages:
        print(f"   Pages count: {pages}")
    print(f"   Voice: {selected_voice}")
    print(f"   Rate: {rate}")
    print(f"   Output: {output}")
    print(f"   Work dir: {work_dir_path}")
    print()
    
    slides_data: List[Tuple[str, Optional[str], float]] = []  # List of (image_path, audio_path, duration)
    
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        )
        page = context.new_page()
        
        print(f"🌐 Navigating to {actual_url}...")
        page.goto(actual_url)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)
        
        # Upload local module if provided
        if module_path:
            print(f"📤 Uploading module: {module_path.name}...")
            if not upload_local_module(page, str(module_path)):
                print("❌ Failed to upload module")
                browser.close()
                return None
            print("✅ Module loaded successfully")
        
        # CAPTURE THE INTRODUCTION PAGE FIRST (before clicking start)
        slide_idx_offset = 0
        start_btn = page.locator(START_BUTTON_SELECTOR)
        if start_btn.count() > 0:
            print(f"📖 Introduction page...", end=" ", flush=True)
            
            # Capture screenshot of introduction
            image_path = work_dir_path / f"slide_{slide_idx_offset:04d}.png"
            page.screenshot(path=str(image_path), timeout=60000)
            
            # Extract text content from introduction
            content = get_slide_content(page)
            
            # Generate audio for introduction
            audio_path = work_dir_path / f"slide_{slide_idx_offset:04d}.mp3"
            if content:
                duration = generate_audio(content, str(audio_path), selected_voice, rate)
                if duration > 0:
                    print(f"✓ ({duration:.1f}s audio)")
                else:
                    duration = 3.0  # Longer default for intro
                    audio_path = None
                    print(f"✓ (TTS failed, {duration:.1f}s)")
            else:
                duration = 3.0  # Default for intro with no text
                audio_path = None
                print(f"✓ (no text, {duration:.1f}s)")
            
            slides_data.append((str(image_path), str(audio_path) if audio_path else None, duration))
            slide_idx_offset = 1  # Start next slides from index 1
        
        total_slides = get_total_slides(page)
        
        # Calculate actual start and end pages
        actual_start = max(1, start_page)
        if pages:
            actual_end = min(actual_start + pages - 1, total_slides)
        elif end_page:
            actual_end = min(end_page, total_slides)
        else:
            actual_end = total_slides
        
        pages_to_process = actual_end - actual_start + 1
        
        print(f"📊 Total slides: {total_slides}")
        print(f"📝 Processing slides {actual_start} to {actual_end} ({pages_to_process} slides)")
        print()
        
        # Click start button if on intro page (NOW after capturing intro)
        if click_start_button(page):
            print("▶️ Clicked start button")
            page.wait_for_timeout(500)
        
        # Navigate to start page if needed
        current = get_current_slide(page)
        if current < actual_start:
            print(f"⏩ Skipping to slide {actual_start}...")
            while current < actual_start:
                if not click_next_slide(page):
                    print(f"⚠️ Could not navigate to start slide")
                    break
                page.wait_for_timeout(300)
                current = get_current_slide(page)
            print(f"   Now at slide {current}")
        
        # Process each slide in range
        for i in range(pages_to_process):
            slide_idx = slide_idx_offset + i  # Account for intro slide if captured
            current = get_current_slide(page)
            print(f"📖 Slide {current}/{total_slides}...", end=" ", flush=True)
            
            # Capture screenshot
            image_path = work_dir_path / f"slide_{slide_idx:04d}.png"
            page.screenshot(path=str(image_path), timeout=60000)  # 60s timeout
            
            # Extract text content
            content = get_slide_content(page)
            
            # Generate audio
            audio_path = work_dir_path / f"slide_{slide_idx:04d}.mp3"
            if content:
                duration = generate_audio(content, str(audio_path), selected_voice, rate)
                if duration > 0:
                    print(f"✓ ({duration:.1f}s audio)")
                else:
                    duration = 2.0
                    audio_path = None
                    print(f"✓ (TTS failed, {duration:.1f}s)")
            else:
                duration = 2.0  # Default for slides with no text
                audio_path = None
                print(f"✓ (no text, {duration:.1f}s)")
            
            slides_data.append((str(image_path), str(audio_path) if audio_path else None, duration))
            
            # Move to next slide
            if i < pages_to_process - 1:
                if not click_next_slide(page):
                    print(f"⚠️ Could not advance to next slide")
                    break
                page.wait_for_timeout(500)
        
        browser.close()
    
    print(f"\n🎬 Creating video segments...")
    
    # Create individual video segments
    segment_paths = []
    for idx, (image_path, audio_path, duration) in enumerate(slides_data):
        segment_path = work_dir_path / f"segment_{idx:04d}.mp4"
        print(f"   Segment {idx + 1}/{len(slides_data)}...", end=" ", flush=True)
        
        if create_slide_video(image_path, audio_path or "", str(segment_path), duration):
            segment_paths.append(str(segment_path))
            print("✓")
        else:
            print("✗")
    
    if not segment_paths:
        print("❌ No video segments created")
        return None
    
    # Create concat list file with absolute paths
    concat_list_path = work_dir_path / "concat_list.txt"
    with open(concat_list_path, 'w') as f:
        for seg_path in segment_paths:
            # Use absolute path to avoid issues
            abs_path = Path(seg_path).resolve()
            f.write(f"file '{abs_path}'\n")
    
    # Concatenate all segments
    print(f"\n🔗 Concatenating {len(segment_paths)} segments...")
    
    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if concatenate_videos(str(concat_list_path.resolve()), output):
        print(f"✅ Video saved to: {output}")
        
        # Get final video info
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", output],
            capture_output=True, text=True
        )
        try:
            total_duration = float(result.stdout.strip())
            print(f"   Duration: {total_duration:.1f} seconds")
        except ValueError:
            pass
    else:
        print("❌ Failed to create final video")
        output = None
    
    # Cleanup
    if cleanup_work_dir:
        shutil.rmtree(work_dir_path)
        print(f"🧹 Cleaned up temp files")
    else:
        print(f"📁 Temp files kept in: {work_dir_path}")
    
    print("\n✨ Done!")
    return output


def main():
    """CLI entry point."""
    args = parse_args()
    
    generate_course_video(
        url=args.url,
        output=args.output,
        start=args.start,
        end=args.end,
        pages=args.pages,
        voice=args.voice,
        custom_voice=args.custom_voice,
        headless=args.headless,
        work_dir=args.work_dir,
        keep_temp=args.keep_temp,
        rate=args.rate,
        module=args.module,
    )


if __name__ == "__main__":
    main()

