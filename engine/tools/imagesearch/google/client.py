"""Google Images search using Playwright (no Google API).

Scrapes https://www.google.com/search?tbm=isch&q=... using a thread-local browser
instance for thread-safety and reuse, following the pattern in freepik client.

Features:
- Direct Google Images scraping (no Startpage proxy)
- Automatic consent dialog handling
- Text CAPTCHA solving with Pixtral vision LLM (when Google shows text CAPTCHA)
- reCAPTCHA solving with Pixtral vision LLM (batched image analysis)
"""

from __future__ import annotations

import atexit
import re
import threading
from typing import List, Optional, Set
from urllib.parse import quote, urlparse, unquote

# Vision LLM for CAPTCHA solving (lazy import to avoid dependency if not needed)
_vision_llm = None


# ---- URL Validation/Sanitization ----
# Patterns that indicate broken or problematic image URLs
_BLOCKED_URL_PATTERNS = [
    r'placeholder',
    r'no[-_]?image',
    r'default[-_]?image',
    r'blank\.',
    r'spacer\.',
    r'1x1\.',
    r'pixel\.',
    r'tracking',
    r'beacon',
    r'analytics',
    r'/ads/',
    r'/ad/',
    r'doubleclick',
    r'googlesyndication',
    r'data:image',  # Data URIs
]

# Blocked file extensions (non-image or problematic for web embedding)
# Note: .webp and .gif are valid web formats, only block truly problematic ones
_BLOCKED_EXTENSIONS = {'.svg', '.bmp', '.ico', '.tiff', '.tif'}

# Domains known to have issues with hotlinking or broken images
_BLOCKED_DOMAINS = {
    'facebook.com',
    'fbcdn.net',
    'instagram.com',
    'twitter.com',
    'x.com',
    'linkedin.com',
    'pinterest.com',
    'tiktok.com',
}


def is_valid_image_url(url: str) -> bool:
    """
    Validate if a URL is likely to be a working image URL for HTML embedding.
    
    Checks:
    - URL is well-formed (http/https)
    - URL isn't too long (browsers have limits)
    - URL doesn't contain blocked patterns (tracking, placeholders, etc.)
    - URL doesn't have problematic file extensions
    - URL isn't from a domain known to block hotlinking
    
    Args:
        url: The image URL to validate
        
    Returns:
        True if the URL appears valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # Must start with http:// or https://
    if not url.startswith(('http://', 'https://')):
        return False
    
    # URL shouldn't be too long (browser limit is ~2000 chars, be conservative)
    if len(url) > 2000:
        return False
    
    # Parse the URL
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    
    # Must have a valid netloc (domain)
    if not parsed.netloc:
        return False
    
    # Check for blocked domains
    domain = parsed.netloc.lower()
    for blocked in _BLOCKED_DOMAINS:
        if blocked in domain:
            return False
    
    # Check URL path for blocked patterns
    url_lower = url.lower()
    for pattern in _BLOCKED_URL_PATTERNS:
        if re.search(pattern, url_lower):
            return False
    
    # Check file extension
    path = unquote(parsed.path).lower()
    for ext in _BLOCKED_EXTENSIONS:
        if path.endswith(ext):
            return False
    
    return True

# Maximum images per Pixtral batch call
PIXTRAL_MAX_BATCH = 8


def _get_vision_llm():
    """Lazy-load vision LLM only when needed for CAPTCHA solving."""
    global _vision_llm
    if _vision_llm is None:
        from LLMs.imagetext2text import create_vision_llm
        _vision_llm = create_vision_llm(provider="pixtral", temperature=0.0)
    return _vision_llm


class GoogleImagesClient:
    """Thread-safe Google Images scraper with persistent browser per thread."""

    _local = threading.local()
    _instances_lock = threading.Lock()
    _all_instances: List["GoogleImagesClient"] = []
    _cleanup_registered = False

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._request_count = 0
        self._max_requests_before_refresh = 75
        self._thread_id = threading.current_thread().name

    @classmethod
    def get_instance(cls) -> "GoogleImagesClient":
        instance = getattr(cls._local, "instance", None)
        if instance is None:
            instance = cls()
            instance._initialize()
            cls._local.instance = instance

            with cls._instances_lock:
                cls._all_instances.append(instance)
                if not cls._cleanup_registered:
                    atexit.register(cls._cleanup_all)
                    cls._cleanup_registered = True

        return instance

    def _initialize(self) -> None:
        """Initialize Playwright browser with anti-detection settings."""
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            raise RuntimeError(
                "Playwright is required for the google image search provider. "
                "Install optional deps: pip install -e '.[scraper]' and run "
                "'playwright install chromium'."
            ) from e

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )

        self._create_context()

    def _create_context(self) -> None:
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass

        self._context = self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
        )

        self._page = self._context.new_page()

        # Best-effort stealth patches (same dependency as Freepik)
        try:
            from playwright_stealth import Stealth

            Stealth(navigator_platform_override="MacIntel").apply_stealth_sync(self._page)
        except Exception:
            pass

        # Override navigator.webdriver via init script
        try:
            self._page.add_init_script(
                """
                Object.defineProperty(navigator, "webdriver", { get: () => undefined });
                window.chrome = { runtime: {} };
                """
            )
        except Exception:
            pass

        self._request_count = 0

    def _ensure_fresh_context(self) -> None:
        self._request_count += 1
        if self._request_count >= self._max_requests_before_refresh:
            self._create_context()

    def _try_handle_consent(self) -> bool:
        """Best-effort click-through for Google consent pages."""
        try:
            url = self._page.url or ""
            body_text = ""
            try:
                body_text = self._page.inner_text("body")
            except Exception:
                body_text = ""
            
            # Check if consent dialog is present
            consent_indicators = [
                "consent.google" in url,
                "consent" in url.lower(),
                "Before you continue" in body_text,
                "Antes de ir a Google" in body_text,  # Spanish
                "Avant d'accéder" in body_text,  # French
            ]
            
            if not any(consent_indicators):
                return True  # No consent needed

            # Common accept buttons (EU/consent variants - multiple languages)
            selectors = [
                # English
                'button:has-text("Accept all")',
                'button:has-text("Accept all cookies")',
                'button:has-text("I agree")',
                'button:has-text("Accept")',
                # Spanish
                'button:has-text("Aceptar todo")',
                'button:has-text("Aceptar")',
                # French
                'button:has-text("Tout accepter")',
                # Generic
                'input[type="submit"][value*="Accept"]',
                'input[type="submit"][value*="I agree"]',
                '[role="button"]:has-text("Accept all")',
                '[role="button"]:has-text("I agree")',
            ]

            for sel in selectors:
                try:
                    loc = self._page.locator(sel).first
                    if loc.count() > 0:
                        loc.click(timeout=2000)
                        self._page.wait_for_timeout(500)
                        return True
                except Exception:
                    continue

            return False
        except Exception:
            return False

    def _is_blocked(self) -> dict:
        """Check if Google is blocking us and determine the type of block.
        
        Returns:
            dict with keys:
            - blocked: bool - whether we're blocked
            - type: str - 'none', 'recaptcha', 'text_captcha', 'unusual_traffic'
            - solvable: bool - whether we can solve this with Pixtral
        """
        try:
            url = self._page.url.lower()
            body_text = ""
            try:
                body_text = self._page.inner_text("body").lower()
            except Exception:
                pass
            
            # Check for /sorry/ page (Google's block page)
            if "/sorry/" in url:
                # Check if it's reCAPTCHA or text CAPTCHA (both now solvable with Pixtral)
                if "recaptcha" in body_text or "g-recaptcha" in self._page.content().lower():
                    return {"blocked": True, "type": "recaptcha", "solvable": True}
                # Text CAPTCHA has an input field for characters
                if self._page.query_selector('input[name="captcha"]') or "enter the characters" in body_text:
                    return {"blocked": True, "type": "text_captcha", "solvable": True}
                return {"blocked": True, "type": "unusual_traffic", "solvable": False}
            
            # Check for unusual traffic message on regular page
            if "unusual traffic" in body_text or "automated queries" in body_text:
                return {"blocked": True, "type": "unusual_traffic", "solvable": False}
            
            # Check for CAPTCHA elements (reCAPTCHA is now solvable with Pixtral batched analysis)
            if "recaptcha" in body_text or self._page.query_selector('.g-recaptcha, [data-sitekey]'):
                return {"blocked": True, "type": "recaptcha", "solvable": True}
            
            return {"blocked": False, "type": "none", "solvable": False}
            
        except Exception:
            return {"blocked": False, "type": "none", "solvable": False}

    def _solve_text_captcha(self, max_attempts: int = 3) -> bool:
        """
        Attempt to solve Google's text CAPTCHA using Pixtral vision LLM.
        
        Google's text CAPTCHA shows an image with distorted text that needs to be typed.
        This is different from reCAPTCHA which requires image selection.
        
        Args:
            max_attempts: Maximum number of attempts to solve
            
        Returns:
            True if solved successfully, False otherwise
        """
        from langchain_core.messages import HumanMessage
        
        for attempt in range(max_attempts):
            try:
                print(f"   🔐 Google text CAPTCHA detected, solving attempt {attempt + 1}/{max_attempts}...")
                
                # Wait for page to stabilize
                self._page.wait_for_timeout(1000)
                
                # Find CAPTCHA image - Google uses various selectors
                captcha_img = (
                    self._page.query_selector('img[src*="sorry"]') or
                    self._page.query_selector('img[src*="captcha"]') or
                    self._page.query_selector('#captcha-form img') or
                    self._page.query_selector('form img')
                )
                
                if not captcha_img:
                    print("   ⚠️ CAPTCHA image not found")
                    return False
                
                # Get image URL
                captcha_url = captcha_img.get_attribute('src')
                if not captcha_url:
                    print("   ❌ Could not get CAPTCHA image URL")
                    return False
                
                # Make URL absolute
                if captcha_url.startswith('/'):
                    captcha_url = 'https://www.google.com' + captcha_url
                elif captcha_url.startswith('//'):
                    captcha_url = 'https:' + captcha_url
                
                print(f"   📸 CAPTCHA image: {captcha_url[:60]}...")
                
                # Use Pixtral to read the CAPTCHA
                try:
                    vision_llm = _get_vision_llm()
                    
                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Read the text shown in this CAPTCHA image. "
                                    "Return ONLY the exact characters you see, nothing else. "
                                    "No quotes, no explanation, just the characters."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": captcha_url}
                            }
                        ]
                    )
                    
                    response = vision_llm.invoke([message])
                    captcha_text = response.content.strip().replace('"', '').replace("'", '').strip()
                    
                    print(f"   🔤 Vision LLM read: '{captcha_text}'")
                    
                    if not captcha_text or len(captcha_text) < 3:
                        print("   ⚠️ Invalid CAPTCHA text (too short)")
                        continue
                        
                except Exception as e:
                    print(f"   ❌ Vision LLM error: {str(e)}")
                    continue
                
                # Find and fill input field
                input_field = (
                    self._page.query_selector('input[name="captcha"]') or
                    self._page.query_selector('input[name="q"]') or
                    self._page.query_selector('#captcha-form input[type="text"]') or
                    self._page.query_selector('form input[type="text"]')
                )
                
                if not input_field:
                    print("   ❌ Could not find CAPTCHA input field")
                    return False
                
                # Type solution
                input_field.fill('')
                input_field.type(captcha_text, delay=50)
                self._page.wait_for_timeout(300)
                
                # Submit
                submit_btn = (
                    self._page.query_selector('input[type="submit"]') or
                    self._page.query_selector('button[type="submit"]') or
                    self._page.query_selector('#captcha-form button')
                )
                
                if submit_btn:
                    submit_btn.click()
                else:
                    # Try pressing Enter
                    input_field.press('Enter')
                
                self._page.wait_for_timeout(2000)
                
                # Check if still blocked
                block_status = self._is_blocked()
                if not block_status["blocked"]:
                    print("   ✅ CAPTCHA solved successfully!")
                    return True
                else:
                    print("   ⚠️ CAPTCHA solution incorrect, retrying...")
                    
            except Exception as e:
                print(f"   ❌ Error solving CAPTCHA: {str(e)}")
                continue
        
        print(f"   ❌ Failed to solve CAPTCHA after {max_attempts} attempts")
        return False

    def _get_recaptcha_frame(self):
        """
        Locate and return the reCAPTCHA challenge iframe.
        
        reCAPTCHA uses nested iframes. The challenge iframe contains the image grid.
        
        Returns:
            Frame object if found, None otherwise
        """
        try:
            # Debug: List all frames
            all_frames = self._page.frames
            print(f"   🔍 Page has {len(all_frames)} frames")
            
            for i, frame in enumerate(all_frames):
                frame_url = frame.url.lower() if frame.url else ""
                # Debug: show frame URLs
                if frame_url and frame_url != "about:blank":
                    short_url = frame_url[:100] if len(frame_url) > 100 else frame_url
                    print(f"      Frame {i}: {short_url}")
            
            # reCAPTCHA challenge is in an iframe with specific patterns
            # Priority 1: Look for bframe (challenge frame with image grid)
            for frame in all_frames:
                frame_url = frame.url.lower() if frame.url else ""
                if "recaptcha" in frame_url and "bframe" in frame_url:
                    print(f"   ✓ Found bframe (challenge frame)")
                    return frame
            
            # Priority 2: Look for any recaptcha frame that's not the anchor
            for frame in all_frames:
                frame_url = frame.url.lower() if frame.url else ""
                if "recaptcha" in frame_url and "anchor" not in frame_url and frame_url != "about:blank":
                    print(f"   ✓ Found recaptcha frame (non-anchor)")
                    return frame
            
            # Priority 3: Check if we're on a /sorry/ page with embedded recaptcha
            current_url = self._page.url.lower()
            if "/sorry/" in current_url:
                print(f"   🔍 On Google /sorry/ page, looking for recaptcha elements...")
                # The recaptcha might be directly in the page, not in an iframe
                # Check for recaptcha elements
                recaptcha_div = self._page.query_selector('.g-recaptcha, #recaptcha, [data-sitekey]')
                if recaptcha_div:
                    print(f"   ✓ Found recaptcha div in main page")
                    # Return the main frame
                    return self._page.main_frame
            
            print(f"   ⚠️ No reCAPTCHA frame found")
            return None
        except Exception as e:
            print(f"   ⚠️ Error finding reCAPTCHA frame: {str(e)}")
            return None

    def _extract_recaptcha_task(self, frame) -> str:
        """
        Extract the task text from the reCAPTCHA challenge.
        
        The task usually appears as "Select all images with <target>" where
        <target> is in a strong element (e.g., "traffic lights", "bicycles").
        
        Args:
            frame: The reCAPTCHA challenge frame
            
        Returns:
            Task text like "Select all images with traffic lights"
        """
        try:
            # First, try to get the specific target from the strong element
            # This is the most specific part (e.g., "bicycles", "traffic lights")
            strong_selectors = [
                '.rc-imageselect-desc strong',
                '.rc-imageselect-desc-wrapper strong',
                '.rc-imageselect-instructions strong',
            ]
            
            target = None
            for selector in strong_selectors:
                try:
                    strong_el = frame.query_selector(selector)
                    if strong_el:
                        target = strong_el.inner_text().strip()
                        if target:
                            break
                except Exception:
                    continue
            
            if target:
                # Found the target, return a clear task description
                return f"Select all images with {target}"
            
            # Fallback: get the full description text
            desc_selectors = [
                '.rc-imageselect-desc-wrapper',
                '.rc-imageselect-desc',
                '.rc-imageselect-instructions',
            ]
            
            for selector in desc_selectors:
                try:
                    element = frame.query_selector(selector)
                    if element:
                        text = element.inner_text().strip()
                        if text and len(text) > 5:  # Reasonable length
                            return text
                except Exception:
                    continue
            
            return "Select all matching images"
        except Exception as e:
            print(f"   ⚠️ Error extracting task: {str(e)}")
            return "Select all matching images"

    def _extract_tile_urls(self, frame) -> List[str]:
        """
        Extract individual tile image URLs from the reCAPTCHA grid.
        
        Args:
            frame: The reCAPTCHA challenge frame
            
        Returns:
            List of image URLs (9 for 3x3 grid, 16 for 4x4 grid)
        """
        try:
            # Debug: print frame URL to understand which frame we're in
            try:
                frame_url = frame.url if hasattr(frame, 'url') else 'unknown'
                print(f"   🔍 Extracting tiles from frame: {frame_url[:80]}...")
            except Exception:
                pass
            
            # reCAPTCHA tiles - try multiple selector strategies
            tile_selectors = [
                # Standard tile selectors
                '.rc-imageselect-tile img',
                'td.rc-imageselect-tile img', 
                '.rc-image-tile-wrapper img',
                '.rc-imageselect-table img',
                # Alternative selectors
                'table.rc-imageselect-table td img',
                '.rc-imageselect-target img',
                'div.rc-image-tile-target img',
                # More generic
                'table td img',
                'img.rc-image-tile-11',
                'img.rc-image-tile-33',
                'img.rc-image-tile-44',
            ]
            
            for selector in tile_selectors:
                try:
                    tiles = frame.query_selector_all(selector)
                    if tiles and len(tiles) > 0:
                        urls = []
                        for tile in tiles:
                            src = tile.get_attribute('src')
                            if src and not src.startswith('data:'):
                                # Make URL absolute if needed
                                if src.startswith('//'):
                                    src = 'https:' + src
                                elif src.startswith('/'):
                                    src = 'https://www.google.com' + src
                                urls.append(src)
                        if urls:
                            print(f"   ✓ Found {len(urls)} tile images with selector: {selector}")
                            return urls
                except Exception as e:
                    continue
            
            # Fallback: Try to get all images in the frame
            try:
                all_imgs = frame.query_selector_all('img')
                if all_imgs:
                    print(f"   🔍 Found {len(all_imgs)} total images in frame, filtering...")
                    urls = []
                    for img in all_imgs:
                        src = img.get_attribute('src')
                        if src and not src.startswith('data:') and 'payload' in src.lower():
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif src.startswith('/'):
                                src = 'https://www.google.com' + src
                            urls.append(src)
                    if urls:
                        print(f"   ✓ Filtered to {len(urls)} payload images")
                        return urls
            except Exception:
                pass
            
            # Debug: Try to understand what's in the frame
            try:
                html_content = frame.content()
                if 'rc-imageselect' in html_content:
                    print(f"   🔍 Frame contains rc-imageselect classes")
                    # Count images
                    img_count = html_content.count('<img')
                    print(f"   🔍 Frame has {img_count} <img> tags")
                else:
                    print(f"   ⚠️ Frame does not contain expected reCAPTCHA classes")
            except Exception as e:
                print(f"   ⚠️ Could not inspect frame content: {str(e)}")
            
            return []
        except Exception as e:
            print(f"   ⚠️ Error extracting tile URLs: {str(e)}")
            return []
    
    def _take_recaptcha_screenshot(self, frame) -> Optional[str]:
        """
        Take a screenshot of the reCAPTCHA challenge as a fallback.
        
        Returns:
            Base64 encoded screenshot or None
        """
        import base64
        try:
            # Try multiple selectors for the challenge area
            selectors = [
                '.rc-imageselect-challenge',
                '.rc-imageselect-table-33',
                '.rc-imageselect-table',
                '.rc-imageselect',
                'table',  # Fallback to any table
            ]
            
            for selector in selectors:
                try:
                    challenge_area = frame.query_selector(selector)
                    if challenge_area:
                        screenshot_bytes = challenge_area.screenshot()
                        return f"data:image/png;base64,{base64.b64encode(screenshot_bytes).decode()}"
                except Exception:
                    continue
            
            # Last resort: screenshot the entire frame
            try:
                # For the main page, take a screenshot of the viewport
                screenshot_bytes = self._page.screenshot()
                return f"data:image/png;base64,{base64.b64encode(screenshot_bytes).decode()}"
            except Exception:
                pass
            
            return None
        except Exception as e:
            print(f"   ⚠️ Screenshot failed: {str(e)}")
            return None
    
    def _solve_recaptcha_with_screenshot(self, frame, task_text: str) -> Optional[List[int]]:
        """
        Solve reCAPTCHA by analyzing a screenshot of the entire grid.
        
        This is the PRIMARY method for solving reCAPTCHA challenges.
        Takes a screenshot of the challenge and asks Pixtral to identify
        which grid positions contain matching images.
        
        Args:
            frame: The reCAPTCHA challenge frame
            task_text: The task description (e.g., "Select all images with bicycles")
            
        Returns:
            List of matching tile indices (0-based), or None if failed
        """
        from langchain_core.messages import HumanMessage
        
        try:
            print("   📸 Taking screenshot of reCAPTCHA challenge...")
            screenshot_data = self._take_recaptcha_screenshot(frame)
            
            if not screenshot_data:
                print("   ❌ Could not take screenshot")
                return None
            
            # Detect grid size by checking the table structure
            grid_size = 9  # Default 3x3
            try:
                table = frame.query_selector('table.rc-imageselect-table')
                if table:
                    rows = frame.query_selector_all('table.rc-imageselect-table tr')
                    cols = frame.query_selector_all('table.rc-imageselect-table tr:first-child td')
                    if rows and cols:
                        detected = len(rows) * len(cols)
                        if detected in [9, 16]:
                            grid_size = detected
                            print(f"   🔍 Detected {int(detected**0.5)}x{int(detected**0.5)} grid")
            except Exception:
                pass
            
            print("   🤖 Analyzing screenshot with Pixtral...")
            vision_llm = _get_vision_llm()
            
            # Extract the target object from task text
            # Task is usually like "Select all images with bicycles"
            target = task_text.replace("Select all images with ", "").replace("Select all squares with ", "").strip()
            
            grid_desc = "3x3 (9 squares, numbered 1-9)" if grid_size == 9 else "4x4 (16 squares, numbered 1-16)"
            
            # Ask Pixtral to analyze the grid screenshot
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""You are solving a reCAPTCHA image challenge.

TASK: {task_text}

Looking at the image grid ({grid_desc}), identify which squares contain "{target}".

Grid positions are numbered left-to-right, top-to-bottom:
- For 3x3: positions 1,2,3 (top row), 4,5,6 (middle), 7,8,9 (bottom)
- For 4x4: positions 1-4 (top), 5-8, 9-12, 13-16 (bottom)

Look carefully at each square. Which positions clearly show "{target}"?

IMPORTANT: Return ONLY a comma-separated list of position numbers.
Example: 1,3,5,9
If no squares match: none"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_data}
                    }
                ]
            )
            
            response = vision_llm.invoke([message])
            raw_response = response.content.strip().lower()
            
            print(f"   🔤 Pixtral response: {raw_response}")
            
            if "none" in raw_response or not raw_response:
                return []
            
            # Parse response to get indices (convert 1-based to 0-based)
            matching_indices = []
            for part in raw_response.replace('\n', ',').split(','):
                part = part.strip()
                # Extract numbers from the response
                import re
                numbers = re.findall(r'\d+', part)
                for num_str in numbers:
                    try:
                        num = int(num_str)
                        if 1 <= num <= 16:  # Valid grid position
                            matching_indices.append(num - 1)  # Convert to 0-based
                    except ValueError:
                        continue
            
            # Remove duplicates and sort
            matching_indices = sorted(list(set(matching_indices)))
            return matching_indices
            
        except Exception as e:
            print(f"   ❌ Screenshot analysis error: {str(e)}")
            return None

    def _analyze_single_batch(self, tile_urls: List[str], task_text: str) -> List[bool]:
        """
        Send up to 8 images to Pixtral, ask which match the task.
        
        Args:
            tile_urls: List of up to 8 image URLs
            task_text: The reCAPTCHA task (e.g., "Select all images with traffic lights")
            
        Returns:
            List of booleans indicating which images match
        """
        from langchain_core.messages import HumanMessage
        
        if not tile_urls:
            return []
        
        try:
            vision_llm = _get_vision_llm()
            
            # Build message with all images
            content = [{
                "type": "text",
                "text": f"""You are analyzing {len(tile_urls)} images for a reCAPTCHA challenge.
Task: "{task_text}"

For each image (Image 1 through Image {len(tile_urls)}), determine if it matches the task.
Return ONLY a comma-separated list of YES or NO for each image in order.
Example for 3 images: YES,NO,YES

Your response (exactly {len(tile_urls)} answers):"""
            }]
            
            for idx, url in enumerate(tile_urls, 1):
                content.append({"type": "text", "text": f"Image {idx}:"})
                content.append({"type": "image_url", "image_url": {"url": url}})
            
            response = vision_llm.invoke([HumanMessage(content=content)])
            
            # Parse "YES,NO,YES,NO,YES,NO,YES,NO" into [True, False, True, ...]
            raw_response = response.content.strip().upper()
            # Clean up response - extract just YES/NO parts
            answers = []
            for part in raw_response.replace('\n', ',').split(','):
                part = part.strip()
                if 'YES' in part:
                    answers.append(True)
                elif 'NO' in part:
                    answers.append(False)
            
            # Ensure we have the right number of answers
            while len(answers) < len(tile_urls):
                answers.append(False)  # Default to NO if unclear
            
            return answers[:len(tile_urls)]
            
        except Exception as e:
            print(f"   ⚠️ Pixtral batch analysis error: {str(e)}")
            # Return all False on error
            return [False] * len(tile_urls)

    def _analyze_tiles_batch(self, tile_urls: List[str], task_text: str) -> List[bool]:
        """
        Analyze tiles in batches of 8 (Pixtral's max), return list of booleans.
        
        Args:
            tile_urls: List of all tile image URLs
            task_text: The reCAPTCHA task
            
        Returns:
            List of booleans indicating which tiles match the task
        """
        results = []
        
        for i in range(0, len(tile_urls), PIXTRAL_MAX_BATCH):
            batch = tile_urls[i:i + PIXTRAL_MAX_BATCH]
            print(f"   🔍 Analyzing batch {i // PIXTRAL_MAX_BATCH + 1} ({len(batch)} images)...")
            batch_results = self._analyze_single_batch(batch, task_text)
            results.extend(batch_results)
        
        return results

    def _click_matching_tiles(self, frame, matching_indices: List[int]) -> None:
        """
        Click tiles at the given indices (0-based).
        
        Args:
            frame: The reCAPTCHA challenge frame
            matching_indices: List of 0-based indices to click
        """
        try:
            tiles = frame.query_selector_all('.rc-imageselect-tile, td.rc-imageselect-tile')
            
            for idx in matching_indices:
                if idx < len(tiles):
                    try:
                        tiles[idx].click()
                        self._page.wait_for_timeout(300)  # Small delay between clicks
                    except Exception as e:
                        print(f"   ⚠️ Error clicking tile {idx}: {str(e)}")
        except Exception as e:
            print(f"   ⚠️ Error clicking tiles: {str(e)}")

    def _check_for_new_tiles(self, frame, old_urls: List[str]) -> List[int]:
        """
        Check if any tiles have been replaced with new images.
        
        Some reCAPTCHA challenges reload tiles after clicking, requiring
        re-analysis of the changed tiles.
        
        Args:
            frame: The reCAPTCHA challenge frame
            old_urls: Previous list of tile URLs
            
        Returns:
            List of indices where tiles have changed
        """
        try:
            new_urls = self._extract_tile_urls(frame)
            
            if len(new_urls) != len(old_urls):
                return []  # Grid size changed, unusual
            
            changed_indices = []
            for i, (old, new) in enumerate(zip(old_urls, new_urls)):
                if old != new:
                    changed_indices.append(i)
            
            return changed_indices
        except Exception:
            return []

    def _click_recaptcha_checkbox(self) -> bool:
        """
        Click the reCAPTCHA checkbox to trigger the image challenge.
        
        The reCAPTCHA checkbox is in the anchor frame. Clicking it triggers
        Google's risk assessment, which may show an image challenge in the bframe.
        
        Returns:
            True if checkbox was clicked, False otherwise
        """
        try:
            # Find the anchor frame (contains the checkbox)
            anchor_frame = None
            for f in self._page.frames:
                frame_url = f.url.lower() if f.url else ""
                if "recaptcha" in frame_url and "anchor" in frame_url:
                    anchor_frame = f
                    break
            
            if not anchor_frame:
                print("   ⚠️ Could not find reCAPTCHA anchor frame")
                return False
            
            # Find and click the checkbox
            checkbox_selectors = [
                '.recaptcha-checkbox-border',
                '.recaptcha-checkbox',
                '#recaptcha-anchor',
                '[role="checkbox"]',
            ]
            
            for selector in checkbox_selectors:
                try:
                    checkbox = anchor_frame.query_selector(selector)
                    if checkbox:
                        checkbox.click()
                        print("   ✓ Clicked reCAPTCHA checkbox")
                        return True
                except Exception:
                    continue
            
            print("   ⚠️ Could not find reCAPTCHA checkbox")
            return False
        except Exception as e:
            print(f"   ⚠️ Error clicking checkbox: {str(e)}")
            return False

    def _wait_for_challenge(self, timeout: int = 5000) -> bool:
        """
        Wait for the reCAPTCHA image challenge to appear.
        
        After clicking the checkbox, the bframe is populated with the challenge.
        This method waits for the challenge content to load.
        
        Args:
            timeout: Maximum time to wait in milliseconds
            
        Returns:
            True if challenge appeared, False otherwise
        """
        try:
            # Wait in increments and check for challenge
            wait_increment = 500
            total_waited = 0
            
            while total_waited < timeout:
                self._page.wait_for_timeout(wait_increment)
                total_waited += wait_increment
                
                # Check if bframe now has challenge content
                for f in self._page.frames:
                    frame_url = f.url.lower() if f.url else ""
                    if "recaptcha" in frame_url and "bframe" in frame_url:
                        # Check if the challenge table exists
                        try:
                            table = f.query_selector('table.rc-imageselect-table, .rc-imageselect-challenge')
                            if table:
                                print("   ✓ Challenge loaded")
                                return True
                        except Exception:
                            pass
            
            print("   ⚠️ Challenge did not load in time")
            return False
        except Exception as e:
            print(f"   ⚠️ Error waiting for challenge: {str(e)}")
            return False

    def _solve_recaptcha(self, max_attempts: int = 3) -> bool:
        """
        Attempt to solve Google's reCAPTCHA using Pixtral vision LLM.
        
        reCAPTCHA shows a grid of images (3x3 or 4x4) and asks to select
        all images matching a description (e.g., "traffic lights").
        
        This method:
        1. Clicks the checkbox to trigger the challenge
        2. Waits for the image challenge to load
        3. Takes a screenshot and uses Pixtral to identify matching tiles
        4. Clicks matching tiles
        5. Clicks Verify and checks result
        
        Args:
            max_attempts: Maximum number of attempts to solve
            
        Returns:
            True if solved successfully, False otherwise
        """
        for attempt in range(max_attempts):
            try:
                print(f"   🔐 Google reCAPTCHA detected, solving attempt {attempt + 1}/{max_attempts}...")
                
                # Wait for page to stabilize
                self._page.wait_for_timeout(1000)
                
                # STEP 1: Click the checkbox to trigger the image challenge
                # This is CRITICAL - the bframe is empty until checkbox is clicked
                if not self._click_recaptcha_checkbox():
                    print("   ⚠️ Could not click checkbox, retrying...")
                    continue
                
                # STEP 2: Wait for the challenge to load
                if not self._wait_for_challenge(timeout=5000):
                    print("   ⚠️ Challenge didn't load, checking if already solved...")
                    # Maybe we passed without an image challenge
                    self._page.wait_for_timeout(1000)
                    if not self._is_blocked():
                        print("   ✓ reCAPTCHA passed without image challenge!")
                        return True
                    continue
                
                # STEP 3: Find the challenge frame
                frame = self._get_recaptcha_frame()
                if not frame:
                    print("   ❌ Could not find challenge frame")
                    continue
                
                # STEP 4: Extract task text
                task_text = self._extract_recaptcha_task(frame)
                print(f"   📋 Task: {task_text}")
                
                # STEP 5: Main solving loop using screenshot-based approach
                # Screenshot is more reliable than extracting individual tile URLs
                max_rounds = 5  # Prevent infinite loops (for dynamic tile reloading)
                for round_num in range(max_rounds):
                    print(f"   🎯 Round {round_num + 1}: Analyzing challenge...")
                    
                    # Use screenshot-based approach (more reliable)
                    matching_indices = self._solve_recaptcha_with_screenshot(frame, task_text)
                    
                    if matching_indices is None:
                        print("   ❌ Screenshot analysis failed")
                        break
                    
                    if not matching_indices:
                        print("   ℹ️ No matching tiles found (might need to skip or verify)")
                        # Try clicking skip if available
                        try:
                            skip_btn = frame.query_selector('button:has-text("Skip"), button:has-text("SKIP")')
                            if skip_btn:
                                print("   🔘 Clicking Skip...")
                                skip_btn.click()
                                self._page.wait_for_timeout(1500)
                                # Check if new challenge appeared or solved
                                new_task = self._extract_recaptcha_task(frame)
                                if new_task != task_text:
                                    task_text = new_task
                                    print(f"   📋 New task: {task_text}")
                                    continue
                        except Exception:
                            pass
                        break
                    
                    print(f"   ✓ Found {len(matching_indices)} matching tiles: {[i+1 for i in matching_indices]}")
                    
                    # Click matching tiles
                    self._click_matching_tiles(frame, matching_indices)
                    
                    # Wait for any dynamic tile reloads
                    self._page.wait_for_timeout(1500)
                    
                    # Check if more tiles need to be selected (dynamic reload case)
                    # Some challenges reload clicked tiles with new images
                    # For now, proceed to verify - more rounds if needed
                
                # Click Verify button
                try:
                    verify_btn = (
                        frame.query_selector('#recaptcha-verify-button') or
                        frame.query_selector('button:has-text("Verify")') or
                        frame.query_selector('button:has-text("VERIFY")') or
                        frame.query_selector('.rc-button-default')
                    )
                    if verify_btn:
                        print("   🔘 Clicking Verify...")
                        verify_btn.click()
                        self._page.wait_for_timeout(2500)
                except Exception as e:
                    print(f"   ⚠️ Error clicking Verify: {str(e)}")
                
                # Check if we're still blocked
                block_status = self._is_blocked()
                if not block_status["blocked"]:
                    print("   ✅ reCAPTCHA solved successfully!")
                    return True
                else:
                    print("   ⚠️ reCAPTCHA not solved, retrying...")
                    
            except Exception as e:
                print(f"   ❌ Error solving reCAPTCHA: {str(e)}")
                continue
        
        print(f"   ❌ Failed to solve reCAPTCHA after {max_attempts} attempts")
        return False

    def _extract_best_preview_image_url(self) -> Optional[str]:
        """Extract a likely full-size image URL from the preview panel."""
        try:
            # Multiple DOM variants exist; look for the biggest http(s) img currently rendered.
            candidate = self._page.evaluate(
                """
                () => {
                  const imgs = Array.from(document.querySelectorAll('img'));
                  const httpImgs = imgs
                    .map(img => {
                      const src = img.currentSrc || img.src || '';
                      return {src, w: img.naturalWidth || 0, h: img.naturalHeight || 0};
                    })
                    .filter(x => x.src && x.src.startsWith('http'));
                  httpImgs.sort((a,b) => (b.w*b.h) - (a.w*a.h));
                  return httpImgs.length ? httpImgs[0].src : null;
                }
                """
            )
            if candidate and isinstance(candidate, str) and candidate.startswith("http"):
                return candidate
            return None
        except Exception:
            return None

    def search_images(self, query: str, max_results: int = 5) -> List[dict]:
        """Search for images on Google Images by scraping."""
        if not query:
            return [{"error": "Query is required"}]

        try:
            self._ensure_fresh_context()

            search_url = f"https://www.google.com/search?tbm=isch&q={quote(query)}"
            self._page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

            # Handle consent dialog (may navigate away)
            self._try_handle_consent()
            
            # After consent, ensure we're on image search results
            # Google uses tbm=isch or newer udm=2 for image search
            current_url = self._page.url
            is_image_search = ("tbm=isch" in current_url or "udm=2" in current_url) and "google.com/search" in current_url
            if not is_image_search:
                # Consent redirected us - navigate back to image search
                self._page.wait_for_timeout(500)
                self._page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

            # Check for blocking/CAPTCHA
            block_status = self._is_blocked()
            if block_status["blocked"]:
                if block_status["solvable"]:
                    solved = False
                    if block_status["type"] == "text_captcha":
                        # Try to solve text CAPTCHA with Pixtral
                        solved = self._solve_text_captcha()
                    elif block_status["type"] == "recaptcha":
                        # Try to solve reCAPTCHA with Pixtral (batched image analysis)
                        solved = self._solve_recaptcha()
                    
                    if solved:
                        # Re-navigate after solving
                        self._page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
                    else:
                        return [{"error": f"Could not solve Google {block_status['type']}. Please try again later."}]
                else:
                    error_msg = f"Google blocked the request ({block_status['type']}). "
                    if block_status["type"] == "unusual_traffic":
                        error_msg += "Too many requests detected."
                    return [{"error": error_msg}]

            results: List[dict] = []
            seen: Set[str] = set()

            # Wait for thumbnails to show up (best-effort)
            try:
                self._page.wait_for_selector('img.YQ4gaf, img.rg_i', timeout=5000, state="attached")
            except Exception:
                pass

            # Scroll to load more images
            for _ in range(3):
                try:
                    self._page.evaluate("() => window.scrollBy(0, 800)")
                    self._page.wait_for_timeout(300)
                except Exception:
                    pass

            # Extract image URLs embedded in page HTML (new Google Images structure)
            import re
            html_content = self._page.content()
            
            # Find external image URLs in the HTML
            url_pattern = r'https?://[^"\s<>]+\.(?:jpg|jpeg|png|gif|webp)'
            all_urls = re.findall(url_pattern, html_content, re.IGNORECASE)
            
            # Filter to external (non-Google) image URLs and validate
            external_urls = []
            for url in all_urls:
                # Clean URL (remove trailing punctuation)
                url = url.rstrip('.,;:')
                # Skip Google's own assets
                if any(x in url.lower() for x in ['google.com', 'gstatic.com', 'googleapis.com']):
                    continue
                # Validate URL for HTML embedding
                if not is_valid_image_url(url):
                    continue
                if url not in seen:
                    seen.add(url)
                    external_urls.append(url)

            # Get descriptions from thumbnails
            thumbnail_descs = self._page.evaluate('''
                () => {
                    const imgs = Array.from(document.querySelectorAll("img.YQ4gaf, img.rg_i, div.F0uyec img"));
                    return imgs.map(img => img.alt || "").filter(alt => alt.length > 0);
                }
            ''') or []

            # Match URLs with descriptions
            for i, url in enumerate(external_urls[:max_results]):
                description = thumbnail_descs[i] if i < len(thumbnail_descs) else ""
                results.append({
                    "url": url,
                    "thumbnail_url": url,
                    "description": description,
                    "author": "Google Images",
                    "page_url": "",
                })

            return results if results else [{"error": "No images found"}]

        except Exception as e:
            msg = str(e)
            if "timeout" in msg.lower():
                return [{"error": "Timeout waiting for images. Google may be blocking requests."}]
            return [{"error": f"Error scraping Google Images: {msg}"}]

    def close(self) -> None:
        """Clean up browser resources for this instance."""
        try:
            if self._page:
                self._page.close()
                self._page = None
            if self._context:
                self._context.close()
                self._context = None
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
        except Exception:
            pass

    @classmethod
    def _cleanup_all(cls) -> None:
        with cls._instances_lock:
            for instance in cls._all_instances:
                try:
                    instance.close()
                except Exception:
                    pass
            cls._all_instances.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset all instances (useful for testing)."""
        cls._cleanup_all()
        if hasattr(cls._local, "instance"):
            del cls._local.instance


def search_images(query: str, max_results: int = 5) -> List[dict]:
    """Backward-compatible function API."""
    return GoogleImagesClient.get_instance().search_images(query, max_results)


if __name__ == "__main__":
    import time

    queries = [
        "chess strategy diagram",
        "python programming illustration",
        "cybersecurity lock icon",
    ]

    print("=" * 60)
    print("Google Images Search - Playwright Thread-Safe")
    print("=" * 60)

    total_start = time.time()
    for i, q in enumerate(queries, 1):
        print(f"\n🔍 [{i}/{len(queries)}] Searching: '{q}'")
        start = time.time()
        res = search_images(q, 3)
        elapsed = time.time() - start

        if res and "error" in res[0]:
            print(f"   ❌ {res[0]['error']}")
        else:
            print(f"   ✓ Found {len(res)} images in {elapsed:.2f}s")
            for img in res[:2]:
                print(f"     - {img.get('description', '')[:60]} | {img.get('url', '')[:80]}...")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"📊 Total time for {len(queries)} searches: {total_elapsed:.2f}s")
    print("=" * 60)
    GoogleImagesClient.reset()

"""Google Images search using Playwright with thread-local browser instances for thread-safety.

Uses Startpage.com as a privacy-respecting proxy to Google Images, which helps avoid
direct IP-based blocking from Google.

Implements anti-detection measures:
- Fingerprint rotation (user agent, viewport, locale) on each request
- Fresh browser context every 3-5 requests
- Human-like delays and behavior (scrolling, mouse jitter)
- Automatic CAPTCHA solving using vision LLM (Pixtral)
- Cookie persistence to avoid repeated CAPTCHA challenges
"""

import atexit
import json
import os
import random
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from urllib.parse import quote, urlparse, unquote, parse_qs

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright

# Storage paths - stored in user's home directory for persistence across runs
STORAGE_STATE_DIR = Path.home() / ".cache" / "course-generator-agent"
# Browser profile directory for persistent context (contains full browser state)
BROWSER_PROFILE_DIR = STORAGE_STATE_DIR / "browser_profile"
# Legacy storage state file (kept for backward compatibility)
STARTPAGE_STORAGE_FILE = STORAGE_STATE_DIR / "startpage_storage_state.json"
STARTPAGE_COOKIES_FILE = STARTPAGE_STORAGE_FILE


# Pool of browser fingerprints for rotation
FINGERPRINTS: List[Dict[str, Any]] = [
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        "viewport": {"width": 1920, "height": 1080},
        "locale": "en-US",
        "timezone": "America/New_York",
    },
    {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "viewport": {"width": 1366, "height": 768},
        "locale": "en-GB",
        "timezone": "Europe/London",
    },
    {
        "user_agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "viewport": {"width": 1536, "height": 864},
        "locale": "en-CA",
        "timezone": "America/Toronto",
    },
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:119.0) Gecko/20100101 Firefox/119.0",
        "viewport": {"width": 1440, "height": 900},
        "locale": "en-AU",
        "timezone": "Australia/Sydney",
    },
    {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "viewport": {"width": 1600, "height": 900},
        "locale": "en-US",
        "timezone": "America/Los_Angeles",
    },
    {
        "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "viewport": {"width": 1280, "height": 720},
        "locale": "en-IE",
        "timezone": "Europe/Dublin",
    },
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
        "viewport": {"width": 1680, "height": 1050},
        "locale": "en-NZ",
        "timezone": "Pacific/Auckland",
    },
    {
        "user_agent": "Mozilla/5.0 (Windows NT 11.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "viewport": {"width": 1920, "height": 1200},
        "locale": "en-US",
        "timezone": "America/Chicago",
    },
]


class GoogleClient:
    """
    Thread-safe Google Images client with persistent browser profile per thread.
    
    Uses Playwright's launch_persistent_context() to maintain a real browser profile
    that persists cookies, localStorage, IndexedDB, cache, and history across sessions.
    This helps avoid repeated CAPTCHA challenges after solving once.
    
    Features:
    - Persistent browser profile (survives restarts)
    - Automatic CAPTCHA solving with Pixtral vision LLM
    - Human-like delays and scrolling behavior
    - Thread-safe with thread-local browser instances
    """
    
    _local = threading.local()  # Thread-local storage for instances
    _instances_lock = threading.Lock()
    _all_instances: List["GoogleClient"] = []  # Track all instances for cleanup
    _cleanup_registered = False
    _cookies_lock = threading.Lock()  # Lock for thread-safe cookie/profile operations
    
    def __init__(self):
        """Initialize the client (use get_instance() instead)."""
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None  # Persistent context (no separate browser)
        self._page: Optional[Page] = None
        self._request_count = 0
    
    @classmethod
    def get_instance(cls) -> "GoogleClient":
        """Get or create the thread-local instance."""
        # Check if this thread already has an instance
        instance = getattr(cls._local, 'instance', None)
        
        if instance is None:
            # Create new instance for this thread
            instance = cls()
            instance._initialize()
            cls._local.instance = instance
            
            # Track instance for cleanup
            with cls._instances_lock:
                cls._all_instances.append(instance)
                
                # Register cleanup on first instance creation
                if not cls._cleanup_registered:
                    atexit.register(cls._cleanup_all)
                    cls._cleanup_registered = True
        
        return instance
    
    @classmethod
    def clear_saved_cookies(cls) -> None:
        """Clear saved browser profile and storage state (useful for troubleshooting)."""
        import shutil
        try:
            with cls._cookies_lock:
                # Remove the entire browser profile directory
                if BROWSER_PROFILE_DIR.exists():
                    shutil.rmtree(BROWSER_PROFILE_DIR)
                    print(f"🗑️ Cleared browser profile at {BROWSER_PROFILE_DIR}")
                # Also remove legacy storage state file
                if STARTPAGE_STORAGE_FILE.exists():
                    STARTPAGE_STORAGE_FILE.unlink()
                    print("🗑️ Cleared legacy storage state")
        except Exception as e:
            print(f"⚠️ Could not clear browser profile: {e}")
    
    def _initialize(self) -> None:
        """Initialize Playwright with a persistent browser context.
        
        Uses launch_persistent_context() which creates a real browser profile
        that persists all state (cookies, localStorage, IndexedDB, cache, history)
        across sessions. This helps avoid repeated CAPTCHA challenges.
        """
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        
        # Ensure profile directory exists
        BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check if profile already exists
        profile_exists = any(BROWSER_PROFILE_DIR.iterdir()) if BROWSER_PROFILE_DIR.exists() else False
        if profile_exists:
            print(f"   🍪 Loading existing browser profile from {BROWSER_PROFILE_DIR}")
        else:
            print(f"   📁 Creating new browser profile at {BROWSER_PROFILE_DIR}")
        
        # Use persistent context - this maintains cookies/storage across sessions
        # Firefox is better for anti-detection than Chromium
        self._context = self._playwright.firefox.launch_persistent_context(
            user_data_dir=str(BROWSER_PROFILE_DIR),
            headless=False,  # Non-headless to avoid detection
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
            }
        )
        
        # Get existing page or create new one
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        
        # Override navigator.webdriver via init script
        self._page.add_init_script('''
            Object.defineProperty(navigator, "webdriver", { get: () => undefined });
        ''')
        
        # Let browser stabilize
        self._page.wait_for_timeout(500)
    
    def _ensure_page_ready(self) -> None:
        """Ensure page is ready for navigation.
        
        With persistent context, we don't recreate the context - we just
        ensure we have a valid page to work with.
        """
        self._request_count += 1
        
        # If page was closed or is invalid, create a new one
        if not self._page or self._page.is_closed():
            self._page = self._context.new_page()
            self._page.add_init_script('''
                Object.defineProperty(navigator, "webdriver", { get: () => undefined });
            ''')
    
    def _add_human_behavior(self) -> None:
        """Add human-like behavior before extracting images."""
        try:
            # Random scroll to simulate looking at the page
            scroll_amount = random.randint(100, 400)
            self._page.evaluate(f'window.scrollBy(0, {scroll_amount})')
            
            # Small random delay to simulate reading
            self._page.wait_for_timeout(random.randint(200, 600))
            
            # Sometimes scroll back up a bit
            if random.random() > 0.7:
                scroll_back = random.randint(50, 150)
                self._page.evaluate(f'window.scrollBy(0, -{scroll_back})')
                self._page.wait_for_timeout(random.randint(100, 300))
            
            # Simulate mouse movement via JavaScript (random positions)
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            self._page.evaluate(f'''
                document.dispatchEvent(new MouseEvent('mousemove', {{
                    clientX: {x},
                    clientY: {y},
                    bubbles: true
                }}));
            ''')
        except Exception:
            pass  # Non-critical, continue even if this fails
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for author field."""
        try:
            parsed = urlparse(url)
            return parsed.netloc or "Unknown"
        except Exception:
            return "Unknown"
    
    def _resolve_proxy_url(self, url: str) -> str:
        """
        Convert Startpage proxy URLs to actual image URLs.
        
        Startpage uses URLs like: /av/proxy-image?piurl=https%3A%2F%2F...
        This extracts the actual image URL from the piurl parameter.
        """
        if not url:
            return url
            
        # If it's a relative startpage proxy URL, extract the real URL
        if '/proxy-image' in url or 'piurl=' in url:
            try:
                # Handle relative URLs
                if url.startswith('/'):
                    url = 'https://www.startpage.com' + url
                    
                parsed = urlparse(url)
                query_params = parse_qs(parsed.query)
                
                if 'piurl' in query_params:
                    # Get the proxied URL and decode it
                    real_url = unquote(query_params['piurl'][0])
                    return real_url
            except Exception:
                pass
        
        # If it's a relative URL, make it absolute
        if url.startswith('/'):
            return 'https://www.startpage.com' + url
            
        return url
    
    def _handle_consent_page(self) -> bool:
        """Handle Google consent/cookie dialog if present. Returns True if handled."""
        try:
            # Check for various consent button patterns
            consent_selectors = [
                'button[id="L2AGLb"]',  # "I agree" button
                'button[id="W0wltc"]',  # "Reject all" button  
                'button:has-text("Accept all")',
                'button:has-text("I agree")',
                'button:has-text("Reject all")',
                '[aria-label="Accept all"]',
                'form[action*="consent"] button',
            ]
            
            for selector in consent_selectors:
                try:
                    button = self._page.query_selector(selector)
                    if button and button.is_visible():
                        button.click()
                        self._page.wait_for_timeout(random.randint(800, 1500))
                        return True
                except Exception:
                    continue
                    
            return False
        except Exception:
            return False
    
    def _is_blocked(self) -> bool:
        """Check if the search engine is blocking us (but not simple text CAPTCHAs we can solve)."""
        try:
            page_content = self._page.content().lower()
            current_url = self._page.url.lower()
            
            # Startpage CAPTCHA page is solvable, don't treat it as blocked
            if '/sp/captcha' in current_url:
                return False
            
            # Check for blocking indicators (reCAPTCHA, etc. that we can't solve)
            blocking_indicators = [
                'unusual traffic from your computer',
                'our systems have detected unusual traffic',
                'please show you\'re not a robot',
                'recaptcha',
            ]
            
            if '/sorry/' in current_url or 'recaptcha' in current_url:
                return True
                
            for indicator in blocking_indicators:
                if indicator in page_content:
                    return True
                    
            return False
        except Exception:
            return False

    def _is_captcha_page(self) -> bool:
        """Check if we're on the Startpage CAPTCHA page."""
        try:
            current_url = self._page.url.lower()
            
            # Check URL for captcha path
            if '/sp/captcha' in current_url:
                return True
            
            # Also check for captcha image element as backup
            captcha_img = self._page.query_selector('img[alt="captcha"]')
            if captcha_img:
                return True
                
            return False
        except Exception:
            return False

    def _solve_captcha(self, max_attempts: int = 3) -> bool:
        """
        Attempt to solve the Startpage CAPTCHA using vision LLM.
        
        Args:
            max_attempts: Maximum number of attempts to solve the CAPTCHA
            
        Returns:
            True if CAPTCHA was solved successfully, False otherwise
        """
        for attempt in range(max_attempts):
            try:
                print(f"   🔐 CAPTCHA detected, solving attempt {attempt + 1}/{max_attempts}...")
                
                # Wait for CAPTCHA image to load
                self._page.wait_for_timeout(1000)
                
                # Find the CAPTCHA image
                captcha_img = self._page.query_selector('img[alt="captcha"]')
                if not captcha_img:
                    print("   ⚠️ CAPTCHA image not found, waiting...")
                    self._page.wait_for_timeout(2000)
                    captcha_img = self._page.query_selector('img[alt="captcha"]')
                    if not captcha_img:
                        print("   ❌ CAPTCHA image still not found")
                        return False
                
                # Get the CAPTCHA image URL
                captcha_url = captcha_img.get_attribute('src')
                if not captcha_url:
                    print("   ❌ Could not get CAPTCHA image URL")
                    return False
                
                # Make URL absolute if needed
                if captcha_url.startswith('/'):
                    captcha_url = 'https://www.startpage.com' + captcha_url
                
                print(f"   📸 CAPTCHA image URL: {captcha_url[:80]}...")
                
                # Use Pixtral vision LLM to read the CAPTCHA
                try:
                    from LLMs.imagetext2text import create_vision_llm
                    from langchain_core.messages import HumanMessage

                    vision_llm = create_vision_llm(provider="pixtral", temperature=0.0)
                    
                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Read the text shown in this CAPTCHA image. "
                                    "The characters are case-sensitive. "
                                    "Return ONLY the exact characters you see, nothing else. "
                                    "No quotes, no explanation, just the characters."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": captcha_url}
                            }
                        ]
                    )
                    
                    response = vision_llm.invoke([message])
                    captcha_text = response.content.strip()
                    
                    # Clean up the response (remove any quotes or extra spaces)
                    captcha_text = captcha_text.replace('"', '').replace("'", '').strip()
                    
                    print(f"   🔤 Vision LLM read: '{captcha_text}'")
                    
                    if not captcha_text or len(captcha_text) < 4:
                        print(f"   ⚠️ Invalid CAPTCHA text (too short or empty)")
                        # Try getting a new image
                        self._click_new_captcha_image()
                        continue
                        
                except Exception as e:
                    print(f"   ❌ Vision LLM error: {str(e)}")
                    # Try getting a new image and retry
                    self._click_new_captcha_image()
                    continue
                
                # Find and fill the input field
                input_field = self._page.query_selector('input[placeholder*="Enter image characters"], input[type="text"]')
                if not input_field:
                    print("   ❌ Could not find CAPTCHA input field")
                    return False
                
                # Clear any existing text and type the solution
                input_field.fill('')
                input_field.type(captcha_text, delay=50)  # Human-like typing speed
                
                self._page.wait_for_timeout(random.randint(300, 600))
                
                # Find and click the submit button
                submit_button = self._page.query_selector('button:has-text("Submit"), button[type="submit"]')
                if not submit_button:
                    print("   ❌ Could not find submit button")
                    return False
                
                submit_button.click()
                
                # Wait for navigation
                self._page.wait_for_timeout(2000)
                
                # Check if we're still on the CAPTCHA page (meaning it failed)
                if self._is_captcha_page():
                    print(f"   ⚠️ CAPTCHA solution was incorrect, trying again...")
                    # Click "Get new image" to get a fresh CAPTCHA
                    self._click_new_captcha_image()
                    continue
                else:
                    print(f"   ✅ CAPTCHA solved successfully!")
                    # Persistent context automatically saves state - no manual save needed
                    return True
                    
            except Exception as e:
                print(f"   ❌ Error solving CAPTCHA: {str(e)}")
                if attempt < max_attempts - 1:
                    self._click_new_captcha_image()
                continue
        
        print(f"   ❌ Failed to solve CAPTCHA after {max_attempts} attempts")
        return False

    def _click_new_captcha_image(self) -> None:
        """Click the 'Get new image' button to get a fresh CAPTCHA."""
        try:
            new_image_button = self._page.query_selector('button:has-text("Get new image")')
            if new_image_button:
                new_image_button.click()
                self._page.wait_for_timeout(1500)
                print("   🔄 Got new CAPTCHA image")
        except Exception:
            pass

    def _extract_startpage_images(self, max_results: int) -> List[dict]:
        """Extract images from Startpage image search results."""
        results = []
        seen_urls = set()
        
        try:
            # Startpage uses image cards with data attributes
            image_elements = self._page.query_selector_all('.image-container img, .image-result img, img[src*="proxy"]')
            
            for img in image_elements:
                if len(results) >= max_results:
                    break
                    
                try:
                    src = img.get_attribute('src') or img.get_attribute('data-src')
                    if not src:
                        continue
                    
                    # Skip base64 placeholders
                    if src.startswith('data:'):
                        continue
                    
                    # Resolve proxy URL to get actual image URL
                    actual_url = self._resolve_proxy_url(src)
                    
                    # Validate URL for HTML embedding
                    if not is_valid_image_url(actual_url):
                        continue
                    
                    # Skip duplicates
                    if actual_url in seen_urls:
                        continue
                    seen_urls.add(actual_url)
                    
                    alt = img.get_attribute('alt') or "No description"
                    
                    # Try to get parent link
                    page_url = ""
                    try:
                        page_url = img.evaluate('el => el.closest("a")?.href || ""')
                    except Exception:
                        pass
                    
                    results.append({
                        "url": actual_url,
                        "thumbnail_url": actual_url,
                        "description": alt,
                        "author": self._extract_domain(actual_url),
                        "page_url": page_url
                    })
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        # Fallback: try generic img extraction
        if not results:
            try:
                all_imgs = self._page.query_selector_all('img[src]')
                for img in all_imgs:
                    if len(results) >= max_results:
                        break
                    try:
                        src = img.get_attribute('src')
                        if not src or src.startswith('data:'):
                            continue
                        # Skip very small images (likely icons)
                        width = img.get_attribute('width')
                        height = img.get_attribute('height')
                        if width and height:
                            try:
                                if int(width) < 50 or int(height) < 50:
                                    continue
                            except ValueError:
                                pass
                        
                        # Skip startpage assets
                        if 'startpage.com' in src and '/assets/' in src:
                            continue
                        
                        # Resolve proxy URL
                        actual_url = self._resolve_proxy_url(src)
                        
                        # Validate URL for HTML embedding
                        if not is_valid_image_url(actual_url):
                            continue
                        
                        if actual_url in seen_urls:
                            continue
                        seen_urls.add(actual_url)
                            
                        alt = img.get_attribute('alt') or "Image"
                        results.append({
                            "url": actual_url,
                            "thumbnail_url": actual_url,
                            "description": alt,
                            "author": self._extract_domain(actual_url),
                            "page_url": ""
                        })
                    except Exception:
                        continue
            except Exception:
                pass
                
        return results

    def search_images(self, query: str, max_results: int = 5) -> List[dict]:
        """
        Search for images using Startpage (Google Images proxy).
        
        Startpage.com is a privacy-focused search engine that uses Google's
        results, helping avoid direct IP blocking from Google.
        
        Features:
        - Automatic CAPTCHA solving using Pixtral vision LLM
        - Fingerprint rotation for anti-detection
        - Human-like browsing behavior
        
        Args:
            query: Search query for images
            max_results: Maximum number of images to return (default: 5)
            
        Returns:
            List of image results with URLs and metadata
        """
        try:
            # Ensure we have a valid page (persistent context handles state)
            self._ensure_page_ready()
            
            # Use Startpage image search (proxies Google)
            search_url = f"https://www.startpage.com/sp/search?query={quote(query)}&cat=images"
            
            # Pre-navigation delay to appear more human-like
            self._page.wait_for_timeout(random.randint(500, 1500))
            
            # Navigate with timeout
            self._page.goto(search_url, wait_until='domcontentloaded', timeout=25000)
            
            # Post-navigation delay
            self._page.wait_for_timeout(random.randint(1000, 2000))
            
            # Handle any consent/cookie dialogs
            self._handle_consent_page()
            
            # Check for CAPTCHA page and attempt to solve it
            if self._is_captcha_page():
                if not self._solve_captcha(max_attempts=3):
                    return [{"error": "Unable to solve CAPTCHA after multiple attempts. Please try again later."}]
                # After solving CAPTCHA, wait a bit and check if we need to navigate again
                self._page.wait_for_timeout(1000)
                # If we're still not on the results page, navigate again
                if 'cat=images' not in self._page.url and '/sp/search' not in self._page.url:
                    self._page.goto(search_url, wait_until='domcontentloaded', timeout=25000)
                    self._page.wait_for_timeout(random.randint(1500, 2500))
            
            # Check for blocking (reCAPTCHA, etc. that we can't solve)
            if self._is_blocked():
                return [{"error": "Search engine blocking detected (reCAPTCHA). Please try again later."}]
            
            # Wait for images to load
            try:
                self._page.wait_for_selector('img[src]', timeout=10000, state='attached')
            except Exception:
                pass
            
            # Add human-like behavior (scrolling, mouse movement)
            self._add_human_behavior()
            
            # Additional variable wait for dynamic content
            self._page.wait_for_timeout(random.randint(400, 800))
            
            # Extract images
            results = self._extract_startpage_images(max_results)
            
            if not results:
                # Try scrolling more to trigger lazy loading
                self._page.evaluate(f'window.scrollBy(0, {random.randint(500, 1000)})')
                self._page.wait_for_timeout(random.randint(600, 1200))
                results = self._extract_startpage_images(max_results)
            
            return results if results else [{"error": "No images found for query"}]
            
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower():
                return [{"error": "Timeout waiting for images. The service may be slow."}]
            return [{"error": f"Error searching for images: {error_msg}"}]
    
    def close(self) -> None:
        """Clean up browser resources for this instance.
        
        Note: With persistent context, closing saves all state (cookies, localStorage, etc.)
        to the profile directory automatically.
        """
        try:
            if self._page:
                self._page.close()
                self._page = None
            if self._context:
                self._context.close()  # Saves persistent state automatically
                self._context = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
        except Exception:
            pass
    
    @classmethod
    def _cleanup_all(cls) -> None:
        """Clean up all thread-local instances (called on program exit)."""
        with cls._instances_lock:
            for instance in cls._all_instances:
                try:
                    instance.close()
                except Exception:
                    pass
            cls._all_instances.clear()
    
    @classmethod
    def reset(cls) -> None:
        """Reset all instances (useful for testing)."""
        cls._cleanup_all()
        # Clear thread-local storage for current thread
        if hasattr(cls._local, 'instance'):
            del cls._local.instance


# Backward-compatible function API
def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Google Images using direct web scraping.
    
    Uses GoogleImagesClient which scrapes google.com/search?tbm=isch directly,
    avoiding Startpage proxy and its aggressive CAPTCHA challenges.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    return GoogleImagesClient.get_instance().search_images(query, max_results)


if __name__ == "__main__":
    import sys
    import time
    
    # Test script with queries - uses direct Google Images (not Startpage)
    queries = [
        "chess strategy", 
        "python programming",
        "artificial intelligence",
    ]
    
    print("=" * 70)
    print("Google Images Search - Direct Google (no Startpage)")
    print("=" * 70)
    print(f"Testing {len(queries)} queries")
    print("Consent dialogs will be auto-accepted")
    print("=" * 70)
    
    total_start = time.time()
    successful = 0
    blocked = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 [{i}/{len(queries)}] Searching: '{query}'")
        
        start = time.time()
        results = search_images(query, 3)
        elapsed = time.time() - start
        
        if results and "error" in results[0]:
            error_msg = results[0]['error']
            print(f"   ❌ {error_msg}")
            if "blocking" in error_msg.lower() or "captcha" in error_msg.lower():
                blocked += 1
        else:
            print(f"   ✓ Found {len(results)} images in {elapsed:.2f}s")
            successful += 1
            for img in results[:2]:
                desc = img.get('description', 'No description')
                if len(desc) > 50:
                    desc = desc[:50] + "..."
                print(f"     - {desc}")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print(f"📊 Results Summary:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Successful: {successful}")
    print(f"   Blocked: {blocked}")
    print(f"   Success rate: {successful/len(queries)*100:.1f}%")
    print(f"   Total time: {total_elapsed:.2f}s")
    print(f"   Average per search: {total_elapsed/len(queries):.2f}s")
    print("=" * 70)
    
    # Cleanup
    GoogleImagesClient.reset()
    print("\n✅ Browser closed")
