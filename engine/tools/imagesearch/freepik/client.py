"""Freepik image search using Playwright with thread-local browser instances for thread-safety.

Playwright and playwright_stealth are imported lazily inside the methods
that use them, so that merely importing this module does not require
those heavy browser-automation packages.
"""

import atexit
import contextlib
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright


class FreepikClient:
    """
    Thread-safe Freepik client with persistent browser per thread.

    Each thread gets its own browser instance stored in thread-local storage,
    ensuring thread-safety while maintaining browser reuse within each thread.
    """

    _local = threading.local()  # Thread-local storage for instances
    _instances_lock = threading.Lock()
    _all_instances: list["FreepikClient"] = []  # Track all instances for cleanup
    _cleanup_registered = False

    def __init__(self):
        """Initialize the client (use get_instance() instead)."""
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._request_count = 0
        self._max_requests_before_refresh = 100  # Refresh context to avoid memory issues

    @classmethod
    def get_instance(cls) -> "FreepikClient":
        """Get or create the thread-local instance."""
        # Check if this thread already has an instance
        instance = getattr(cls._local, "instance", None)

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

    def _initialize(self) -> None:
        """Initialize Playwright browser with anti-detection settings."""
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()

        # Launch browser with anti-detection options
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
        """Create a new browser context with realistic settings."""
        if self._context:
            with contextlib.suppress(Exception):
                self._context.close()

        self._context = self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
        )

        self._page = self._context.new_page()

        # Apply stealth patches to avoid detection
        from playwright_stealth import Stealth

        Stealth(navigator_platform_override="MacIntel").apply_stealth_sync(self._page)

        # Additional CDP-based anti-detection
        try:
            client = self._context.new_cdp_session(self._page)
            client.send(
                "Network.setUserAgentOverride",
                {
                    "userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "platform": "MacIntel",
                },
            )
        except Exception:
            pass  # CDP might not be available in all configurations

        # Override navigator.webdriver via init script
        self._page.add_init_script("""
            Object.defineProperty(navigator, "webdriver", { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

        self._request_count = 0

    def _ensure_fresh_context(self) -> None:
        """Refresh context periodically to prevent memory issues."""
        self._request_count += 1
        if self._request_count >= self._max_requests_before_refresh:
            self._create_context()

    _BLOCK_TITLES = ["access denied", "attention required", "just a moment", "blocked", "error", "forbidden"]
    _BLOCK_BODY_KEYWORDS = [
        "access denied",
        "captcha",
        "blocked",
        "rate limit",
        "please verify",
        "unusual traffic",
        "security check",
    ]

    def _detect_block(self, response) -> tuple[bool, str]:
        """Check if Freepik/Akamai served a block page instead of real results.

        Inspects the HTTP status, page title, and visible body text for
        common anti-bot / WAF indicators.
        """
        if response and response.status == 403:
            return True, "HTTP 403 Forbidden"

        try:
            title = self._page.title().lower()
        except Exception:
            title = ""
        for bt in self._BLOCK_TITLES:
            if bt in title:
                return True, f"Block page title: '{self._page.title()}'"

        try:
            body_text = self._page.inner_text("body")[:500].lower()
        except Exception:
            body_text = ""
        for kw in self._BLOCK_BODY_KEYWORDS:
            if kw in body_text:
                return True, f"Block keyword in body: '{kw}'"

        return False, ""

    def _search_once(self, query: str, max_results: int) -> tuple[list[dict], bool, str]:
        """Execute a single search attempt and return results with block info.

        Returns:
            (results, is_blocked, block_reason) where *results* is an empty
            list when no valid images were found.
        """
        try:
            self._ensure_fresh_context()

            search_url = f"https://www.freepik.com/search?format=search&query={quote(query)}"

            response = self._page.goto(search_url, wait_until="domcontentloaded", timeout=15000)

            with contextlib.suppress(Exception):
                self._page.wait_for_selector("figure img[src]", timeout=5000, state="attached")

            self._page.wait_for_timeout(200)

            results: list[dict] = []
            image_elements = self._page.query_selector_all("figure img[src], img[data-src]")

            for img_element in image_elements[: max_results * 2]:
                try:
                    image_url = img_element.get_attribute("src") or img_element.get_attribute("data-src")
                    if not image_url or "placeholder" in image_url.lower() or "loading" in image_url.lower():
                        continue

                    description = img_element.get_attribute("alt") or "No description"
                    page_url = ""
                    with contextlib.suppress(Exception):
                        page_url = img_element.evaluate('el => el.closest("a")?.href || ""')

                    results.append(
                        {
                            "url": image_url,
                            "thumbnail_url": image_url,
                            "description": description,
                            "author": "Freepik",
                            "page_url": page_url,
                        }
                    )
                    if len(results) >= max_results:
                        break
                except Exception:
                    continue

            if results:
                return results, False, ""

            is_blocked, reason = self._detect_block(response)
            return [], is_blocked, reason if is_blocked else "No images found"

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return [], True, "Navigation timeout (possible block)"
            return [], False, f"Error scraping Freepik: {error_msg}"

    def search_images(self, query: str, max_results: int = 5) -> list[dict]:
        """Search for images on Freepik with automatic block detection and retry.

        When a block/rate-limit is detected (HTTP 403, challenge page, or
        zero results with block signals), the client refreshes its browser
        context and retries with exponential backoff (2 s, 4 s, 8 s).

        Args:
            query: Search query for images
            max_results: Maximum number of images to return (default: 5)

        Returns:
            List of image results with URLs and metadata.  On persistent
            blocking the list contains a single dict with ``"error"`` and
            ``"blocked": True``.
        """
        max_retries = 3
        base_delay = 2.0
        last_reason = ""

        for attempt in range(1 + max_retries):
            results, is_blocked, reason = self._search_once(query, max_results)

            if results:
                if attempt > 0:
                    print(f"   ✅ Freepik retry {attempt} succeeded for '{query}'")
                return results

            last_reason = reason

            # Not blocked and first attempt → genuine "no images"
            if not is_blocked and attempt == 0:
                return [{"error": "No images found"}]

            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                print(f"   ⚠️ Freepik block detected ({reason}), retry {attempt + 1}/{max_retries} in {delay:.0f}s…")
                self._create_context()
                time.sleep(delay)

        return [{"error": f"Blocked by Freepik after {max_retries} retries: {last_reason}", "blocked": True}]

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
        """Clean up all thread-local instances (called on program exit)."""
        with cls._instances_lock:
            for instance in cls._all_instances:
                with contextlib.suppress(Exception):
                    instance.close()
            cls._all_instances.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset all instances (useful for testing)."""
        cls._cleanup_all()
        # Clear thread-local storage for current thread
        if hasattr(cls._local, "instance"):
            del cls._local.instance


# Backward-compatible function API
def search_images(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for images on Freepik using web scraping.

    This function maintains backward compatibility with the original API
    while using the optimized thread-local client internally.

    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)

    Returns:
        List of image results with URLs and metadata
    """
    return FreepikClient.get_instance().search_images(query, max_results)


if __name__ == "__main__":
    import time

    # Test script with benchmark
    queries = ["hacker", "programming", "artificial intelligence", "chess strategy", "cybersecurity"]

    print("=" * 60)
    print("Freepik Image Search - Playwright Thread-Safe")
    print("=" * 60)

    total_start = time.time()

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 [{i}/{len(queries)}] Searching: '{query}'")

        start = time.time()
        results = search_images(query, 3)
        elapsed = time.time() - start

        if results and "error" in results[0]:
            print(f"   ❌ {results[0]['error']}")
        else:
            print(f"   ✓ Found {len(results)} images in {elapsed:.2f}s")
            for img in results[:2]:
                print(f"     - {img.get('description', 'No description')[:50]}...")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print(f"📊 Total time for {len(queries)} searches: {total_elapsed:.2f}s")
    print(f"   Average per search: {total_elapsed / len(queries):.2f}s")
    print("   (First search includes browser startup ~3s)")
    print("=" * 60)

    # Cleanup
    FreepikClient.reset()
    print("\n✅ Browser closed successfully")
