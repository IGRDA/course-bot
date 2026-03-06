"""
Extract content images from web pages by parsing raw HTML.

Covers four image delivery methods that WebFetch's markdown converter misses:
1. Standard <img src="..."> tags
2. Lazy-loaded images via data-src / data-lazy-src attributes
3. Responsive <picture><source srcset="..."> elements
4. Inline CSS background-image: url(...) styles

Falls back to a headless Playwright browser when the site blocks
requests/curl (e.g. CAPTCHA, bot protection, Cloudflare, SiteGround).

Usage:
    python -m tools.web_image_extractor URL [URL ...]

Outputs JSON to stdout. Errors/warnings go to stderr.
"""

from __future__ import annotations

import atexit
import json
import re
import subprocess
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

_TIMEOUT = 15

_FILTER_URL_SUBSTRINGS = {
    "favicon", "pixel", "spacer", "blank",
    "transparent", "1x1", "spinner", "loader",
    "doubleclick", "google-analytics", "facebook.com/tr",
}

_FILTER_EXTENSIONS = {".svg"}

_MIN_DIMENSION = 30

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}

_BG_IMAGE_RE = re.compile(r"background-image:\s*url\(\s*['\"]?([^'\")\s]+)['\"]?\s*\)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _is_filtered(src: str, tag: Tag | None = None) -> bool:
    """Return True if the image should be excluded as non-content."""
    if not src or src.startswith("data:"):
        return True

    lower = src.lower()

    parsed = urlparse(lower)
    ext = parsed.path.rsplit(".", 1)[-1] if "." in parsed.path else ""
    if f".{ext}" in _FILTER_EXTENSIONS:
        return True

    for substring in _FILTER_URL_SUBSTRINGS:
        if substring in lower:
            return True

    filename = parsed.path.rsplit("/", 1)[-1] if "/" in parsed.path else parsed.path
    if filename.startswith("logo"):
        return True

    if tag and isinstance(tag, Tag):
        try:
            w = int(tag.get("width", 0) or 0)
            h = int(tag.get("height", 0) or 0)
            if (w and w < _MIN_DIMENSION) or (h and h < _MIN_DIMENSION):
                return True
        except (ValueError, TypeError):
            pass

    return False


def _normalize_url(src: str, page_url: str) -> str:
    """Resolve relative URLs and strip fragments."""
    resolved = urljoin(page_url, src)
    parsed = urlparse(resolved)
    return parsed._replace(fragment="").geturl()


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def _find_preceding_heading(element: Tag) -> str:
    """Walk backwards in the DOM to find the nearest heading text."""
    node = element
    while node:
        for sibling in _previous_siblings_and_parents(node):
            if isinstance(sibling, Tag) and sibling.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                return sibling.get_text(strip=True)
        node = node.parent if node.parent and node.parent.name != "[document]" else None
    return ""


def _previous_siblings_and_parents(tag: Tag):
    """Yield previous siblings, then move to parent and repeat."""
    current = tag.previous_sibling
    while current:
        if isinstance(current, Tag):
            yield current
            for desc in reversed(list(current.descendants)):
                if isinstance(desc, Tag) and desc.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    yield desc
        current = current.previous_sibling


def _find_context_text(element: Tag) -> str:
    """Get text from the nearest meaningful parent container."""
    containers = ("section", "article", "main", "div")
    node = element.parent
    while node:
        if isinstance(node, Tag) and node.name in containers:
            text = node.get_text(separator=" ", strip=True)
            if len(text) > 50:
                return text[:200]
        node = node.parent
    return ""


# ---------------------------------------------------------------------------
# Extraction methods
# ---------------------------------------------------------------------------

def _extract_img_tags(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """Method 1 & 2: <img src> and <img data-src> / data-lazy-src."""
    results = []
    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-original")
        )
        if not src:
            continue

        src = _normalize_url(src.strip(), page_url)
        if _is_filtered(src, img):
            continue

        alt = img.get("alt", "").strip()
        results.append({
            "src": src,
            "alt": alt,
            "_element": img,
        })
    return results


def _extract_picture_sources(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """Method 3: <picture><source srcset> -- pick the largest variant."""
    results = []
    for picture in soup.find_all("picture"):
        sources = picture.find_all("source")
        img_fallback = picture.find("img")

        best_src = None
        best_width = 0

        for source in sources:
            srcset = source.get("srcset", "")
            for entry in srcset.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                parts = entry.split()
                url = parts[0]
                width = 0
                if len(parts) > 1 and parts[1].endswith("w"):
                    try:
                        width = int(parts[1][:-1])
                    except ValueError:
                        pass
                if width > best_width:
                    best_width = width
                    best_src = url

        if not best_src and img_fallback:
            best_src = img_fallback.get("src")

        if not best_src:
            continue

        best_src = _normalize_url(best_src.strip(), page_url)
        if _is_filtered(best_src):
            continue

        alt = ""
        if img_fallback:
            alt = img_fallback.get("alt", "").strip()

        results.append({
            "src": best_src,
            "alt": alt,
            "_element": picture,
        })
    return results


def _extract_background_images(soup: BeautifulSoup, page_url: str) -> list[dict]:
    """Method 4: inline style background-image: url(...)."""
    results = []
    for tag in soup.find_all(style=True):
        style = tag.get("style", "")
        for match in _BG_IMAGE_RE.finditer(style):
            src = _normalize_url(match.group(1).strip(), page_url)
            if _is_filtered(src, tag):
                continue
            results.append({
                "src": src,
                "alt": "",
                "_element": tag,
            })
    return results


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

_MIN_VALID_HTML_LENGTH = 500
_CAPTCHA_MARKERS = ("sgcaptcha", "cf-browser-verification", "challenge-platform", "captcha")


def _html_looks_valid(html: str) -> bool:
    """Return False if the HTML is a CAPTCHA/bot-gate page or too short to contain real content."""
    if len(html) < _MIN_VALID_HTML_LENGTH:
        return False
    lower = html[:4000].lower()
    if any(marker in lower for marker in _CAPTCHA_MARKERS):
        return False
    # Detect pages that are mostly an SVG loading animation (< 10 div tags)
    if html.count("<div") < 10 and html.count("<polygon") > 50:
        return False
    return True


# ---------------------------------------------------------------------------
# Playwright browser singleton (lazy-initialised, reused across URLs)
# ---------------------------------------------------------------------------

_pw_instance = None
_pw_browser = None


def _get_playwright_browser():
    """Return a reusable Playwright browser instance, launching one if needed."""
    global _pw_instance, _pw_browser
    if _pw_browser is not None:
        return _pw_browser
    try:
        from playwright.sync_api import sync_playwright
        from playwright_stealth import Stealth  # noqa: F401 – verify import
    except ImportError:
        print("[web_image_extractor] playwright or playwright-stealth not installed", file=sys.stderr)
        return None

    _pw_instance = sync_playwright().start()
    _pw_browser = _pw_instance.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )
    atexit.register(_cleanup_playwright)
    return _pw_browser


def _cleanup_playwright():
    global _pw_instance, _pw_browser
    try:
        if _pw_browser:
            _pw_browser.close()
        if _pw_instance:
            _pw_instance.stop()
    except Exception:
        pass
    _pw_browser = None
    _pw_instance = None


def _fetch_html_playwright(url: str, timeout: int = _TIMEOUT) -> str | None:
    """Fetch fully-rendered HTML via a headless Chromium browser.

    Handles multi-stage loads common with bot-protection (CAPTCHA redirects,
    loading animations, lazy JS content injection) by:
      1. Navigating and waiting for domcontentloaded
      2. Sleeping to let CAPTCHA/redirect chains settle
      3. Waiting for the page to stabilise (load event)
      4. Scrolling to trigger lazy-loaded images
    """
    browser = _get_playwright_browser()
    if browser is None:
        return None
    context = None
    try:
        from playwright_stealth import Stealth
        stealth = Stealth()
        context = browser.new_context(
            user_agent=_HEADERS["User-Agent"],
            locale="es-ES",
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()
        stealth.apply_stealth_sync(page)

        page.goto(url, wait_until="domcontentloaded", timeout=timeout * 2 * 1000)
        # Allow CAPTCHA/redirect chains to complete
        page.wait_for_timeout(5000)

        # After redirects settle, wait for the real page to finish loading
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(3000)

        # Scroll to trigger lazy-loaded images
        try:
            page.evaluate("""
                async () => {
                    const delay = ms => new Promise(r => setTimeout(r, ms));
                    for (let i = 0; i < document.body.scrollHeight; i += 400) {
                        window.scrollTo(0, i);
                        await delay(100);
                    }
                    window.scrollTo(0, 0);
                }
            """)
            page.wait_for_timeout(2000)
        except Exception:
            pass

        html = page.content()
        page.close()
        context.close()
        return html
    except Exception as exc:
        print(f"[web_image_extractor] Playwright failed for {url}: {exc}", file=sys.stderr)
        if context:
            try:
                context.close()
            except Exception:
                pass
        return None


def _fetch_html(url: str, timeout: int = _TIMEOUT) -> str | None:
    """Fetch HTML with a three-tier fallback: requests -> curl -> Playwright."""
    # --- Tier 1: requests (fastest) ---
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        if _html_looks_valid(resp.text):
            return resp.text
        print(f"[web_image_extractor] requests returned bot-gate page for {url}, trying curl", file=sys.stderr)
    except requests.RequestException:
        pass

    # --- Tier 2: curl ---
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode == 0 and _html_looks_valid(result.stdout):
            return result.stdout
        print(f"[web_image_extractor] curl returned bot-gate page for {url}, trying Playwright", file=sys.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # --- Tier 3: headless Playwright browser ---
    print(f"[web_image_extractor] Falling back to Playwright for {url} ...", file=sys.stderr)
    html = _fetch_html_playwright(url, timeout)
    if html and _html_looks_valid(html):
        return html

    print(f"[web_image_extractor] All methods failed for {url}", file=sys.stderr)
    return None


def extract_images(url: str, timeout: int = _TIMEOUT) -> dict:
    """Fetch a URL and extract all content images with context.

    Returns::

        {
            "page_url": str,
            "images": [{"src": str, "alt": str, "context_heading": str, "context_text": str}, ...]
        }
    """
    html = _fetch_html(url, timeout)
    if not html:
        return {"page_url": url, "images": []}

    soup = BeautifulSoup(html, "html.parser")

    raw_images = []
    raw_images.extend(_extract_img_tags(soup, url))
    raw_images.extend(_extract_picture_sources(soup, url))
    raw_images.extend(_extract_background_images(soup, url))

    seen_urls: set[str] = set()
    images = []
    for entry in raw_images:
        src = entry["src"]
        if src in seen_urls:
            continue
        seen_urls.add(src)

        element = entry.pop("_element", None)
        heading = _find_preceding_heading(element) if element else ""
        context = _find_context_text(element) if element else ""

        images.append({
            "src": src,
            "alt": entry["alt"],
            "context_heading": heading,
            "context_text": context,
        })

    return {"page_url": url, "images": images}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tools.web_image_extractor URL [URL ...]", file=sys.stderr)
        sys.exit(1)

    urls = sys.argv[1:]
    results = []
    for url in urls:
        print(f"[web_image_extractor] Extracting images from {url} ...", file=sys.stderr)
        result = extract_images(url)
        n = len(result["images"])
        print(f"[web_image_extractor]   Found {n} content images", file=sys.stderr)
        results.append(result)

    json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()
