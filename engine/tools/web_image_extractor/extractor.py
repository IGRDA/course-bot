"""
Extract content images from web pages by parsing raw HTML.

Covers four image delivery methods that WebFetch's markdown converter misses:
1. Standard <img src="..."> tags
2. Lazy-loaded images via data-src / data-lazy-src attributes
3. Responsive <picture><source srcset="..."> elements
4. Inline CSS background-image: url(...) styles

Falls back to a headless Playwright browser (with stealth patches) when the
site blocks requests/curl (e.g. CAPTCHA, bot protection, Cloudflare, SiteGround).

Usage:
    python -m tools.web_image_extractor URL [URL ...]

Outputs JSON to stdout. Errors/warnings go to stderr.

Public API:
    extract_images(url) -> dict with page_url and images list
    extract_images_from_urls([url, ...]) -> list of dicts
"""

from __future__ import annotations

import atexit
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

_TIMEOUT = 15
_INTER_PAGE_DELAY = 2.0

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

_BG_IMAGE_RE = re.compile(
    r"background-image:\s*url\(\s*['\"]?([^'\")\s]+)['\"]?\s*\)",
    re.IGNORECASE,
)


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
# HTML fetching with fallbacks
# ---------------------------------------------------------------------------

_MIN_VALID_HTML_LENGTH = 500
_CAPTCHA_MARKERS = ("sgcaptcha", "cf-browser-verification", "challenge-platform", "captcha")


def _html_looks_valid(html: str) -> bool:
    """Return False if the HTML is a CAPTCHA/bot-gate page or too short."""
    if len(html) < _MIN_VALID_HTML_LENGTH:
        return False
    lower = html[:4000].lower()
    if any(marker in lower for marker in _CAPTCHA_MARKERS):
        return False
    if html.count("<div") < 10 and html.count("<polygon") > 50:
        return False
    return True


# ---------------------------------------------------------------------------
# Playwright browser singleton (lazy-initialised, reused across URLs)
# ---------------------------------------------------------------------------

_pw_instance = None
_pw_browser = None
_pw_context = None


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


def _get_playwright_context():
    """Return a reusable Playwright browser context that persists cookies across pages."""
    global _pw_context
    if _pw_context is not None:
        return _pw_context
    browser = _get_playwright_browser()
    if browser is None:
        return None
    from playwright_stealth import Stealth
    _pw_context = browser.new_context(
        user_agent=_HEADERS["User-Agent"],
        locale="es-ES",
        viewport={"width": 1440, "height": 900},
    )
    stealth = Stealth()
    stealth.apply_stealth_sync(_pw_context)
    return _pw_context


def _cleanup_playwright():
    global _pw_instance, _pw_browser, _pw_context
    try:
        if _pw_context:
            _pw_context.close()
        if _pw_browser:
            _pw_browser.close()
        if _pw_instance:
            _pw_instance.stop()
    except Exception:
        pass
    _pw_context = None
    _pw_browser = None
    _pw_instance = None


_SG_CAPTCHA_WAIT = 10000
_POST_CAPTCHA_SETTLE = 5000


def _fetch_html_playwright(url: str, timeout: int = _TIMEOUT) -> str | None:
    """Fetch fully-rendered HTML via a headless Chromium browser with stealth.

    Uses a persistent browser context so that cookies set during the first
    CAPTCHA challenge (e.g. SiteGround) carry over to subsequent page loads.
    Waits longer for CAPTCHA interstitials to resolve automatically.
    """
    context = _get_playwright_context()
    if context is None:
        return None
    page = None
    try:
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=timeout * 2 * 1000)
        page.wait_for_timeout(_SG_CAPTCHA_WAIT)

        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        page.wait_for_timeout(_POST_CAPTCHA_SETTLE)

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
        return html
    except Exception as exc:
        print(f"[web_image_extractor] Playwright failed for {url}: {exc}", file=sys.stderr)
        if page:
            try:
                page.close()
            except Exception:
                pass
        return None


def _fetch_html(url: str, timeout: int = _TIMEOUT) -> str | None:
    """Fetch HTML with fallback: requests -> curl -> Playwright (with stealth)."""
    # Tier 1: requests (fastest)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        if _html_looks_valid(resp.text):
            return resp.text
        print(f"[web_image_extractor] requests got bot-gate for {url}, trying curl", file=sys.stderr)
    except requests.RequestException:
        pass

    # Tier 2: curl
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode == 0 and _html_looks_valid(result.stdout):
            return result.stdout
        print(f"[web_image_extractor] curl got bot-gate for {url}, trying Playwright", file=sys.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Tier 3: headless Playwright with stealth patches
    print(f"[web_image_extractor] Falling back to Playwright for {url} ...", file=sys.stderr)
    html = _fetch_html_playwright(url, timeout)
    if html and _html_looks_valid(html):
        return html

    print(f"[web_image_extractor] All methods failed for {url}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

_IMAGE_SIGNATURES = {
    b'\x89PNG\r\n\x1a\n': '.png',
    b'\xff\xd8\xff': '.jpg',
    b'GIF87a': '.gif',
    b'GIF89a': '.gif',
    b'RIFF': '.webp',  # RIFF....WEBP
    b'%PDF': '.pdf',
}

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def _detect_image_ext(data: bytes) -> str | None:
    """Return file extension if *data* starts with a known image signature."""
    for sig, ext in _IMAGE_SIGNATURES.items():
        if data[:len(sig)] == sig:
            if sig == b'RIFF' and data[8:12] != b'WEBP':
                return None
            return ext
    return None


def _download_image(src: str, output_dir: Path, use_playwright: bool = False) -> Path | None:
    """Download a single image to *output_dir* with tiered fallback.

    Tier 1: ``requests.get`` with browser-like headers (fast, works for CDNs).
    Tier 2: Playwright ``context.request.get`` sharing stealth cookies.

    Returns the destination ``Path`` on success, ``None`` on failure.
    """
    url_hash = hashlib.md5(src.encode()).hexdigest()[:12]
    parsed = urlparse(src)
    url_ext = Path(parsed.path).suffix.lower()
    if url_ext not in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf'):
        url_ext = '.jpg'

    filename = f"img_{url_hash}{url_ext}"
    dest = output_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    body: bytes | None = None

    # Tier 1: requests
    try:
        resp = requests.get(src, headers=_HEADERS, timeout=_TIMEOUT, stream=True)
        resp.raise_for_status()
        body = resp.content
    except Exception:
        pass

    if body and _detect_image_ext(body) is None:
        body = None

    # Tier 2: Playwright context API (shares cookies from CAPTCHA bypass)
    if body is None and use_playwright:
        context = _get_playwright_context()
        if context is not None:
            try:
                api_resp = context.request.get(src, timeout=_TIMEOUT * 1000)
                if api_resp.ok:
                    body = api_resp.body()
                    if _detect_image_ext(body) is None:
                        body = None
            except Exception as exc:
                print(f"[web_image_extractor] Playwright download failed for {src}: {exc}",
                      file=sys.stderr)

    if body is None or len(body) > _MAX_IMAGE_BYTES:
        return None

    real_ext = _detect_image_ext(body)
    if real_ext and real_ext != url_ext:
        filename = f"img_{url_hash}{real_ext}"
        dest = output_dir / filename

    dest.write_bytes(body)
    return dest


def _download_extracted_images(
    images: list[dict],
    output_dir: Path,
    use_playwright: bool,
) -> None:
    """Download all extracted images and annotate each dict with ``local_path``.

    ``local_path`` is set relative to *output_dir*'s parent so markdown files
    in sibling locations can reference ``images/img_xxx.jpg``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rel_base = output_dir.parent
    downloaded = 0

    for img in images:
        dest = _download_image(img["src"], output_dir, use_playwright=use_playwright)
        if dest is not None:
            try:
                img["local_path"] = str(dest.relative_to(rel_base))
            except ValueError:
                img["local_path"] = str(dest)
            downloaded += 1
        else:
            img["local_path"] = None

    print(f"[web_image_extractor]   Downloaded {downloaded}/{len(images)} images",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_images(url: str, timeout: int = _TIMEOUT, output_dir: str | None = None) -> dict:
    """Fetch a URL and extract all content images with context.

    When *output_dir* is given the images are also downloaded to that
    directory and each entry gets a ``local_path`` key (relative to the
    parent of *output_dir*) so markdown files can reference them as
    ``images/img_xxx.jpg``.

    Returns::

        {
            "page_url": str,
            "images": [{
                "src": str, "alt": str,
                "context_heading": str, "context_text": str,
                "local_path": str | None   # only when output_dir is set
            }, ...]
        }
    """
    html = _fetch_html(url, timeout)
    if not html:
        return {"page_url": url, "images": []}

    pw_was_used = _pw_context is not None

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

    if output_dir and images:
        _download_extracted_images(images, Path(output_dir), use_playwright=pw_was_used)

    return {"page_url": url, "images": images}


def extract_images_from_urls(
    urls: list[str],
    timeout: int = _TIMEOUT,
    output_dir: str | None = None,
) -> list[dict]:
    """Extract images from multiple URLs with inter-page delays to avoid rate-limiting.

    When *output_dir* is given, all images are downloaded to that single
    shared directory (de-duplicated by URL hash).
    """
    results = []
    for i, url in enumerate(urls):
        if i > 0:
            print(f"[web_image_extractor] Waiting {_INTER_PAGE_DELAY}s between pages ...", file=sys.stderr)
            time.sleep(_INTER_PAGE_DELAY)
        print(f"[web_image_extractor] Extracting from {url} ...", file=sys.stderr)
        result = extract_images(url, timeout, output_dir=output_dir)
        print(f"[web_image_extractor]   Found {len(result['images'])} images", file=sys.stderr)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    raw_args = sys.argv[1:]

    output_dir: str | None = None
    if "--output-dir" in raw_args:
        idx = raw_args.index("--output-dir")
        if idx + 1 < len(raw_args):
            output_dir = raw_args[idx + 1]
            raw_args = raw_args[:idx] + raw_args[idx + 2:]
        else:
            print("Error: --output-dir requires a path argument", file=sys.stderr)
            sys.exit(1)

    urls = [arg for arg in raw_args if arg.startswith(("http://", "https://"))]

    if not urls:
        print("Usage: python -m tools.web_image_extractor [--output-dir DIR] URL [URL ...]", file=sys.stderr)
        print("Outputs JSON to stdout with image URLs, alt text, and context.", file=sys.stderr)
        if output_dir:
            print("  --output-dir DIR  Download images to DIR and include local_path in output", file=sys.stderr)
        sys.exit(1)

    skipped = [arg for arg in raw_args if arg not in urls]
    if skipped:
        print(f"[web_image_extractor] Ignoring non-URL arguments: {skipped}", file=sys.stderr)

    if output_dir:
        print(f"[web_image_extractor] Images will be downloaded to: {output_dir}", file=sys.stderr)

    results = extract_images_from_urls(urls, output_dir=output_dir)
    json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()
