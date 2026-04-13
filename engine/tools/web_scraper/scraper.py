"""
Scrape a website: crawl root + child pages, extract text and images from all.

Uses the Playwright stealth browser (proven on Cloud Run against SiteGround
CAPTCHA and similar bot protection) by reusing the singleton from
``web_image_extractor``.

Usage:
    python -m tools.web_scraper https://example.com/

Public API:
    scrape_website(url, ...) -> dict with pages, text, images
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from tools.web_image_extractor.extractor import (
    _download_extracted_images,
    _extract_background_images,
    _extract_img_tags,
    _extract_picture_sources,
    _fetch_html,
    _find_context_text,
    _find_preceding_heading,
)

_LOG_PREFIX = "[web_scraper]"

# ---------------------------------------------------------------------------
# Link discovery
# ---------------------------------------------------------------------------

_SKIP_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".rar",
    ".7z",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".css",
    ".js",
    ".xml",
    ".json",
    ".rss",
}

_SKIP_PATH_PATTERNS = {
    "/wp-admin",
    "/wp-login",
    "/wp-json",
    "/xmlrpc",
    "/feed",
    "/cart",
    "/checkout",
    "/my-account",
    "/login",
    "/register",
    "/signup",
    "/signin",
    "/search",
    "/tag/",
    "/author/",
}

_SKIP_PATH_EXACT = {
    "/aviso-legal",
    "/aviso-legal/",
    "/politica-de-privacidad",
    "/politica-de-privacidad/",
    "/politica-de-cookies",
    "/politica-de-cookies/",
    "/privacy-policy",
    "/privacy-policy/",
    "/cookie-policy",
    "/cookie-policy/",
    "/terms-of-service",
    "/terms-of-service/",
    "/terms-and-conditions",
    "/terms-and-conditions/",
    "/legal",
    "/legal/",
    "/impressum",
    "/impressum/",
}


def _is_same_domain(url: str, root_domain: str) -> bool:
    """Check if a URL belongs to the same domain (or a subdomain of it)."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    root = root_domain.lower()
    return host == root or host.endswith("." + root)


def _should_skip_url(url: str) -> bool:
    """Return True if the URL should be excluded from crawling."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    ext = Path(path).suffix.lower()
    if ext in _SKIP_EXTENSIONS:
        return True

    for pattern in _SKIP_PATH_PATTERNS:
        if pattern in path:
            return True

    return bool(path in _SKIP_PATH_EXACT or path.rstrip("/") in {p.rstrip("/") for p in _SKIP_PATH_EXACT})


def _discover_links(html: str, page_url: str, root_domain: str) -> list[str]:
    """Extract same-domain internal links from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[str] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        resolved = urljoin(page_url, href)
        parsed = urlparse(resolved)
        normalized = parsed._replace(fragment="", query="").geturl()

        if not _is_same_domain(normalized, root_domain):
            continue

        if _should_skip_url(normalized):
            continue

        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)

    return links


# ---------------------------------------------------------------------------
# Text content extraction
# ---------------------------------------------------------------------------

_BOILERPLATE_TAGS = {"nav", "header", "footer", "script", "style", "noscript", "aside", "svg"}
_COOKIE_CLASSES = {"cookie", "consent", "gdpr", "privacy-banner", "cc-banner"}


def _extract_page_content(html: str) -> dict:
    """Extract cleaned text content and heading structure from HTML.

    Returns::
        {
            "title": str,
            "headings": [str, ...],
            "content": str,
            "content_length": int,
        }
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()

    for tag in soup.find_all(attrs={"class": True}):
        classes = " ".join(tag.get("class", [])).lower()
        if any(kw in classes for kw in _COOKIE_CLASSES):
            tag.decompose()
    for tag in soup.find_all(attrs={"id": True}):
        id_val = (tag.get("id") or "").lower()
        if any(kw in id_val for kw in _COOKIE_CLASSES):
            tag.decompose()

    headings: list[str] = []
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        text = h.get_text(strip=True)
        if text:
            headings.append(text)

    main_el = soup.find("main") or soup.find("article") or soup.find("body")
    if main_el:
        content = main_el.get_text(separator="\n", strip=True)
    else:
        content = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in content.split("\n") if line.strip()]
    content = "\n".join(lines)

    return {
        "title": title,
        "headings": headings,
        "content": content,
        "content_length": len(content),
    }


# ---------------------------------------------------------------------------
# Image extraction (wraps web_image_extractor methods)
# ---------------------------------------------------------------------------


def _extract_page_images(html: str, page_url: str) -> list[dict]:
    """Extract all content images from a page's HTML with context metadata."""
    soup = BeautifulSoup(html, "html.parser")

    raw_images: list[dict] = []
    raw_images.extend(_extract_img_tags(soup, page_url))
    raw_images.extend(_extract_picture_sources(soup, page_url))
    raw_images.extend(_extract_background_images(soup, page_url))

    seen_urls: set[str] = set()
    images: list[dict] = []

    for entry in raw_images:
        src = entry["src"]
        if src in seen_urls:
            continue
        seen_urls.add(src)

        element = entry.pop("_element", None)
        heading = _find_preceding_heading(element) if element else ""
        context = _find_context_text(element) if element else ""

        images.append(
            {
                "src": src,
                "alt": entry["alt"],
                "context_heading": heading,
                "context_text": context,
            }
        )

    return images


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def scrape_website(
    url: str,
    output_dir: str | None = None,
    max_pages: int = 20,
    max_depth: int = 1,
    timeout: int = 15,
    inter_page_delay: float = 2.0,
) -> dict:
    """Scrape a website: fetch root + child pages, extract text and images.

    Parameters
    ----------
    url:
        Root URL to start crawling from.
    output_dir:
        Directory to download images into.  When set, each image entry
        gets a ``local_path`` key (relative to *output_dir*'s parent).
    max_pages:
        Maximum number of child pages to fetch (excluding the root).
    max_depth:
        Crawl depth. 1 = root + direct children (default).
    timeout:
        Per-page HTTP timeout in seconds.
    inter_page_delay:
        Seconds to wait between page fetches to avoid rate-limiting.

    Returns
    -------
    dict with keys ``root_url``, ``pages_fetched``, ``total_images``,
    and ``pages`` (list of page dicts each containing ``url``, ``title``,
    ``content``, ``headings``, ``content_length``, ``images``, and
    ``internal_links``).
    """
    root_parsed = urlparse(url)
    root_domain = root_parsed.netloc

    result: dict = {
        "root_url": url,
        "pages_fetched": 0,
        "total_images": 0,
        "pages": [],
    }

    all_images: list[dict] = []
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(url, 0)]

    while queue:
        current_url, depth = queue.pop(0)

        normalized = urlparse(current_url)._replace(fragment="", query="").geturl()
        if normalized in visited:
            continue
        visited.add(normalized)

        if depth > 0 and len(visited) - 1 > max_pages:
            print(f"{_LOG_PREFIX} Reached max_pages={max_pages}, stopping crawl", file=sys.stderr)
            break

        if depth > 0:
            print(f"{_LOG_PREFIX} Waiting {inter_page_delay}s between pages ...", file=sys.stderr)
            time.sleep(inter_page_delay)

        print(f"{_LOG_PREFIX} [{len(visited)}/{max_pages + 1}] Fetching {current_url} ...", file=sys.stderr)
        html = _fetch_html(current_url, timeout)
        if not html:
            print(f"{_LOG_PREFIX}   FAILED to fetch {current_url}", file=sys.stderr)
            continue

        page_content = _extract_page_content(html)
        page_images = _extract_page_images(html, current_url)
        child_links = _discover_links(html, current_url, root_domain)

        page_data = {
            "url": current_url,
            "title": page_content["title"],
            "content": page_content["content"],
            "headings": page_content["headings"],
            "content_length": page_content["content_length"],
            "images": page_images,
            "internal_links": child_links,
        }
        result["pages"].append(page_data)
        all_images.extend(page_images)

        print(
            f"{_LOG_PREFIX}   {page_content['content_length']} chars, "
            f"{len(page_images)} images, {len(child_links)} links",
            file=sys.stderr,
        )

        if depth < max_depth:
            for link in child_links:
                link_normalized = urlparse(link)._replace(fragment="", query="").geturl()
                if link_normalized not in visited:
                    queue.append((link, depth + 1))

    # Deduplicate images globally by src before downloading
    seen_src: set[str] = set()
    unique_images: list[dict] = []
    for img in all_images:
        if img["src"] not in seen_src:
            seen_src.add(img["src"])
            unique_images.append(img)

    if output_dir and unique_images:
        import tools.web_image_extractor.extractor as _wie

        pw_was_used = _wie._pw_context is not None
        print(
            f"{_LOG_PREFIX} Downloading {len(unique_images)} unique images to {output_dir} ...",
            file=sys.stderr,
        )
        _download_extracted_images(unique_images, Path(output_dir), use_playwright=pw_was_used)

        # Propagate local_path back to per-page image entries
        src_to_path: dict[str, str | None] = {img["src"]: img.get("local_path") for img in unique_images}
        for page in result["pages"]:
            for img in page["images"]:
                img["local_path"] = src_to_path.get(img["src"])

        downloaded = sum(1 for img in unique_images if img.get("local_path"))
        print(f"{_LOG_PREFIX}   Downloaded {downloaded}/{len(unique_images)} images", file=sys.stderr)

    result["pages_fetched"] = len(result["pages"])
    result["total_images"] = len(unique_images)

    print(
        f"{_LOG_PREFIX} Done: {result['pages_fetched']} pages, {result['total_images']} unique images",
        file=sys.stderr,
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Scrape a website: crawl root + child pages, extract text "
            "content and images from all pages. Outputs JSON to stdout."
        ),
    )
    parser.add_argument("url", help="Root URL to scrape")
    parser.add_argument(
        "--output-dir",
        help="Download images to this directory and include local_path in output",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Max child pages to fetch (default: 20)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Crawl depth from root (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Per-page HTTP timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between page fetches (default: 2.0)",
    )
    args = parser.parse_args()

    result = scrape_website(
        url=args.url,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        timeout=args.timeout,
        inter_page_delay=args.delay,
    )

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print(file=sys.stdout)
