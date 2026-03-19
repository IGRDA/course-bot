"""YouTube video search with API v3 primary and yt-dlp flat fallback.

Primary path uses the YouTube Data API v3 (requires YOUTUBE_API_KEY env var).
When the API key is missing or any API error occurs (quota exceeded, network,
etc.), the module transparently falls back to yt-dlp with extract_flat=True,
which returns partial metadata but avoids the "sign in to confirm you're not
a bot" blocking on server/datacenter IPs.
"""

import logging
import os
import re
from datetime import datetime
from typing import List

import requests

logger = logging.getLogger(__name__)

_YT_API_BASE = "https://www.googleapis.com/youtube/v3"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search_videos(
    query: str,
    max_results: int = 5,
    language: str | None = None,
) -> List[dict]:
    """Search for YouTube videos and return metadata dicts.

    Tries the YouTube Data API v3 first.  If that fails for *any* reason
    (missing key, quota exceeded, network error, …) it falls back to yt-dlp
    with ``extract_flat=True``.

    Args:
        query: Search query for videos.
        max_results: Maximum number of videos to return (default: 5).
        language: ISO 639-1 language code (e.g. "es", "en") to bias results.

    Returns:
        List of dicts, each with keys: title, url, duration, published_at,
        thumbnail, channel, views, likes.  Some values may be 0 / empty
        when returned by the flat-fallback path.
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()

    if api_key:
        try:
            return _search_via_api(query, max_results, api_key, language=language)
        except Exception as exc:
            logger.warning(
                "YouTube Data API failed (%s), falling back to yt-dlp flat extraction",
                exc,
            )

    return _search_via_ytdlp_flat(query, max_results)


# ---------------------------------------------------------------------------
# YouTube Data API v3
# ---------------------------------------------------------------------------

def _search_via_api(
    query: str,
    max_results: int,
    api_key: str,
    language: str | None = None,
) -> List[dict]:
    """Search using YouTube Data API v3 (search.list + videos.list)."""
    # Step 1 – search.list  (costs 100 quota units)
    params: dict = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": api_key,
    }
    if language:
        params["relevanceLanguage"] = language

    search_resp = requests.get(
        f"{_YT_API_BASE}/search",
        params=params,
        timeout=15,
    )
    search_resp.raise_for_status()
    search_data = search_resp.json()

    items = search_data.get("items", [])
    if not items:
        return []

    video_ids = [it["id"]["videoId"] for it in items]

    # Step 2 – videos.list for duration, views, likes  (costs 1 unit per call)
    details_resp = requests.get(
        f"{_YT_API_BASE}/videos",
        params={
            "part": "contentDetails,statistics",
            "id": ",".join(video_ids),
            "key": api_key,
        },
        timeout=15,
    )
    details_resp.raise_for_status()
    details_map: dict[str, dict] = {
        it["id"]: it for it in details_resp.json().get("items", [])
    }

    results: list[dict] = []
    for item in items:
        vid = item["id"]["videoId"]
        snippet = item.get("snippet", {})
        detail = details_map.get(vid, {})
        stats = detail.get("statistics", {})
        content = detail.get("contentDetails", {})

        thumbnails = snippet.get("thumbnails", {})
        thumbnail = (
            thumbnails.get("maxres", {}).get("url")
            or thumbnails.get("high", {}).get("url")
            or thumbnails.get("medium", {}).get("url")
            or thumbnails.get("default", {}).get("url", "")
        )

        results.append({
            "title": snippet.get("title", ""),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "duration": _parse_iso8601_duration(content.get("duration", "")),
            "published_at": _parse_iso_timestamp(snippet.get("publishedAt", "")),
            "thumbnail": thumbnail,
            "channel": snippet.get("channelTitle", ""),
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
        })

    return results


def _parse_iso8601_duration(iso_str: str) -> int:
    """Convert ISO 8601 duration (e.g. ``PT4M13S``) to seconds."""
    if not iso_str:
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _parse_iso_timestamp(iso_str: str) -> int:
    """Convert an ISO-8601 timestamp to milliseconds since epoch."""
    if not iso_str:
        return 0
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# yt-dlp fallback (flat extraction – no per-video page fetch)
# ---------------------------------------------------------------------------

def _search_via_ytdlp_flat(query: str, max_results: int) -> List[dict]:
    """Search using yt-dlp with extract_flat=True (partial metadata)."""
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is not installed; cannot fall back from API")
        return []

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "ignoreerrors": True,
    }

    results: list[dict] = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch{max_results}:{query}"
            search_result = ydl.extract_info(search_query, download=False)

            if not search_result or "entries" not in search_result:
                return results

            for entry in search_result.get("entries", []):
                if entry is None:
                    continue

                video_id = entry.get("id", "")
                thumbnail = _get_best_thumbnail(entry, video_id)
                published_at = _parse_upload_date(entry.get("upload_date"))

                results.append({
                    "title": entry.get("title", ""),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "duration": entry.get("duration", 0) or 0,
                    "published_at": published_at,
                    "thumbnail": thumbnail,
                    "channel": entry.get("channel", "") or entry.get("uploader", ""),
                    "views": entry.get("view_count", 0) or 0,
                    "likes": entry.get("like_count", 0) or 0,
                })

    except Exception as exc:
        logger.error("yt-dlp flat search failed: %s", exc)
        return []

    return results


def _get_best_thumbnail(entry: dict, video_id: str) -> str:
    """Pick the highest-resolution thumbnail available from a yt-dlp entry."""
    thumbnails = entry.get("thumbnails", [])
    if thumbnails:
        sorted_thumbs = sorted(
            thumbnails,
            key=lambda t: (t.get("height", 0) or 0) * (t.get("width", 0) or 0),
            reverse=True,
        )
        if sorted_thumbs:
            return sorted_thumbs[0].get("url", "")

    if video_id:
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    return entry.get("thumbnail", "")


def _parse_upload_date(upload_date: str | None) -> int:
    """Convert YYYYMMDD string to milliseconds timestamp."""
    if not upload_date or len(upload_date) != 8:
        return 0
    try:
        dt = datetime.strptime(upload_date, "%Y%m%d")
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "tutorial entrelazamiento cuántico"
    max_results = 3

    print(f"Searching videos for: '{query}'")
    print("-" * 80)

    results = search_videos(query, max_results)

    print(f"\nFound {len(results)} YouTube videos:\n")
    for i, video in enumerate(results, 1):
        if "error" in video:
            print(f"  Error: {video['error']}")
        else:
            print(f"{i}. {video['title']}")
            print(f"   URL: {video['url']}")
            print(f"   Channel: {video['channel']}")
            print(f"   Duration: {video['duration']}s")
            print(f"   Views: {video.get('views', 0):,}")
            print(f"   Likes: {video.get('likes', 0):,}")
            print(f"   Thumbnail: {video['thumbnail']}")
            print()
