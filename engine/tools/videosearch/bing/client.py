"""Simple YouTube video search."""

from typing import List
import requests
import re


def search_videos(query: str, max_results: int = 5, **kwargs) -> List[dict]:
    """
    Search for YouTube videos using a simple scraping approach.
    
    Args:
        query: Search query for videos
        max_results: Maximum number of videos to return (default: 5)
        **kwargs: Accepted for interface compatibility (language, etc.)
        
    Returns:
        List of video results with YouTube URLs
    """
    try:
        # Use YouTube search directly
        url = "https://www.youtube.com/results"
        
        params = {
            "search_query": query
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Extract video IDs from the response
        video_ids = re.findall(r'"videoId":"([\w-]{11})"', response.text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for vid_id in video_ids:
            if vid_id not in seen:
                seen.add(vid_id)
                unique_ids.append(vid_id)
        
        # Convert to YouTube URLs
        results = []
        for vid_id in unique_ids[:max_results]:
            results.append({
                "url": f"https://www.youtube.com/watch?v={vid_id}"
            })
        
        return results
        
    except requests.RequestException as e:
        return [{"error": f"YouTube search request failed: {str(e)}"}]
    except Exception as e:
        return [{"error": f"YouTube search failed: {str(e)}"}]


if __name__ == "__main__":
    query = """" tutorial entrelazamiento quantico"""
    max_results = 10
    
    print(f"🔍 Searching videos for: '{query}'")
    print("-" * 80)
    
    results = search_videos(query, max_results)
    
    print(f"\n🎥 Found {len(results)} YouTube videos:\n")
    for i, video in enumerate(results, 1):
        if "error" in video:
            print(f"❌ {video['error']}")
        else:
            print(f"{i}. {video['url']}")

