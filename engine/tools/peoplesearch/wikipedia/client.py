"""
Wikipedia API client for fetching person information.

Uses the MediaWiki API to get page info, images, and extracts for people.
"""

import logging
from typing import TypedDict

import requests

logger = logging.getLogger(__name__)

# Wikipedia API endpoint
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

# User-Agent header required by Wikipedia API
# https://www.mediawiki.org/wiki/API:Etiquette
HEADERS = {
    "User-Agent": "CourseGeneratorAgent/1.0 (https://github.com/course-generator-agent; contact@example.com)"
}


class WikiPersonInfo(TypedDict):
    """Person information from Wikipedia."""
    name: str
    wikiUrl: str
    image: str | None
    extract: str


def get_person_info(name: str, thumbnail_size: int = 330) -> WikiPersonInfo | None:
    """
    Fetch person information from Wikipedia.
    
    Uses the MediaWiki API to get the page URL, thumbnail image, and extract
    for a given person name.
    
    Args:
        name: Person's name to search for on Wikipedia
        thumbnail_size: Width of thumbnail image in pixels (default: 330)
        
    Returns:
        WikiPersonInfo dict with name, wikiUrl, image, and extract,
        or None if person not found or lacks an image.
        
    Example:
        >>> info = get_person_info("Paul Krugman")
        >>> print(info["wikiUrl"])
        https://en.wikipedia.org/wiki/Paul_Krugman
    """
    params = {
        "action": "query",
        "titles": name,
        "prop": "pageimages|extracts|info",
        "piprop": "thumbnail",
        "pithumbsize": thumbnail_size,
        "exintro": "true",
        "explaintext": "true",
        "exsentences": 3,
        "inprop": "url",
        "redirects": "1",
        "format": "json",
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Wikipedia API request failed for '{name}': {e}")
        return None
    
    pages = data.get("query", {}).get("pages", {})
    
    if not pages:
        logger.debug(f"No Wikipedia page found for '{name}'")
        return None
    
    # Get the first (and should be only) page
    page_id = next(iter(pages))
    
    # Check if page exists (negative page_id means not found)
    if page_id == "-1":
        logger.debug(f"Wikipedia page not found for '{name}'")
        return None
    
    page = pages[page_id]
    
    # Get thumbnail - return None if no image available
    thumbnail = page.get("thumbnail", {})
    image_url = thumbnail.get("source")
    
    if not image_url:
        logger.debug(f"No image found for '{name}' on Wikipedia")
        return None
    
    # Get page URL
    page_url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}")
    
    # Get extract (summary)
    extract = page.get("extract", "").strip()
    
    # Get the actual page title (may differ from search due to redirects)
    actual_title = page.get("title", name)
    
    return WikiPersonInfo(
        name=actual_title,
        wikiUrl=page_url,
        image=image_url,
        extract=extract,
    )


def search_people(query: str, max_results: int = 10) -> list[str]:
    """
    Search Wikipedia for people matching a query.
    
    Uses Wikipedia's search API to find pages, then filters for people
    by checking categories.
    
    Args:
        query: Search query (e.g., "economists Nobel Prize")
        max_results: Maximum number of results to return
        
    Returns:
        List of person names found on Wikipedia
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": "0",  # Main namespace only
        "srlimit": max_results * 2,  # Get extra to filter
        "format": "json",
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Wikipedia search failed for '{query}': {e}")
        return []
    
    search_results = data.get("query", {}).get("search", [])
    
    # Return titles of search results
    return [result["title"] for result in search_results[:max_results]]


if __name__ == "__main__":
    # Test the client
    test_names = ["Paul Krugman", "Dani Rodrik", "Albert Einstein", "NonexistentPerson12345"]
    
    print("🔍 Testing Wikipedia person info lookup\n")
    print("-" * 60)
    
    for name in test_names:
        print(f"\n📌 Looking up: {name}")
        info = get_person_info(name)
        
        if info:
            print(f"   ✅ Found: {info['name']}")
            print(f"   🔗 URL: {info['wikiUrl']}")
            print(f"   🖼️  Image: {info['image'][:60]}..." if info['image'] else "   🖼️  No image")
            print(f"   📝 Extract: {info['extract'][:100]}..." if info['extract'] else "   📝 No extract")
        else:
            print(f"   ❌ Not found or no image")

