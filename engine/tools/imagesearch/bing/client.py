"""Bing image search.

requests and beautifulsoup4 are imported lazily inside the search function.
"""

from typing import List
import json
import re


# License filter options for Bing image search
# --------------------------------------------------
# all                                      - No license filter (all images)
# free_to_share_and_use                    - View, download, share (no modification, no commercial)
# free_to_modify_share_and_use             - Modify + share (no commercial use)
# free_to_share_and_use_commercial         - Commercial use allowed (no modification)
# free_to_modify_share_and_use_commercial  - Modify + share + commercial (most permissive CC)
# public_domain                            - Anything allowed, no attribution required (least restrictive)
# --------------------------------------------------

LICENSE_FILTERS = {
    "all": "",
    # Public domain only
    "public_domain": "+filterui:license-L1",

    # Free to share and use (non-commercial + commercial)
    "free_to_share_and_use": "+filterui:license-L1_L2_L3_L4",

    # Free to modify, share and use (non-commercial + commercial)
    "free_to_modify_share_and_use": "+filterui:license-L1_L2_L3",

    # Free to share and use commercially
    "free_to_share_and_use_commercial": "+filterui:license-L1_L2_L3_L4_L5_L6_L7",

    # Free to modify, share and use commercially
    "free_to_modify_share_and_use_commercial": "+filterui:license-L1_L2_L3_L5_L6",
}

# Default license filter (change this to use a different license)
LICENSE_FILTER = "all"


def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Bing.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://www.bing.com/images/search"
        
        params = {
            "q": query,
            "form": "HDRSC2",
            "first": 1,
            "tsc": "ImageHoverTitle",
            "qft": LICENSE_FILTERS.get(LICENSE_FILTER, "")
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.bing.com/"
        }
        
        response = requests.get(url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Parse JSON from anchor tags
        for a_tag in soup.find_all('a', class_='iusc'):
            if len(results) >= max_results:
                break
            
            m_attr = a_tag.get('m')
            if m_attr:
                data = json.loads(m_attr)
                image_url = data.get("murl")
                if image_url:
                    results.append({
                        "url": image_url,
                        "thumbnail_url": data.get("turl", image_url),
                        "description": data.get("t", ""),
                        "author": data.get("purl", "Unknown")
                    })
        
        # Fallback: parse from script tags
        if not results:
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'murl' in script.string:
                    urls = re.findall(r'"murl":"(https?://[^"]+)"', script.string)
                    for image_url in urls[:max_results]:
                        results.append({
                            "url": image_url,
                            "thumbnail_url": image_url,
                            "description": "",
                            "author": "Unknown"
                        })
                    break
        
        return results[:max_results]
        
    except Exception as e:
        return [{"error": f"Bing image search failed: {str(e)}"}]


if __name__ == "__main__":
    query = "python programming"
    max_results = 5
    
    print(f"🔍 Searching images for: '{query}'")
    print("-" * 80)
    
    results = search_images(query, max_results)
    
    print(f"\n📸 Found {len(results)} images:\n")
    for i, img in enumerate(results, 1):
        if "error" in img:
            print(f"❌ {img['error']}")
        else:
            print(f"{i}. {img.get('description', 'No description')}")
            print(f"   URL: {img['url']}")
            print()
