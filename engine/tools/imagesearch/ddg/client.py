"""DuckDuckGo image search."""

from typing import List


def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on DuckDuckGo.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    try:
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            images = ddgs.images(keywords=query, max_results=max_results)
            
            for image in images:
                results.append({
                    "url": image.get("image", ""),
                    "thumbnail_url": image.get("thumbnail", ""),
                    "description": image.get("title", ""),
                    "author": image.get("source", "Unknown")
                })
        
        return results
        
    except Exception as e:
        return [{"error": f"DuckDuckGo image search failed: {str(e)}"}]

