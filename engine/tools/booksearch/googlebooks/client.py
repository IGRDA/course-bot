"""
Google Books API client for book search.

Google Books provides a free API for searching book metadata.
Requires an API key from Google Cloud Console.

API Documentation: https://developers.google.com/books/docs/v1/using
"""

import logging
import os
from typing import TypedDict

import requests

logger = logging.getLogger(__name__)

# API endpoint
SEARCH_URL = "https://www.googleapis.com/books/v1/volumes"


class GoogleBookResult(TypedDict):
    """Structured book result from Google Books."""
    title: str
    authors: list[str]
    year: int | None
    publisher: str | None
    isbn: str | None
    isbn_13: str | None
    google_books_url: str  # Direct link to book page
    thumbnail_url: str | None
    language: str | None
    description: str | None
    average_rating: float | None  # 1-5 scale
    ratings_count: int | None


def _format_author_name(author: str) -> str:
    """
    Format author name to APA style: Last, F. M.
    
    Args:
        author: Author name in various formats
        
    Returns:
        APA-formatted author name
    """
    # Handle already formatted names (Last, First)
    if ", " in author:
        return author
    
    # Handle "First Last" or "First Middle Last" format
    parts = author.strip().split()
    if len(parts) == 1:
        return parts[0]
    
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1])
    return f"{last}, {initials}"


def _extract_isbn(industry_identifiers: list[dict] | None) -> tuple[str | None, str | None]:
    """
    Extract ISBN-10 and ISBN-13 from Google Books identifiers.
    
    Args:
        industry_identifiers: List of identifier dicts from API
        
    Returns:
        Tuple of (isbn_10, isbn_13)
    """
    if not industry_identifiers:
        return None, None
    
    isbn_10 = None
    isbn_13 = None
    
    for identifier in industry_identifiers:
        id_type = identifier.get("type", "")
        id_value = identifier.get("identifier", "")
        
        if id_type == "ISBN_10":
            isbn_10 = id_value
        elif id_type == "ISBN_13":
            isbn_13 = id_value
    
    return isbn_10, isbn_13


def _extract_year(published_date: str | None) -> int | None:
    """
    Extract year from Google Books published date.
    
    Args:
        published_date: Date string (various formats: "2020", "2020-05", "2020-05-15")
        
    Returns:
        Year as integer, or None
    """
    if not published_date:
        return None
    
    # Extract first 4 digits (year)
    try:
        year_str = published_date[:4]
        return int(year_str)
    except (ValueError, IndexError):
        return None


def search_books(
    query: str,
    max_results: int = 5,
    language: str | None = None,
    order_by: str | None = None,
) -> list[GoogleBookResult]:
    """
    Search Google Books for books matching query.
    
    Args:
        query: Search query (title, author, or combined)
        max_results: Maximum number of results to return (max 40)
        language: Optional language filter (e.g., "en", "es")
        order_by: Optional sort order ("relevance" or "newest")
        
    Returns:
        List of GoogleBookResult dictionaries with book metadata and direct URLs
    """
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_BOOKS_API_KEY not set, skipping Google Books search")
        return []
    
    params = {
        "q": query,
        "maxResults": min(max_results, 40),  # API max is 40
        "key": api_key,
        "printType": "books",  # Only books, not magazines
    }
    
    if language:
        params["langRestrict"] = language
    
    if order_by:
        params["orderBy"] = order_by  # "relevance" or "newest"
    
    try:
        response = requests.get(SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Google Books search failed: {e}")
        return []
    
    results: list[GoogleBookResult] = []
    
    for item in data.get("items", [])[:max_results]:
        volume_info = item.get("volumeInfo", {})
        
        if not volume_info.get("title"):
            continue
        
        # Get ISBNs
        isbn_10, isbn_13 = _extract_isbn(volume_info.get("industryIdentifiers"))
        
        # Build direct URL to book page
        volume_id = item.get("id", "")
        google_books_url = f"https://books.google.com/books?id={volume_id}" if volume_id else ""
        
        # Get thumbnail
        image_links = volume_info.get("imageLinks", {})
        thumbnail_url = image_links.get("thumbnail") or image_links.get("smallThumbnail")
        
        # Format authors
        authors = volume_info.get("authors", [])
        formatted_authors = [_format_author_name(a) for a in authors]
        
        result: GoogleBookResult = {
            "title": volume_info.get("title", "Unknown Title"),
            "authors": formatted_authors,
            "year": _extract_year(volume_info.get("publishedDate")),
            "publisher": volume_info.get("publisher"),
            "isbn": isbn_10,
            "isbn_13": isbn_13,
            "google_books_url": google_books_url,
            "thumbnail_url": thumbnail_url,
            "language": volume_info.get("language"),
            "description": volume_info.get("description"),
            "average_rating": volume_info.get("averageRating"),
            "ratings_count": volume_info.get("ratingsCount"),
        }
        results.append(result)
    
    return results


def search_book_by_title(
    title: str,
    author: str | None = None,
    max_results: int = 3,
) -> list[GoogleBookResult]:
    """
    Search for a specific book by title and optionally author.
    
    More precise search than general query.
    
    Args:
        title: Book title to search for
        author: Optional author name for better matching
        max_results: Maximum results to return
        
    Returns:
        List of matching GoogleBookResult dictionaries
    """
    # Build query with intitle and inauthor for better precision
    query_parts = [f'intitle:"{title}"']
    if author:
        # Use just last name for better matching
        author_parts = author.split()
        last_name = author_parts[-1] if author_parts else author
        query_parts.append(f'inauthor:"{last_name}"')
    
    query = " ".join(query_parts)
    return search_books(query, max_results=max_results)


if __name__ == "__main__":
    # Quick test
    import json
    
    query = "quantum mechanics"
    print(f"🔍 Searching Google Books for: '{query}'")
    print("-" * 60)
    
    results = search_books(query, max_results=3)
    
    if not results:
        print("No results found (check GOOGLE_BOOKS_API_KEY)")
    else:
        for i, book in enumerate(results, 1):
            print(f"\n{i}. {book['title']}")
            print(f"   Authors: {', '.join(book['authors'][:3])}")
            print(f"   Year: {book['year']}")
            print(f"   Publisher: {book['publisher']}")
            print(f"   URL: {book['google_books_url']}")

