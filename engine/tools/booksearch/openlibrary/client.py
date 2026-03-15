"""
Open Library API client for book search and validation.

Open Library is a free, open API that provides book metadata.
No authentication required.

API Documentation: https://openlibrary.org/developers/api
"""

import logging
import re
from typing import TypedDict
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

# API endpoints
SEARCH_URL = "https://openlibrary.org/search.json"
BOOK_URL = "https://openlibrary.org/api/books"
WORKS_URL = "https://openlibrary.org"

# ISO 639-1 to Open Library language code mapping
# Open Library uses ISO 639-2/B (bibliographic) codes, not ISO 639-1
# Reference: https://openlibrary.org/languages
ISO_TO_OPENLIBRARY_LANG = {
    "es": "spa",  # Spanish
    "en": "eng",  # English
    "fr": "fre",  # French
    "de": "ger",  # German
    "pt": "por",  # Portuguese
    "it": "ita",  # Italian
    "ru": "rus",  # Russian
    "zh": "chi",  # Chinese
    "ja": "jpn",  # Japanese
    "ko": "kor",  # Korean
    "ar": "ara",  # Arabic
    "nl": "dut",  # Dutch
    "pl": "pol",  # Polish
    "ca": "cat",  # Catalan
    "gl": "glg",  # Galician
    "eu": "baq",  # Basque
}


class BookResult(TypedDict):
    """Structured book result from Open Library."""
    title: str
    authors: list[str]
    year: int | None
    publisher: str | None
    isbn: str | None
    isbn_13: str | None
    cover_url: str | None
    openlibrary_url: str | None
    edition_count: int
    language: list[str] | None


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
        parts = author.split(", ", 1)
        last = parts[0].strip()
        first_parts = parts[1].strip().split() if len(parts) > 1 else []
        initials = " ".join(f"{p[0]}." for p in first_parts if p)
        return f"{last}, {initials}" if initials else last
    
    # Handle "First Last" or "First Middle Last" format
    parts = author.strip().split()
    if len(parts) == 1:
        return parts[0]
    
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1])
    return f"{last}, {initials}"


def search_books(
    query: str,
    max_results: int = 5,
    language: str | None = None,
) -> list[BookResult]:
    """
    Search Open Library for books matching query.
    
    Args:
        query: Search query (title, author, or combined)
        max_results: Maximum number of results to return
        language: Optional language filter (ISO 639-1 like "es", "en" or 
                  Open Library codes like "spa", "eng")
        
    Returns:
        List of BookResult dictionaries with book metadata
    """
    params = {
        "q": query,
        "limit": max_results,
        "fields": "key,title,author_name,first_publish_year,publisher,isbn,cover_i,edition_count,language",
    }
    
    if language:
        # Convert ISO 639-1 codes to Open Library codes if needed
        lang_code = ISO_TO_OPENLIBRARY_LANG.get(language.lower(), language)
        params["language"] = lang_code
    
    try:
        response = requests.get(SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Open Library search failed: {e}")
        return []
    
    results = []
    for doc in data.get("docs", [])[:max_results]:
        # Get ISBNs
        isbns = doc.get("isbn", [])
        isbn_10 = None
        isbn_13 = None
        for isbn in isbns:
            if len(isbn) == 10 and isbn_10 is None:
                isbn_10 = isbn
            elif len(isbn) == 13 and isbn_13 is None:
                isbn_13 = isbn
            if isbn_10 and isbn_13:
                break
        
        # Get cover URL
        cover_id = doc.get("cover_i")
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None
        
        # Get Open Library URL
        key = doc.get("key", "")
        ol_url = f"https://openlibrary.org{key}" if key else None
        
        # Format authors for APA (deduplicate to avoid repeated author names)
        authors = doc.get("author_name", [])
        seen_authors = set()
        formatted_authors = []
        for a in authors:
            formatted = _format_author_name(a)
            # Deduplicate by normalized name
            normalized = formatted.lower().strip()
            if normalized not in seen_authors:
                seen_authors.add(normalized)
                formatted_authors.append(formatted)
        
        # Get publisher (first one if multiple)
        publishers = doc.get("publisher", [])
        publisher = publishers[0] if publishers else None
        
        result: BookResult = {
            "title": doc.get("title", "Unknown Title"),
            "authors": formatted_authors,
            "year": doc.get("first_publish_year"),
            "publisher": publisher,
            "isbn": isbn_10,
            "isbn_13": isbn_13,
            "cover_url": cover_url,
            "openlibrary_url": ol_url,
            "edition_count": doc.get("edition_count", 1),
            "language": doc.get("language"),
        }
        results.append(result)
    
    return results


def search_books_by_title_author(
    title: str,
    author: str | None = None,
    max_results: int = 3,
) -> list[BookResult]:
    """
    Search for books by title and optionally author.
    
    More precise search than general query.
    
    Args:
        title: Book title to search for
        author: Optional author name
        max_results: Maximum results to return
        
    Returns:
        List of matching BookResult dictionaries
    """
    # Build query with title and author fields
    query_parts = [f'title:"{title}"']
    if author:
        # Extract last name for better matching
        author_parts = author.split()
        last_name = author_parts[-1] if author_parts else author
        query_parts.append(f'author:"{last_name}"')
    
    query = " ".join(query_parts)
    return search_books(query, max_results=max_results)


def get_book_details(isbn: str) -> BookResult | None:
    """
    Get detailed book information by ISBN.
    
    Args:
        isbn: ISBN-10 or ISBN-13
        
    Returns:
        BookResult with full details, or None if not found
    """
    # Clean ISBN (remove dashes)
    clean_isbn = re.sub(r"[-\s]", "", isbn)
    
    params = {
        "bibkeys": f"ISBN:{clean_isbn}",
        "format": "json",
        "jscmd": "data",
    }
    
    try:
        response = requests.get(BOOK_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Open Library book lookup failed: {e}")
        return None
    
    key = f"ISBN:{clean_isbn}"
    if key not in data:
        return None
    
    book = data[key]
    
    # Extract authors (deduplicate to avoid repeated author names)
    authors = []
    seen_authors = set()
    for author in book.get("authors", []):
        name = author.get("name", "")
        if name:
            formatted = _format_author_name(name)
            normalized = formatted.lower().strip()
            if normalized not in seen_authors:
                seen_authors.add(normalized)
                authors.append(formatted)
    
    # Extract publisher
    publishers = book.get("publishers", [])
    publisher = publishers[0].get("name") if publishers else None
    
    # Get cover
    cover = book.get("cover", {})
    cover_url = cover.get("medium") or cover.get("small")
    
    # Determine ISBN-10 vs ISBN-13
    isbn_10 = clean_isbn if len(clean_isbn) == 10 else None
    isbn_13 = clean_isbn if len(clean_isbn) == 13 else None
    
    result: BookResult = {
        "title": book.get("title", "Unknown Title"),
        "authors": authors,
        "year": book.get("publish_date", "").split()[-1] if book.get("publish_date") else None,
        "publisher": publisher,
        "isbn": isbn_10,
        "isbn_13": isbn_13,
        "cover_url": cover_url,
        "openlibrary_url": book.get("url"),
        "edition_count": 1,
        "language": None,
    }
    
    # Try to convert year to int
    if result["year"]:
        try:
            result["year"] = int(result["year"])
        except ValueError:
            result["year"] = None
    
    return result


def validate_book(
    title: str,
    author: str | None = None,
) -> BookResult | None:
    """
    Validate a book exists and get its metadata.
    
    Searches by title/author and returns the best match.
    
    Args:
        title: Book title to validate
        author: Optional author for better matching
        
    Returns:
        BookResult if found, None otherwise
    """
    results = search_books_by_title_author(title, author, max_results=1)
    return results[0] if results else None

