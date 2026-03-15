"""
Bibliography Generator Agent.

Generates bibliographies for course modules using a hybrid approach:
1. LLM suggests relevant books based on module content
2. Open Library API validates and enriches book metadata
3. Academic article search APIs find relevant papers
4. Citations are formatted in APA 7 style
5. URLs are validated to ensure they are accessible
6. Output: books first, then articles
"""

import json
import logging
from typing import Optional

import requests

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from workflows.state import (
    CourseState,
    BookReference,
    ModuleBibliography,
    CourseBibliography,
    ModuleBibliographyEmbed,
    BibliographyItem,
    Module,
)
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.booksearch.openlibrary.client import (
    validate_book,
    search_books_by_title_author,
    BookResult,
)
from tools.booksearch.googlebooks.client import (
    search_book_by_title as google_search_book,
    GoogleBookResult,
)
from tools.articlesearch import create_article_search, ArticleResult
from .prompts import book_suggestion_prompt, apa_formatting_prompt

logger = logging.getLogger(__name__)


# ============================================================
# JSON Parsing Utilities
# ============================================================

import re


def _parse_json_response(raw: str, context: str = "LLM response") -> list:
    """Robustly parse a JSON array from an LLM response.

    Handles markdown fences, trailing commas, and other common LLM quirks.
    Returns an empty list on failure instead of raising.
    """
    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                text = candidate
                break

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before ] or }
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON array from the text
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            fixed_match = re.sub(r",\s*([}\]])", r"\1", match.group())
            try:
                return json.loads(fixed_match)
            except json.JSONDecodeError:
                pass

    logger.error("Failed to parse %s as JSON (length=%d)", context, len(raw))
    return []


# ============================================================
# Deduplication Utilities
# ============================================================

def _normalize_title_for_dedup(title: str) -> str:
    """
    Normalize a book/article title for deduplication.
    
    Handles variations like:
    - "Quantum Mechanics" vs "quantum mechanics"
    - "Introduction to Quantum Mechanics: A Modern Approach" vs "Introduction to Quantum Mechanics"
    - "Quantum Mechanics (2nd Edition)" vs "Quantum Mechanics"
    
    Args:
        title: Original title
        
    Returns:
        Normalized title string for comparison
    """
    if not title:
        return ""
    
    # Convert to lowercase
    normalized = title.lower().strip()
    
    # Remove edition markers (various formats)
    edition_patterns = [
        r'\(\d+(?:st|nd|rd|th)?\s*(?:ed\.?|edition)\)',  # (2nd ed.), (3rd edition)
        r',?\s*\d+(?:st|nd|rd|th)?\s*(?:ed\.?|edition)',  # , 2nd ed.
        r'\[\d+(?:st|nd|rd|th)?\s*(?:ed\.?|edition)\]',   # [2nd edition]
        r'\(\d{4}\)',  # (2020) - year in parentheses
        r',?\s*vol\.?\s*\d+',  # , vol. 1
        r',?\s*volume\s*\d+',  # , volume 1
    ]
    for pattern in edition_patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    
    # Remove subtitle after colon or dash (keep main title)
    if ':' in normalized:
        normalized = normalized.split(':')[0]
    if ' - ' in normalized:
        normalized = normalized.split(' - ')[0]
    
    # Remove punctuation and special characters
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Remove common filler words that don't distinguish titles
    filler_words = [
        'explained', 'introduction', 'a', 'an', 'the', 'to', 'of', 'and', 'in',
        'complete', 'modern', 'new', 'guide', 'fundamentals', 'principles',
        'basics', 'handbook', 'textbook', 'manual', 'course', 'laboratory', 'lab',
    ]
    words = normalized.split()
    words = [w for w in words if w not in filler_words]
    normalized = ' '.join(words)
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    # Keep only first 40 chars (main concept - shorter to catch more variants)
    normalized = normalized[:40].strip()
    
    return normalized


def _is_duplicate_title(
    new_title: str,
    existing_titles: set[str],
) -> bool:
    """
    Check if a title is a duplicate of any existing title.
    
    Args:
        new_title: Title to check
        existing_titles: Set of normalized existing titles
        
    Returns:
        True if duplicate found, False otherwise
    """
    normalized_new = _normalize_title_for_dedup(new_title)
    return normalized_new in existing_titles


def _get_normalized_title_key(title: str) -> str:
    """Get a normalized title key for deduplication tracking."""
    return _normalize_title_for_dedup(title)


# ============================================================
# Recency Scoring
# ============================================================

# Cutoffs for "recent" publications
RECENT_BOOK_YEAR = 2010  # Books from 2010+ are considered recent
RECENT_ARTICLE_YEAR = 2015  # Articles from 2015+ are considered recent
CURRENT_YEAR = 2026  # For calculating age


def _get_recency_score(year: int | str | None, is_book: bool = True) -> int:
    """
    Calculate a recency score for sorting (higher = more recent = better).
    
    Scoring:
    - Recent publications (after cutoff): 100 + years_since_cutoff (newest first)
    - Older publications: years_before_cutoff (oldest last)
    
    This creates a clear preference for recent publications while still
    allowing classic/foundational works.
    
    Args:
        year: Publication year (None or string like 'n.d.' treated as old)
        is_book: True for books, False for articles
        
    Returns:
        Recency score (higher is better)
    """
    if not year:
        return 0  # Unknown year = lowest priority
    
    # Handle string years
    if isinstance(year, str):
        try:
            year = int(year)
        except ValueError:
            return 0  # Can't parse (e.g., 'n.d.') = lowest priority
    
    cutoff = RECENT_BOOK_YEAR if is_book else RECENT_ARTICLE_YEAR
    
    if year >= cutoff:
        # Recent: bonus + how many years after cutoff
        return 100 + (year - cutoff)
    else:
        # Older: just the year difference (so 2005 scores higher than 1995)
        return max(0, year - 1950)  # Normalize to roughly 0-60 range


def _sort_books_by_recency(books: list, prefer_recent: bool = True, num_books: int | None = None) -> list:
    """
    Sort books preferring recent publications and optionally filter out old ones.
    
    Strategy:
    - Sort all books by recency score (newest first)
    - If we have enough recent books (post-2010), prioritize them
    - Include 1-2 classic books only if they are foundational
    
    Args:
        books: List of BookReference objects
        prefer_recent: If True, sort newest first; if False, keep original order
        num_books: Target number of books (used for filtering)
        
    Returns:
        Sorted list of books (may exclude very old publications if enough recent ones exist)
    """
    if not prefer_recent:
        return books
    
    # Sort by recency
    sorted_books = sorted(
        books,
        key=lambda b: _get_recency_score(b.year, is_book=True),
        reverse=True  # Higher score = more recent = first
    )
    
    # If we have more books than needed, prefer recent ones
    if num_books and len(sorted_books) > num_books:
        # Count recent books (post-2010)
        recent_books = [b for b in sorted_books if _get_recency_score(b.year, is_book=True) >= 100]
        old_books = [b for b in sorted_books if _get_recency_score(b.year, is_book=True) < 100]
        
        # If we have enough recent books, only include 1 classic
        if len(recent_books) >= num_books - 1:
            # Take enough recent books, plus at most 1 classic
            max_classics = 1
            return recent_books[:num_books-max_classics] + old_books[:max_classics]
    
    return sorted_books


# ============================================================
# Title Translation (for English fallback search)
# ============================================================

def _translate_title_to_english(
    title: str,
    llm_provider: str = "mistral",
) -> str | None:
    """
    Translate a book title to English using LLM.
    
    Args:
        title: Book title in any language
        llm_provider: LLM provider to use
        
    Returns:
        English translation of the title, or None if translation fails
    """
    # Skip if already in English (basic heuristic)
    spanish_indicators = ["á", "é", "í", "ó", "ú", "ñ", "ü", "¿", "¡", " de ", " la ", " el ", " los ", " las "]
    is_likely_spanish = any(indicator in title.lower() for indicator in spanish_indicators)
    
    if not is_likely_spanish:
        return None  # Likely already English
    
    try:
        model_name = resolve_text_model_name(llm_provider)
        llm_kwargs = {"temperature": 0}
        if model_name:
            llm_kwargs["model_name"] = model_name
        llm = create_text_llm(provider=llm_provider, **llm_kwargs)
        
        prompt = f"""Translate this book title to English. Return ONLY the translated title, nothing else.

Title: {title}

English translation:"""
        
        result = llm.invoke(prompt)
        translated = result.content.strip().strip('"').strip("'")
        
        # Basic validation
        if translated and len(translated) > 3 and translated.lower() != title.lower():
            return translated
        return None
    except Exception as e:
        logger.debug(f"Title translation failed: {e}")
        return None


# ============================================================
# Language-Aware Keyword Extraction
# ============================================================

# Language code mappings
LANGUAGE_CODES = {
    "español": "es",
    "spanish": "es",
    "es": "es",
    "english": "en",
    "inglés": "en",
    "en": "en",
    "french": "fr",
    "français": "fr",
    "fr": "fr",
    "german": "de",
    "deutsch": "de",
    "de": "de",
    "portuguese": "pt",
    "português": "pt",
    "pt": "pt",
}


def _get_language_code(language: str) -> str:
    """Convert language name to ISO 639-1 code."""
    return LANGUAGE_CODES.get(language.lower(), "en")


def _extract_search_keywords(
    module_title: str,
    module_description: str,
    language: str,
    llm_provider: str = "mistral",
) -> tuple[list[str], list[str]]:
    """
    Extract academic search keywords in both native language and English.
    
    For non-English courses, returns keywords in both the course language
    (for language-filtered API searches) and English (for fallback searches).
    
    Args:
        module_title: Module title (may be in any language)
        module_description: Module description
        language: Course language (e.g., "Español", "English")
        llm_provider: LLM provider to use
        
    Returns:
        Tuple of (native_keywords, english_keywords)
        - native_keywords: Keywords in the course language
        - english_keywords: Keywords in English (for fallback)
    """
    lang_code = _get_language_code(language)
    is_english = lang_code == "en"
    
    try:
        model_name = resolve_text_model_name(llm_provider)
        llm_kwargs = {"temperature": 0}
        if model_name:
            llm_kwargs["model_name"] = model_name
        llm = create_text_llm(provider=llm_provider, **llm_kwargs)
        
        if is_english:
            # English course: only need English keywords
            prompt = f"""Extract 3-5 academic search keywords from this module content.

Module Title: {module_title}
Module Description: {module_description}

IMPORTANT:
1. Use standard academic terminology
2. Include both broad topic and specific subtopics
3. Each keyword/phrase should be 1-4 words
4. Focus on the main academic concepts, not filler words

Return ONLY a JSON object with one key "english", nothing else. Example:
{{"english": ["quantum mechanics", "wave-particle duality", "photon behavior"]}}

Keywords:"""
        else:
            # Non-English course: need both native and English keywords
            lang_name = "Spanish" if lang_code == "es" else language.title()
            prompt = f"""Extract 3-5 academic search keywords from this module content in BOTH {lang_name} and English.

Module Title: {module_title}
Module Description: {module_description}

IMPORTANT:
1. Provide keywords in {lang_name} (native language of the course)
2. Provide equivalent keywords in English (for broader academic search)
3. Use standard academic terminology in both languages
4. Each keyword/phrase should be 1-4 words
5. Focus on the main academic concepts, not filler words

Return ONLY a JSON object with two keys: "native" and "english". Example for a Spanish course on quantum physics:
{{"native": ["mecánica cuántica", "dualidad onda-partícula", "comportamiento del fotón"], "english": ["quantum mechanics", "wave-particle duality", "photon behavior"]}}

Keywords:"""
        
        result = llm.invoke(prompt)
        content = result.content.strip()
        
        # Parse JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        
        data = json.loads(content.strip())
        
        def validate_keywords(keywords: list) -> list[str]:
            if not isinstance(keywords, list):
                return []
            return [
                k.strip() for k in keywords 
                if isinstance(k, str) and 2 < len(k) < 100
            ][:5]
        
        if is_english:
            english_kw = validate_keywords(data.get("english", []))
            logger.debug(f"Extracted English keywords: {english_kw}")
            return english_kw, english_kw
        else:
            native_kw = validate_keywords(data.get("native", []))
            english_kw = validate_keywords(data.get("english", []))
            logger.debug(f"Extracted native keywords: {native_kw}")
            logger.debug(f"Extracted English keywords: {english_kw}")
            return native_kw, english_kw
        
    except Exception as e:
        logger.debug(f"Keyword extraction failed: {e}")
        return [], []


def _extract_english_search_keywords(
    module_title: str,
    module_description: str,
    llm_provider: str = "mistral",
) -> list[str]:
    """
    Extract English academic search keywords from module content.
    
    Legacy wrapper for backward compatibility.
    
    Args:
        module_title: Module title (may be in any language)
        module_description: Module description
        llm_provider: LLM provider to use
        
    Returns:
        List of 3-5 English academic keywords/phrases
    """
    _, english_keywords = _extract_search_keywords(
        module_title, module_description, "English", llm_provider
    )
    return english_keywords


def _build_article_search_query(
    module_title: str,
    english_keywords: list[str],
) -> str:
    """
    Build an optimized search query for academic article APIs.
    
    Args:
        module_title: Original module title
        english_keywords: Extracted English keywords
        
    Returns:
        Optimized search query string (2-5 words)
    """
    if english_keywords:
        # Use the first 2-3 keywords for focused search
        query = " ".join(english_keywords[:2])
        # Keep query short for API compatibility
        if len(query) > 50:
            query = english_keywords[0]
        return query
    
    # Fallback: clean up the title
    title = module_title
    
    # Remove question marks and special chars
    for char in ["?", "!", "¿", "¡", ":", ";", "(", ")"]:
        title = title.replace(char, " ")
    
    # Remove common filler words
    fillers = [
        "los fundamentos de", "introducción a", "el problema de",
        "¿por qué", "por qué", "¿cómo", "cómo", "¿qué", "qué",
        "the basics of", "introduction to", "the problem of",
        "why", "how", "what is", "understanding",
    ]
    title_lower = title.lower()
    for filler in fillers:
        if title_lower.startswith(filler):
            title = title[len(filler):].strip()
            title_lower = title.lower()
    
    # Take first ~40 chars
    words = title.split()[:5]
    return " ".join(words)


# ============================================================
# Article Relevance Filtering
# ============================================================

# Blacklist of venue/journal keywords that indicate unrelated content
IRRELEVANT_VENUE_KEYWORDS = [
    # Medical/Healthcare (common false positives)
    "aids", "hiv", "cardiology", "cardiac", "heart", "cancer", "oncology",
    "diabetes", "nursing", "dental", "dentistry", "veterinary", "agriculture",
    "pediatric", "obstetric", "gynecology", "surgery", "surgical", "clinical trial",
    "nephrology", "kidney", "liver", "hepatology", "dermatology", "skin",
    "ophthalmology", "eye", "otolaryngology", "ear", "nose", "throat",
    "gastroenterology", "digestive", "pulmonology", "lung", "respiratory",
    "rheumatology", "arthritis", "endocrinology", "hormone", "urology",
    "immunology vaccine", "tropical medicine", "parasitology", "infectious disease",
    "psychiatry", "psychiatric", "mental health", "psychology clinical",
    "radiology", "radiotherapy", "oncologic", "tumor", "tumour",
    "pharmacology", "drug discovery", "pharmaceutical", "therapeutics",
    "epidemiology", "public health", "health policy", "medical education",
    "anesthesiology", "anesthesia", "intensive care", "emergency medicine",
    "pathology", "histology", "cytology", "microbiology clinical",
    "gerontology", "geriatric", "palliative", "hospice",
    # Law/Legal (common false positive for Spanish "materia")
    "derecho", "jurisprudencia", "legal", "law review", "tribunal",
    "contrato", "contract law", "civil law", "criminal law", "penal",
    # Business/Economics unrelated
    "marketing", "advertising", "retail", "consumer behavior",
    "human resources", "management accounting", "auditing",
    # Agriculture/Food
    "agronomy", "crop science", "animal science", "livestock",
    "food science", "nutrition", "dietary", "culinary",
    # Unrelated Arts
    "art history", "music theory", "dance", "theater studies",
    "fashion", "textile", "interior design",
]

# Generic off-topic keywords that indicate irrelevant content
# These appear in titles/abstracts and suggest the article is about something else
OFF_TOPIC_TITLE_KEYWORDS = [
    # Education/Pedagogy (common false positives when searching in Spanish)
    "pedagógico", "pedagógica", "pedagogía", "pedagogical",
    "campus virtual", "virtual campus", "plataforma virtual",
    "didáctica", "didáctico", "didactic",
    "enseñanza de", "teaching of", "aprendizaje de",
    "evaluación educativa", "educational assessment",
    "formación docente", "teacher training",
    "aula virtual", "virtual classroom",
    "educación a distancia", "distance education",
    "e-learning", "moodle", "blackboard",
    # Case studies / reviews that are meta-content
    "estudio de caso en", "case study in",
    "revisión bibliográfica", "literature review",
    "estado del arte en", "state of the art in",
    # Public health (unless topic is health-related)
    "salud pública", "public health",
    "enfermería", "nursing",
    "atención primaria", "primary care",
    # Generic business/management
    "gestión empresarial", "business management",
    "recursos humanos", "human resources",
    "plan de negocios", "business plan",
]

# Minimum citation count threshold for quality filtering
MIN_CITATION_COUNT = 10

# Minimum book rating threshold for quality filtering (Google Books 1-5 scale)
MIN_BOOK_RATING = 4.0

# URL validation settings
URL_VALIDATION_TIMEOUT = 5  # seconds


# ============================================================
# URL Validation
# ============================================================

def _validate_url(url: str, timeout: int = URL_VALIDATION_TIMEOUT) -> bool:
    """
    Validate a URL by making a HEAD request.
    
    Args:
        url: URL to validate
        timeout: Request timeout in seconds
        
    Returns:
        True if URL is accessible, False otherwise
    """
    if not url:
        return False
    
    try:
        # Use HEAD request to avoid downloading content
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BibliographyBot/1.0)"}
        )
        return response.status_code < 400
    except requests.exceptions.Timeout:
        logger.debug(f"URL validation timeout: {url}")
        return False
    except requests.exceptions.ConnectionError:
        logger.debug(f"URL validation connection error: {url}")
        return False
    except Exception as e:
        logger.debug(f"URL validation failed for {url}: {e}")
        return False


def _get_alternative_url(book: BookReference) -> str | None:
    """
    Get an alternative URL for a book if the primary URL fails.
    
    Args:
        book: BookReference with ISBN or title
        
    Returns:
        Alternative URL or None
    """
    # Try to build a Google Books URL from ISBN
    if book.isbn_13:
        return f"https://www.google.com/books/edition/_/{book.isbn_13}"
    elif book.isbn:
        return f"https://www.google.com/search?tbm=bks&q=isbn:{book.isbn}"
    
    # Try Open Library search URL as fallback
    if book.title:
        from urllib.parse import quote_plus
        return f"https://openlibrary.org/search?q={quote_plus(book.title)}"
    
    return None

# Whitelist of academic domains relevant to STEM topics
STEM_VENUE_KEYWORDS = [
    # Physics
    "physics", "physical review", "physica", "phys. rev", "phys rev",
    "quantum", "optics", "photonics", "nuclear", "atomic",
    # Computer Science
    "computer", "computing", "computational", "ieee", "acm",
    "machine learning", "artificial intelligence", "neural",
    "software", "algorithm", "data science", "informatics",
    # Mathematics
    "mathematics", "mathematical", "math", "algebra", "calculus",
    "geometry", "topology", "statistics", "probability",
    # Engineering
    "engineering", "electronics", "electrical", "mechanical",
    "materials science", "nanotechnology", "robotics",
    # Chemistry
    "chemistry", "chemical", "molecular", "biochemistry",
    # General Science
    "science", "scientific", "nature", "research", "arxiv",
    "proceedings", "journal", "transactions", "letters",
]


def _extract_topic_keywords(module_title: str) -> set[str]:
    """
    Extract key topic words from module title for relevance checking.
    
    Args:
        module_title: Module title string
        
    Returns:
        Set of lowercase keywords
    """
    # Remove common Spanish/English filler words
    stopwords = {
        "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
        "the", "of", "in", "and", "to", "a", "an", "for", "with", "on",
        "es", "son", "como", "que", "por", "para", "su", "sus",
        "is", "are", "as", "what", "how", "why", "their", "its",
        "introducción", "introduction", "fundamentos", "fundamentals",
        "básico", "básicos", "basic", "basics", "avanzado", "advanced",
        "problema", "problem", "aplicación", "aplicaciones", "application", "applications",
    }
    
    # Clean and split
    title_clean = module_title.lower()
    for char in [":", "-", "–", "?", "!", "¿", "¡", ",", ".", "(", ")"]:
        title_clean = title_clean.replace(char, " ")
    
    words = title_clean.split()
    keywords = {w for w in words if len(w) > 3 and w not in stopwords}
    
    return keywords


def _is_article_relevant(
    article: ArticleResult,
    topic_keywords: set[str],
    core_keywords: set[str] | None = None,
    require_citations: bool = True,
) -> bool:
    """
    Check if an article is relevant to the topic.
    
    Uses strict filtering to avoid false positives:
    1. Check for off-topic title keywords (pedagogy, education methodology, etc.)
    2. Require at least 1 core keyword in the TITLE (not just abstract)
    3. Filter by citation count for quality
    
    Args:
        article: ArticleResult to check
        topic_keywords: Set of all topic keywords from module
        core_keywords: Set of CORE keywords that MUST appear (subset of topic_keywords)
        require_citations: Whether to require minimum citations (default True)
        
    Returns:
        True if article appears relevant, False otherwise
    """
    venue = (article.get("venue") or "").lower()
    title = (article.get("title") or "").lower()
    snippet = (article.get("snippet") or "").lower()
    abstract = (article.get("abstract") or "").lower()
    citation_count = article.get("citation_count") or 0
    source = article.get("source", "")
    
    # Use core_keywords if provided, otherwise use first 2 topic keywords as core
    if core_keywords is None and topic_keywords:
        core_keywords = set(list(topic_keywords)[:2])
    elif core_keywords is None:
        core_keywords = set()
    
    # ========== STRICT TITLE CHECKS ==========
    
    # Check if title contains OFF-TOPIC keywords (pedagogy, virtual campus, etc.)
    for keyword in OFF_TOPIC_TITLE_KEYWORDS:
        if keyword in title:
            logger.debug(f"Filtering article '{article.get('title')[:50]}' - off-topic title keyword: {keyword}")
            return False
    
    # Check if venue contains blacklisted keywords
    for keyword in IRRELEVANT_VENUE_KEYWORDS:
        if keyword in venue:
            logger.debug(f"Filtering article '{article.get('title')[:50]}' - irrelevant venue: {venue}")
            return False
    
    # Check if title contains blacklisted venue keywords (stricter check)
    for keyword in IRRELEVANT_VENUE_KEYWORDS:
        if keyword in title and len(keyword) > 4:
            logger.debug(f"Filtering article '{article.get('title')[:50]}' - irrelevant title keyword: {keyword}")
            return False
    
    # ========== TITLE + ABSTRACT RELEVANCE CHECK ==========
    
    # Count keyword matches in TITLE specifically
    title_matches = sum(1 for kw in topic_keywords if kw in title)
    
    # Count matches in full content (title + abstract)
    content_to_check = f"{title} {snippet} {abstract}"
    total_matches = sum(1 for kw in topic_keywords if kw in content_to_check)
    core_matches = sum(1 for kw in core_keywords if kw in content_to_check)
    
    # Also check for individual words from multi-word keywords (for better Spanish matching)
    # e.g., "dualidad onda-partícula" -> also match "onda", "partícula", "dualidad"
    extended_matches = 0
    for kw in topic_keywords:
        words = kw.split()
        if len(words) > 1:
            # Check if ANY significant word from the keyword is in the title
            for word in words:
                if len(word) > 4 and word in title:
                    extended_matches += 1
                    break
    
    # RELEVANCE REQUIREMENT:
    # Option 1: At least 1 keyword in the title
    # Option 2: At least 1 keyword match in content + partial word match in title
    # Option 3: At least 2 keyword matches in content (for articles with technical titles)
    has_title_match = title_matches > 0 or extended_matches > 0
    has_strong_content_match = total_matches >= 2
    
    if not has_title_match and not has_strong_content_match:
        logger.debug(f"Filtering article '{article.get('title')[:50]}' - insufficient relevance")
        return False
    
    # Require at least 1 core keyword somewhere in content (title or abstract)
    if core_keywords and core_matches == 0:
        logger.debug(f"Filtering article '{article.get('title')[:50]}' - no core keyword match")
        return False
    
    # ========== CITATION QUALITY CHECK ==========
    
    if require_citations and citation_count < MIN_CITATION_COUNT:
        if citation_count == 0 and source in ["semanticscholar", "openalex", "arxiv"]:
            has_venue = bool(venue)
            has_abstract = bool(abstract)
            if not (has_venue and has_abstract):
                logger.debug(f"Filtering article '{article.get('title')[:50]}' - no citations and missing metadata")
                return False
    
    # ========== VENUE QUALITY CHECK ==========
    
    is_stem_venue = any(kw in venue for kw in STEM_VENUE_KEYWORDS)
    if not is_stem_venue and venue:
        # If venue exists but is not STEM, require stronger keyword match
        if total_matches < 2 and len(topic_keywords) > 2:
            logger.debug(f"Filtering article '{article.get('title')[:50]}' - non-STEM venue with weak match")
            return False
    
    return True


def _is_book_relevant(
    book_title: str,
    topic_keywords: set[str],
    core_keywords: set[str] | None = None,
) -> bool:
    """
    Check if a book title is relevant to the topic.
    
    Validates that the book is actually about the topic, not tangentially related.
    For example, "Principios de Química" is NOT relevant to "Quantum Mechanics".
    
    Args:
        book_title: Title of the book
        topic_keywords: Set of topic keywords from module
        core_keywords: Set of CORE keywords that SHOULD appear in title
        
    Returns:
        True if book appears relevant, False otherwise
    """
    title_lower = book_title.lower()
    
    # Use core_keywords if provided, otherwise use first 2 topic keywords
    if core_keywords is None and topic_keywords:
        core_keywords = set(list(topic_keywords)[:2])
    elif core_keywords is None:
        core_keywords = set()
    
    # Check for off-topic title keywords
    for keyword in OFF_TOPIC_TITLE_KEYWORDS:
        if keyword in title_lower:
            logger.debug(f"Filtering book '{book_title[:50]}' - off-topic keyword: {keyword}")
            return False
    
    # Check for clearly off-topic subjects
    # These are common false positives when topic keywords are too broad
    off_topic_subjects = [
        # Chemistry when looking for physics (unless topic includes chemistry)
        ("química", {"física", "quantum", "cuántica", "physics"}),
        ("chemistry", {"physics", "quantum"}),
        # Medicine when looking for science
        ("medicina", {"física", "quantum", "cuántica", "química", "biología"}),
        ("medicine", {"physics", "quantum", "chemistry", "biology"}),
        # Education methodology
        ("didáctica de", set()),  # Always filter unless it's the actual topic
        ("teaching of", set()),
        ("enseñanza de", set()),
    ]
    
    for off_topic, exceptions in off_topic_subjects:
        if off_topic in title_lower:
            # Check if any exception keyword is in the topic
            if not any(exc in kw for kw in topic_keywords for exc in exceptions):
                # Also check if the off-topic term itself is a core keyword
                if off_topic not in core_keywords:
                    logger.debug(f"Filtering book '{book_title[:50]}' - off-topic subject: {off_topic}")
                    return False
    
    # Count keyword matches in title
    title_matches = sum(1 for kw in topic_keywords if kw in title_lower)
    core_matches = sum(1 for kw in core_keywords if kw in title_lower)
    
    # Require at least 1 topic keyword to appear in the title
    if title_matches == 0 and topic_keywords:
        logger.debug(f"Filtering book '{book_title[:50]}' - no topic keyword in title")
        return False
    
    return True


# ============================================================
# APA 7 Formatting Functions
# ============================================================

def _format_author_for_apa(author: str) -> str:
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


def _format_authors_apa(authors: list[str]) -> str:
    """
    Format a list of authors according to APA 7 rules.
    
    Args:
        authors: List of author names
        
    Returns:
        Formatted authors string
    """
    if not authors:
        return "Unknown Author"
    
    formatted = [_format_author_for_apa(a) for a in authors]
    
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    else:
        # APA 7: list up to 20 authors, use "..." for 21+
        if len(formatted) <= 20:
            return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
        else:
            first_19 = ", ".join(formatted[:19])
            return f"{first_19}, ... {formatted[-1]}"


def _format_book_apa7(book: BookReference) -> str:
    """
    Format a book reference in APA 7 style.
    
    Format: Author, A. A. (Year). *Book title* (Edition). Publisher. URL
    
    Args:
        book: BookReference with book metadata
        
    Returns:
        Formatted APA 7 citation string
    """
    authors_str = _format_authors_apa(book.authors)
    year_str = f"({book.year})" if book.year else "(n.d.)"
    title_str = book.title
    
    # Add edition if present
    if book.edition:
        title_str += f" ({book.edition})"
    
    # Format publisher
    publisher_str = book.publisher if book.publisher else ""
    
    # Build citation
    citation_parts = [authors_str, year_str, f"{title_str}."]
    if publisher_str:
        citation_parts.append(f"{publisher_str}.")
    
    # Add DOI or URL if present
    if book.doi:
        citation_parts.append(f"https://doi.org/{book.doi}")
    elif book.url:
        citation_parts.append(book.url)
    
    return " ".join(citation_parts)


def _format_article_apa7(article: ArticleResult) -> str:
    """
    Format an article in APA 7 style.
    
    Format: Author, A. A. (Year). Article title. *Journal Name*, vol(issue), pages. DOI/URL
    
    Args:
        article: ArticleResult from article search
        
    Returns:
        Formatted APA 7 citation string
    """
    authors_str = _format_authors_apa(article["authors"])
    year_str = f"({article['year']})" if article.get("year") else "(n.d.)"
    title_str = article["title"]
    
    # Build citation parts
    citation_parts = [authors_str, year_str, f"{title_str}."]
    
    # Add venue/journal if present
    venue = article.get("venue")
    if venue:
        citation_parts.append(f"*{venue}*.")
    
    # Add DOI or URL
    doi = article.get("doi")
    url = article.get("url", "")
    if doi:
        if not doi.startswith("http"):
            citation_parts.append(f"https://doi.org/{doi}")
        else:
            citation_parts.append(doi)
    elif url:
        citation_parts.append(url)
    
    return " ".join(citation_parts)


# ============================================================
# Book Search Functions
# ============================================================

def _book_result_to_reference(
    result: BookResult,
    relevance: str = "",
) -> BookReference:
    """
    Convert Open Library BookResult to BookReference model.
    
    Args:
        result: BookResult from Open Library search
        relevance: Optional relevance explanation
        
    Returns:
        BookReference with APA citation
    """
    book = BookReference(
        title=result["title"],
        authors=result["authors"],
        year=result["year"],
        publisher=result["publisher"],
        isbn=result["isbn"],
        isbn_13=result["isbn_13"],
        url=result["openlibrary_url"],
    )
    
    # Generate APA citation
    book.apa_citation = _format_book_apa7(book)
    
    return book


def _extract_module_topics(module: Module) -> str:
    """
    Extract topic list from module content for LLM context.
    
    Args:
        module: Module with submodules and sections
        
    Returns:
        Formatted string of topics
    """
    topics = []
    
    for submodule in module.submodules:
        topics.append(f"- {submodule.title}")
        for section in submodule.sections:
            topics.append(f"  - {section.title}")
            if section.description:
                topics.append(f"    ({section.description})")
    
    return "\n".join(topics)


def _search_books_by_topic_direct(
    topic_keywords: list[str],
    num_books: int,
    existing_keys: set[str],
    language: str | None = None,
    core_keywords: set[str] | None = None,
) -> tuple[list[BookReference], set[str]]:
    """
    Search for books directly by topic keywords (fallback method).
    
    Uses Open Library's subject/topic search and Google Books as fallback
    when LLM suggestions don't validate well.
    
    Args:
        topic_keywords: Keywords to search for (in target language)
        num_books: Target number of books
        existing_keys: Set of dedup keys for books already cited
        language: ISO 639-1 language code for filtering (e.g., "es", "en")
        core_keywords: Core keywords for relevance validation
        
    Returns:
        Tuple of (list of BookReference, updated existing_keys)
    """
    from tools.booksearch.openlibrary.client import search_books as ol_search_books
    from tools.booksearch.googlebooks.client import search_books as gb_search_books
    
    validated_books: list[BookReference] = []
    new_keys: set[str] = set()
    new_title_keys: set[str] = set()  # Track normalized titles for better dedup
    
    # Build topic keywords set for relevance checking
    topic_kw_set = set(kw.lower() for kw in topic_keywords)
    if core_keywords is None:
        core_keywords = set(list(topic_kw_set)[:2])
    
    for keyword in topic_keywords[:3]:  # Try first 3 keywords
        if len(validated_books) >= num_books:
            break
        
        # Try Google Books first (better recency sorting and language support)
        try:
            gb_results = gb_search_books(
                keyword, 
                max_results=8,  # More results to account for filtering
                language=language,
                order_by="newest",  # Prefer recent publications
            )
            for gb in gb_results:
                if len(validated_books) >= num_books:
                    break
                
                # Check rating (Google Books only - strict: skip unrated books)
                rating = gb.get("average_rating")
                if rating is None or rating < MIN_BOOK_RATING:
                    logger.debug(f"Skipping book '{gb['title'][:50]}' - rating: {rating} (min: {MIN_BOOK_RATING})")
                    continue
                
                # Check relevance BEFORE creating book reference
                if not _is_book_relevant(gb["title"], topic_kw_set, core_keywords):
                    continue
                
                # Check title-based deduplication FIRST (catches same book, different ISBN)
                title_key = _get_normalized_title_key(gb["title"])
                if title_key in new_title_keys:
                    logger.debug(f"Skipping duplicate title: {gb['title']}")
                    continue
                
                book = BookReference(
                    title=gb["title"],
                    authors=gb["authors"],
                    year=gb["year"],
                    publisher=gb["publisher"],
                    isbn=gb.get("isbn"),
                    isbn_13=gb.get("isbn_13"),
                    url=gb["google_books_url"],
                )
                book.apa_citation = _format_book_apa7(book)
                
                # Check ISBN-based deduplication
                dedup_key = book.get_dedup_key()
                if dedup_key in existing_keys or dedup_key in new_keys:
                    continue
                
                validated_books.append(book)
                new_keys.add(dedup_key)
                new_title_keys.add(title_key)
        except Exception as e:
            logger.debug(f"Google Books search failed for '{keyword}': {e}")
        
        # Fall back to Open Library (no language filter but good coverage)
        if len(validated_books) < num_books:
            try:
                ol_results = ol_search_books(keyword, max_results=5)
                for result in ol_results:
                    if len(validated_books) >= num_books:
                        break
                    
                    # Check relevance BEFORE creating book reference
                    if not _is_book_relevant(result["title"], topic_kw_set, core_keywords):
                        continue
                    
                    # Check title-based deduplication FIRST
                    title_key = _get_normalized_title_key(result["title"])
                    if title_key in new_title_keys:
                        logger.debug(f"Skipping duplicate title: {result['title']}")
                        continue
                    
                    book = _book_result_to_reference(result, f"Found via topic search: {keyword}")
                    
                    # Check ISBN-based deduplication
                    dedup_key = book.get_dedup_key()
                    if dedup_key in existing_keys or dedup_key in new_keys:
                        continue
                    
                    validated_books.append(book)
                    new_keys.add(dedup_key)
                    new_title_keys.add(title_key)
            except Exception as e:
                logger.debug(f"Open Library search failed for '{keyword}': {e}")
    
    updated_keys = existing_keys | new_keys
    return validated_books, updated_keys


def _search_books_for_module(
    module: Module,
    course_title: str,
    language: str,
    provider: str,
    num_books: int,
    existing_keys: set[str],
) -> tuple[list[BookReference], set[str]]:
    """
    Search for books relevant to a module.
    
    Uses language-aware search strategy:
    - For Spanish courses: LLM suggests Spanish books, validates via Google Books with language filter
    - For English courses: LLM suggests English books, validates via Open Library and Google Books
    
    Args:
        module: Module to find books for
        course_title: Course title for context
        language: Course language
        provider: LLM provider
        num_books: Target number of books
        existing_keys: Set of dedup keys for books already cited
        
    Returns:
        Tuple of (list of BookReference, updated existing_keys)
    """
    if num_books <= 0:
        return [], existing_keys
    
    # Get language code for filtering
    lang_code = _get_language_code(language)
    is_spanish = lang_code == "es"
    
    # Extract keywords in both native language and English
    native_keywords, english_keywords = _extract_search_keywords(
        module.title,
        module.description,
        language,
        provider,
    )
    
    # Build topic keywords for book relevance validation
    book_topic_keywords = set()
    for kw_list in [native_keywords, english_keywords]:
        for kw in kw_list:
            book_topic_keywords.add(kw.lower())
            # Also add individual words for broader matching
            book_topic_keywords.update(w.lower() for w in kw.split() if len(w) > 3)
    
    # Core keywords for stricter matching (first 2 from each list)
    book_core_keywords = set()
    if english_keywords:
        book_core_keywords.update(kw.lower() for kw in english_keywords[:2])
    if native_keywords:
        book_core_keywords.update(kw.lower() for kw in native_keywords[:2])
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.3}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Extract module topics
    module_topics = _extract_module_topics(module)
    
    # Build exclusion note if we have existing books
    exclusion_note = ""
    if existing_keys:
        exclusion_note = f"NOTE: Avoid suggesting books already recommended. We have {len(existing_keys)} items already."
    
    # Get LLM suggestions (prompt now requests books in course language)
    chain = book_suggestion_prompt | llm | StrOutputParser()
    
    # Request more books than needed to account for validation failures
    request_count = num_books * 3
    
    raw_suggestions = chain.invoke({
        "course_title": course_title,
        "module_title": module.title,
        "module_description": module.description,
        "module_topics": module_topics,
        "language": language,
        "num_books": request_count,
        "exclusion_note": exclusion_note,
    })
    
    # Parse LLM response
    suggestions = _parse_json_response(raw_suggestions, context="book suggestions")
    
    # Validate books and build references
    validated_books: list[BookReference] = []
    new_keys: set[str] = set()
    new_title_keys: set[str] = set()  # Track normalized titles for better dedup
    
    # Import Google Books search for language-filtered searches
    from tools.booksearch.googlebooks.client import search_books as gb_search_books
    
    for suggestion in suggestions:
        if len(validated_books) >= num_books:
            break
        
        title = suggestion.get("title", "")
        authors = suggestion.get("authors", [])
        year = suggestion.get("year")
        publisher = suggestion.get("publisher")
        relevance = suggestion.get("relevance", "")
        book_language = suggestion.get("language", lang_code)  # LLM now provides language
        
        if not title:
            continue
        
        # Try to validate via Open Library (multi-step with fallbacks)
        author_query = authors[0] if authors else None
        validated = validate_book(title, author_query)
        english_title = None
        
        # Step 2: If not found, try English translation of the title (for Open Library search)
        if not validated:
            english_title = _translate_title_to_english(title, provider)
            if english_title:
                logger.debug(f"Trying English title: '{english_title}'")
                validated = validate_book(english_title, author_query)
        
        if validated:
            book = _book_result_to_reference(validated, relevance)
        else:
            # Step 3: Try Google Books API with language filter
            # Build query with title and author
            search_query = f'intitle:"{title}"'
            if author_query:
                author_parts = author_query.split()
                last_name = author_parts[-1] if author_parts else author_query
                search_query += f' inauthor:"{last_name}"'
            
            # Search with language filter for non-English books
            google_results = gb_search_books(
                search_query,
                max_results=3,
                language=book_language if book_language != "en" else None,
            )
            
            # If no results with language filter, try without
            if not google_results and book_language != "en":
                google_results = gb_search_books(search_query, max_results=3)
            
            # Also try with English title as fallback
            if not google_results and english_title:
                search_query_en = f'intitle:"{english_title}"'
                if author_query:
                    author_parts = author_query.split()
                    last_name = author_parts[-1] if author_parts else author_query
                    search_query_en += f' inauthor:"{last_name}"'
                google_results = gb_search_books(search_query_en, max_results=3)
            
            if google_results:
                # Filter Google Books results by rating (strict: skip unrated)
                gb = None
                for candidate in google_results:
                    rating = candidate.get("average_rating")
                    if rating is not None and rating >= MIN_BOOK_RATING:
                        gb = candidate
                        break
                    else:
                        logger.debug(f"Skipping Google Books result '{candidate['title'][:50]}' - rating: {rating} (min: {MIN_BOOK_RATING})")
                
                if gb:
                    # Found in Google Books with sufficient rating - use direct link
                    book = BookReference(
                        title=title,  # Keep original title
                        authors=[_format_author_for_apa(a) for a in authors] if authors else gb["authors"],
                        year=year or gb["year"],
                        publisher=publisher or gb["publisher"],
                        isbn=gb.get("isbn"),
                        isbn_13=gb.get("isbn_13"),
                        url=gb["google_books_url"],  # Direct link!
                    )
                    book.apa_citation = _format_book_apa7(book)
                else:
                    # No Google Books results with sufficient rating
                    logger.debug(f"Skipping book '{title}' - no Google Books results with rating >= {MIN_BOOK_RATING}")
                    continue
            else:
                # Step 4: Skip book - no reliable direct link available
                logger.debug(f"Skipping book '{title}' - no direct link found")
                continue
        
        # Check book relevance to topic
        if not _is_book_relevant(book.title, book_topic_keywords, book_core_keywords):
            logger.debug(f"Skipping irrelevant book: {book.title}")
            continue
        
        # Check title-based deduplication FIRST (catches same book, different ISBN/edition)
        title_key = _get_normalized_title_key(book.title)
        if title_key in new_title_keys:
            logger.debug(f"Skipping duplicate title: {book.title}")
            continue
        
        # Check ISBN-based deduplication
        dedup_key = book.get_dedup_key()
        if dedup_key in existing_keys or dedup_key in new_keys:
            logger.debug(f"Skipping duplicate book (ISBN): {book.title}")
            continue
        
        validated_books.append(book)
        new_keys.add(dedup_key)
        new_title_keys.add(title_key)
    
    # Fallback: If we didn't find enough books, try direct topic search
    if len(validated_books) < num_books:
        remaining = num_books - len(validated_books)
        logger.debug(f"LLM found {len(validated_books)} books, trying direct search for {remaining} more")
        
        # For Spanish courses, try Spanish keywords first with language filter
        if is_spanish and native_keywords:
            fallback_books, new_keys_fb = _search_books_by_topic_direct(
                topic_keywords=native_keywords,
                num_books=remaining,
                existing_keys=existing_keys | new_keys,
                language="es",
                core_keywords=book_core_keywords,
            )
            validated_books.extend(fallback_books)
            new_keys.update(new_keys_fb)
            remaining = num_books - len(validated_books)
        
        # Then try English keywords as additional fallback
        if remaining > 0 and english_keywords:
            fallback_books, new_keys_fb = _search_books_by_topic_direct(
                topic_keywords=english_keywords,
                num_books=remaining,
                existing_keys=existing_keys | new_keys,
                language=None,  # No filter for English fallback
                core_keywords=book_core_keywords,
            )
            validated_books.extend(fallback_books)
            new_keys.update(new_keys_fb)
    
    # Sort books by recency and filter out old ones if we have enough recent
    validated_books = _sort_books_by_recency(validated_books, prefer_recent=True, num_books=num_books)
    
    updated_keys = existing_keys | new_keys
    return validated_books, updated_keys


# ============================================================
# Article Search Functions
# ============================================================

def _search_articles_for_module(
    module: Module,
    language: str,
    article_provider: str,
    num_articles: int,
    existing_keys: set[str],
    llm_provider: str = "mistral",
    validate_urls: bool = False,
) -> tuple[list[ArticleResult], set[str]]:
    """
    Search for academic articles relevant to a module.
    
    Uses language-aware search strategy:
    - For Spanish courses: Search OpenAlex with Spanish filter + native keywords,
      supplement with English results from arXiv/Semantic Scholar
    - For English courses: Search all providers with English keywords
    
    Args:
        module: Module to find articles for
        language: Course language
        article_provider: Article search provider
        num_articles: Target number of articles
        existing_keys: Set of dedup keys for items already cited
        llm_provider: LLM provider for keyword extraction
        validate_urls: Whether to validate URLs (slower but ensures links work)
        
    Returns:
        Tuple of (list of ArticleResult, updated existing_keys)
    """
    if num_articles <= 0:
        return [], existing_keys
    
    # Get language code for filtering
    lang_code = _get_language_code(language)
    is_spanish = lang_code == "es"
    
    # Extract keywords in both native language and English
    native_keywords, english_keywords = _extract_search_keywords(
        module.title,
        module.description,
        language,
        llm_provider,
    )
    
    # Build search queries
    native_query = _build_article_search_query(module.title, native_keywords) if native_keywords else None
    english_query = _build_article_search_query(module.title, english_keywords)
    
    logger.debug(f"Article search - native query: '{native_query}', english query: '{english_query}'")
    
    # Extract topic keywords for relevance filtering (combine both for better matching)
    topic_keywords = set()
    for kw_list in [native_keywords, english_keywords]:
        for kw in kw_list:
            topic_keywords.add(kw.lower())
            topic_keywords.update(w.lower() for w in kw.split() if len(w) > 3)
    
    if not topic_keywords:
        topic_keywords = _extract_topic_keywords(module.title)
    
    # Extract CORE keywords (first 2 from each list - these MUST appear in results)
    # Core keywords are the most specific/important terms
    core_keywords = set()
    if english_keywords:
        core_keywords.update(kw.lower() for kw in english_keywords[:2])
    if native_keywords:
        core_keywords.update(kw.lower() for kw in native_keywords[:2])
    
    all_results: list[ArticleResult] = []
    per_provider_count = max(num_articles, 3)
    
    if is_spanish:
        # SPANISH COURSE: Prioritize Spanish articles
        # 1. Search OpenAlex with Spanish language filter (it supports this natively)
        try:
            openalex_search = create_article_search("openalex")
            # Primary search: Spanish articles with native keywords
            if native_query:
                spanish_results = openalex_search(
                    native_query, 
                    max_results=per_provider_count * 2,  # Request more since we're filtering
                    language="es"  # OpenAlex native language filter
                )
                all_results.extend(spanish_results)
                logger.debug(f"OpenAlex (Spanish): found {len(spanish_results)} articles")
            
            # Secondary search: Spanish articles with English keywords (some Spanish papers have English titles)
            if english_query and len(all_results) < num_articles:
                spanish_results_en = openalex_search(
                    english_query,
                    max_results=per_provider_count,
                    language="es"
                )
                all_results.extend(spanish_results_en)
                logger.debug(f"OpenAlex (Spanish with EN keywords): found {len(spanish_results_en)} articles")
        except Exception as e:
            logger.debug(f"OpenAlex Spanish search failed: {e}")
        
        # 2. ALWAYS add high-quality English papers as supplements
        # Spanish academic content is limited for many STEM topics
        # We'll sort by language preference later, so Spanish results still come first
        english_providers = ["arxiv", "openalex"]  # OpenAlex without lang filter, arXiv for preprints
        for provider in english_providers:
            try:
                search_fn = create_article_search(provider)
                results = search_fn(english_query, max_results=per_provider_count, language=None)
                all_results.extend(results)
                logger.debug(f"{provider} (English supplement): found {len(results)} articles")
            except Exception as e:
                logger.debug(f"Provider {provider} failed: {e}")
    else:
        # ENGLISH/OTHER COURSE: Search all providers with English keywords
        providers = ["openalex", "arxiv", "semanticscholar"]
        for provider in providers:
            try:
                search_fn = create_article_search(provider)
                results = search_fn(english_query, max_results=per_provider_count, language=None)
                all_results.extend(results)
                logger.debug(f"Provider {provider}: found {len(results)} articles")
            except Exception as e:
                logger.debug(f"Provider {provider} failed: {e}")
    
    # Filter by relevance and deduplicate across all providers
    articles: list[ArticleResult] = []
    new_keys: set[str] = set()
    
    # Sort: Spanish articles first (for Spanish courses), then by recency, then by citations
    def sort_key(a):
        art_lang = a.get("language", "")
        is_target_lang = 1 if (is_spanish and art_lang == "es") else 0
        recency = _get_recency_score(a.get("year"), is_book=False)
        citations = a.get("citation_count") or 0
        # Priority: language match, then recency, then citations
        return (-is_target_lang, -recency, -citations)  # Negative for descending
    
    all_results.sort(key=sort_key)
    
    for article in all_results:
        if len(articles) >= num_articles:
            break
        
        # Check relevance first (filter out false positives)
        # Pass core_keywords for stricter title-based matching
        if not _is_article_relevant(article, topic_keywords, core_keywords):
            continue
        
        # Create dedup key using normalized title (catches same article from different providers)
        title_normalized = _get_normalized_title_key(article["title"])
        dedup_key = f"article:{title_normalized}"
        
        if dedup_key in existing_keys or dedup_key in new_keys:
            logger.debug(f"Skipping duplicate article: {article['title']}")
            continue
        
        # Optional URL validation (slower but ensures links work)
        if validate_urls and article.get("url"):
            if not _validate_url(article["url"]):
                logger.debug(f"Skipping article with invalid URL: {article['title'][:50]}")
                continue
        
        articles.append(article)
        new_keys.add(dedup_key)
    
    updated_keys = existing_keys | new_keys
    return articles, updated_keys


def _article_to_bibliography_item(article: ArticleResult) -> BibliographyItem:
    """
    Convert ArticleResult to BibliographyItem.
    
    Args:
        article: ArticleResult from search
        
    Returns:
        BibliographyItem with APA 7 citation
    """
    # Build URL (prefer DOI)
    doi = article.get("doi")
    url = article.get("url", "")
    if doi and not doi.startswith("http"):
        url = f"https://doi.org/{doi}"
    elif doi:
        url = doi
    
    return BibliographyItem(
        title=article["title"],
        url=url,
        apa_citation=_format_article_apa7(article),
        item_type="article",
    )


def _book_to_bibliography_item(book: BookReference) -> BibliographyItem:
    """
    Convert BookReference to BibliographyItem.
    
    Args:
        book: BookReference with full metadata
        
    Returns:
        BibliographyItem with APA 7 citation
    """
    return BibliographyItem(
        title=book.title,
        url=book.url or "",
        apa_citation=book.apa_citation or _format_book_apa7(book),
        item_type="book",
    )


# ============================================================
# Main Generation Functions
# ============================================================

def generate_module_bibliography(
    module: Module,
    course_title: str,
    language: str,
    provider: str = "mistral",
    num_books: int = 5,
    num_articles: int = 5,
    article_provider: str = "openalex",
    existing_keys: set[str] | None = None,
) -> tuple[ModuleBibliography, set[str]]:
    """
    Generate bibliography for a single module.
    
    Uses LLM to suggest books, validates via Open Library API,
    and searches for academic articles. Skips items in existing_keys.
    
    Args:
        module: Module to generate bibliography for
        course_title: Course title for context
        language: Course language
        provider: LLM provider for book suggestions
        num_books: Target number of books
        num_articles: Target number of articles
        article_provider: Article search provider
        existing_keys: Set of dedup keys for items already cited
        
    Returns:
        Tuple of (ModuleBibliography, updated existing_keys set)
    """
    if existing_keys is None:
        existing_keys = set()
    
    # Search for books
    books, existing_keys = _search_books_for_module(
        module=module,
        course_title=course_title,
        language=language,
        provider=provider,
        num_books=num_books,
        existing_keys=existing_keys,
    )
    
    # Search for articles
    articles, existing_keys = _search_articles_for_module(
        module=module,
        language=language,
        article_provider=article_provider,
        num_articles=num_articles,
        existing_keys=existing_keys,
        llm_provider=provider,
    )
    
    # Create module bibliography (for backward compatibility)
    module_bib = ModuleBibliography(
        module_index=module.index,
        module_title=module.title,
        books=books,
    )
    
    return module_bib, existing_keys


def _create_module_bibliography_embed(
    module: Module,
    books: list[BookReference],
    articles: list[ArticleResult],
    language: str,
) -> ModuleBibliographyEmbed:
    """
    Create ModuleBibliographyEmbed from books and articles.
    
    Books are listed first, then articles.
    
    Args:
        module: Module for context
        books: List of BookReference objects
        articles: List of ArticleResult objects
        language: Course language for query
        
    Returns:
        ModuleBibliographyEmbed for embedding in module
    """
    # Create search query
    if language.lower() in ["español", "spanish", "es"]:
        query = f"libro académico {module.title} site:.edu"
    else:
        query = f"academic book {module.title} site:.edu"
    
    # Convert to BibliographyItems: books first, then articles
    items = [_book_to_bibliography_item(b) for b in books]
    items += [_article_to_bibliography_item(a) for a in articles]
    
    return ModuleBibliographyEmbed(
        type="biblio",
        query=query,
        content=items,
    )


def _deduplicate_bibliography(
    module_results: dict[int, tuple[list[BookReference], list]],
) -> dict[int, tuple[list[BookReference], list]]:
    """
    Post-process parallel bibliography results to remove cross-module duplicates.
    
    Each module generated its bibliography independently, so the same book or
    article may appear in multiple modules. This pass keeps only the first
    occurrence (by module index order).
    
    Args:
        module_results: Dict mapping module index to (books, articles) tuple.
        
    Returns:
        Deduplicated module_results (same structure, duplicates removed).
    """
    seen_book_keys: set[str] = set()
    seen_title_keys: set[str] = set()
    seen_article_keys: set[str] = set()

    deduped: dict[int, tuple[list[BookReference], list]] = {}

    for idx in sorted(module_results.keys()):
        books, articles = module_results[idx]

        unique_books: list[BookReference] = []
        for book in books:
            dedup_key = book.get_dedup_key()
            title_key = _get_normalized_title_key(book.title)
            if dedup_key not in seen_book_keys and title_key not in seen_title_keys:
                unique_books.append(book)
                seen_book_keys.add(dedup_key)
                seen_title_keys.add(title_key)

        unique_articles = []
        for article in articles:
            title_key = _get_normalized_title_key(article["title"])
            article_key = f"article:{title_key}"
            if article_key not in seen_article_keys:
                unique_articles.append(article)
                seen_article_keys.add(article_key)

        deduped[idx] = (unique_books, unique_articles)

    return deduped


def generate_course_bibliography(
    state: CourseState,
    provider: str | None = None,
    books_per_module: int | None = None,
    articles_per_module: int | None = None,
    article_provider: str | None = None,
    embed_in_modules: bool = True,
) -> CourseBibliography:
    """
    Generate bibliography for entire course with deduplication.
    
    Processes modules in parallel using ThreadPoolExecutor, then deduplicates
    across modules in a post-processing step.
    Also embeds bibliography data directly in each Module.
    
    Args:
        state: CourseState with modules
        provider: LLM provider (defaults to state.config.text_llm_provider)
        books_per_module: Books per module (defaults to config)
        articles_per_module: Articles per module (defaults to config)
        article_provider: Article search provider (defaults to config)
        embed_in_modules: Whether to embed bibliography in each module
        
    Returns:
        CourseBibliography with per-module and deduplicated master list
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    provider = provider or state.config.text_llm_provider
    books_per_module = books_per_module or state.config.bibliography_books_per_module
    articles_per_module = articles_per_module or state.config.bibliography_articles_per_module
    article_provider = article_provider or state.config.article_search_provider
    
    print(f"📚 Generating bibliography for {len(state.modules)} modules...")
    print(f"   Target: {books_per_module} books + {articles_per_module} articles per module")
    print(f"   Providers: LLM={provider}, Articles={article_provider}")
    
    def _process_module(idx: int, module: Module) -> tuple[int, list[BookReference], list]:
        print(f"\n   📖 Module {idx + 1}/{len(state.modules)}: {module.title}")
        books, _ = _search_books_for_module(
            module=module,
            course_title=state.title,
            language=state.config.language,
            provider=provider,
            num_books=books_per_module,
            existing_keys=set(),
        )
        print(f"      ✓ Module {idx + 1}: Found {len(books)} books")
        articles, _ = _search_articles_for_module(
            module=module,
            language=state.config.language,
            article_provider=article_provider,
            num_articles=articles_per_module,
            existing_keys=set(),
            llm_provider=provider,
        )
        print(f"      ✓ Module {idx + 1}: Found {len(articles)} articles")
        return idx, books, articles

    raw_results: dict[int, tuple[list[BookReference], list]] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_process_module, idx, module): idx
            for idx, module in enumerate(state.modules)
        }
        for future in as_completed(futures):
            idx, books, articles = future.result()
            raw_results[idx] = (books, articles)

    deduped_results = _deduplicate_bibliography(raw_results)

    module_bibliographies: list[ModuleBibliography] = []
    all_books: list[BookReference] = []

    for idx in range(len(state.modules)):
        books, articles = deduped_results[idx]
        module = state.modules[idx]

        module_bib = ModuleBibliography(
            module_index=module.index,
            module_title=module.title,
            books=books,
        )
        module_bibliographies.append(module_bib)
        all_books.extend(books)

        if embed_in_modules and (books or articles):
            module.bibliography = _create_module_bibliography_embed(
                module=module,
                books=books,
                articles=articles,
                language=state.config.language,
            )
    
    all_books_sorted = sorted(
        all_books,
        key=lambda b: b.authors[0].lower() if b.authors else "zzz"
    )
    
    bibliography = CourseBibliography(
        modules=module_bibliographies,
        all_books=all_books_sorted,
    )
    
    print(f"\n✅ Bibliography complete!")
    print(f"   Total unique books: {len(all_books_sorted)}")
    
    return bibliography


def generate_bibliography_node(
    state: CourseState,
    config: Optional[RunnableConfig] = None,
) -> CourseState:
    """
    LangGraph node for bibliography generation.
    
    Generates bibliography for all modules and stores in state.
    Only runs if state.config.generate_bibliography is True.
    
    Args:
        state: CourseState with modules
        config: LangGraph runtime config
        
    Returns:
        Updated CourseState with bibliography
    """
    if not state.config.generate_bibliography:
        print("📚 Bibliography generation disabled, skipping...")
        return state
    
    bibliography = generate_course_bibliography(state)
    state.bibliography = bibliography
    
    return state
