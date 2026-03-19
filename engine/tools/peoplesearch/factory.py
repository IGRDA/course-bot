"""
Factory for searching relevant people on a topic.

Orchestrates LLM-based people suggestion with Wikipedia validation
and optional translation.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .models import PersonResult
from .suggester import suggest_people, translate_description
from .wikipedia.client import get_person_info

logger = logging.getLogger(__name__)


def search_relevant_people(
    topic: str,
    max_results: int = 5,
    language: str = "en",
    llm_provider: str = "mistral",
    concurrency: int = 5,
) -> list[PersonResult]:
    """
    Search for people relevant to a topic using LLM + Wikipedia.
    
    This function:
    1. Uses an LLM to suggest notable people related to the topic
    2. Validates each person against Wikipedia and fetches their info
    3. Filters out people without Wikipedia images
    4. Optionally translates descriptions to target language
    
    Args:
        topic: The topic to find relevant people for (e.g., "quantum physics")
        max_results: Maximum number of people to return (default: 5)
        language: Target language code for descriptions (default: "en")
        llm_provider: LLM provider for suggestion/translation (default: "gemini")
        concurrency: Number of parallel Wikipedia API calls (default: 5)
        
    Returns:
        List of PersonResult objects with validated Wikipedia information
        
    Example:
        >>> people = search_relevant_people(
        ...     topic="international trade policy",
        ...     max_results=3,
        ...     language="es"
        ... )
        >>> for person in people:
        ...     print(f"{person.name}: {person.description[:50]}...")
    """
    # Request more candidates than needed to account for validation failures
    candidate_count = max_results * 2 + 3
    
    logger.info(f"Suggesting {candidate_count} candidates for topic: {topic}")
    
    # Step 1: Get LLM suggestions
    candidates = suggest_people(
        topic=topic,
        count=candidate_count,
        llm_provider=llm_provider,
    )
    
    if not candidates:
        logger.warning(f"No candidates suggested for topic: {topic}")
        return []
    
    logger.info(f"Got {len(candidates)} candidate names, validating against Wikipedia...")
    
    # Step 2: Validate candidates against Wikipedia (parallel)
    validated: list[PersonResult] = []
    
    def validate_candidate(name: str) -> PersonResult | None:
        """Validate a single candidate and return PersonResult if valid."""
        info = get_person_info(name)
        if info and info.get("image"):
            return PersonResult(
                name=info["name"],
                description=info["extract"],
                wikiUrl=info["wikiUrl"],
                image=info["image"],
            )
        return None
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_name = {
            executor.submit(validate_candidate, name): name 
            for name in candidates
        }
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                if result:
                    validated.append(result)
                    logger.debug(f"✓ Validated: {name}")
                    
                    # Stop early if we have enough
                    if len(validated) >= max_results:
                        break
                else:
                    logger.debug(f"✗ Skipped (no image): {name}")
            except Exception as e:
                logger.error(f"Error validating '{name}': {e}")
    
    # Limit to max_results
    validated = validated[:max_results]
    
    logger.info(f"Validated {len(validated)} people with Wikipedia images")
    
    # Step 3: Translate descriptions if needed
    if language.lower() != "en" and validated:
        logger.info(f"Translating descriptions to {language}...")
        
        def translate_person(person: PersonResult) -> PersonResult:
            """Translate a person's description."""
            translated_desc = translate_description(
                person.description,
                language,
                llm_provider=llm_provider,
            )
            return PersonResult(
                name=person.name,
                description=translated_desc,
                wikiUrl=person.wikiUrl,
                image=person.image,
            )
        
        # Translate in parallel
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            validated = list(executor.map(translate_person, validated))
    
    return validated


# Type alias for the search function
PeopleSearchFunc = Callable[[str, int], list[PersonResult]]


def create_people_search(
    language: str = "en",
    llm_provider: str = "mistral",
) -> PeopleSearchFunc:
    """
    Create a people search function with preset configuration.
    
    Args:
        language: Target language code for descriptions
        llm_provider: LLM provider for suggestion/translation
        
    Returns:
        A search function that accepts (topic: str, max_results: int)
        
    Example:
        >>> search = create_people_search(language="es", llm_provider="gemini")
        >>> people = search("machine learning", 5)
    """
    def search_fn(topic: str, max_results: int = 5) -> list[PersonResult]:
        return search_relevant_people(
            topic=topic,
            max_results=max_results,
            language=language,
            llm_provider=llm_provider,
        )
    
    return search_fn

