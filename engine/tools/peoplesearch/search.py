"""
People search: find relevant people for a topic using LLM + Wikipedia.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import PersonResult
from .suggester import suggest_people, generate_description
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
    
    1. LLM suggests notable people related to the topic
    2. Validates each person against Wikipedia
    3. Filters out people without Wikipedia images
    4. Generates concise descriptions in target language
    
    Args:
        topic: Topic to find relevant people for
        max_results: Maximum number of people to return
        language: Target language code for descriptions
        llm_provider: LLM provider (mistral, gemini, openai, etc.)
        concurrency: Number of parallel API calls
        
    Returns:
        List of PersonResult with validated Wikipedia information
    """
    # Request more candidates to account for validation failures
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
    
    logger.info(f"Got {len(candidates)} candidates, validating against Wikipedia...")
    
    # Step 2: Validate candidates against Wikipedia and collect raw data
    validated_raw: list[dict] = []
    
    def validate_candidate(name: str) -> dict | None:
        info = get_person_info(name)
        if info and info.get("image"):
            return {
                "name": info["name"],
                "extract": info["extract"],
                "wikiUrl": info["wikiUrl"],
                "image": info["image"],
            }
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
                    validated_raw.append(result)
                    logger.debug(f"✓ Validated: {name}")
                    if len(validated_raw) >= max_results:
                        break
                else:
                    logger.debug(f"✗ Skipped (no image): {name}")
            except Exception as e:
                logger.error(f"Error validating '{name}': {e}")
    
    validated_raw = validated_raw[:max_results]
    logger.info(f"Validated {len(validated_raw)} people with Wikipedia images")
    
    # Step 3: Generate concise descriptions in target language
    logger.info(f"Generating descriptions in {language}...")
    
    def create_person_result(raw: dict) -> PersonResult:
        description = generate_description(
            name=raw["name"],
            wiki_extract=raw["extract"],
            language=language,
            llm_provider=llm_provider,
        )
        return PersonResult(
            name=raw["name"],
            description=description,
            wikiUrl=raw["wikiUrl"],
            image=raw["image"],
        )
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(create_person_result, validated_raw))
    
    return results
