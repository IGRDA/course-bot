"""
LLM-based people suggester for topic-based people search.

Uses an LLM to suggest relevant people for a given topic,
returning candidate names for Wikipedia validation.
"""

import logging

from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from LLMs.text2text.factory import create_text_llm

from .models import PeopleSuggestionResponse

logger = logging.getLogger(__name__)


SUGGESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at identifying notable people relevant to academic and professional topics.

Given a topic, suggest people who are:
1. Notable figures historically associated with the topic
2. Contemporary experts, researchers, or practitioners
3. People with significant contributions or influence in the field

Requirements:
- Use the person's full name as it would appear on their Wikipedia page (English Wikipedia)
- Include a mix of historical and contemporary figures when relevant
- Focus on people who are likely to have Wikipedia pages with photos
- Avoid fictional characters or very obscure individuals

{format_instructions}""",
        ),
        ("human", "Topic: {topic}\n\nSuggest {count} notable people relevant to this topic."),
    ]
)


def suggest_people(
    topic: str,
    count: int = 10,
    llm_provider: str = "mistral",
    temperature: float = 0.3,
) -> list[str]:
    """
    Use an LLM to suggest notable people relevant to a topic.

    Args:
        topic: The topic to find relevant people for
        count: Number of people to suggest (will request more to account for validation failures)
        llm_provider: LLM provider to use (gemini, openai, mistral, etc.)
        temperature: LLM temperature for generation (lower = more deterministic)

    Returns:
        List of person names suggested by the LLM

    Example:
        >>> names = suggest_people("international trade policy", count=5)
        >>> print(names)
        ['Paul Krugman', 'Dani Rodrik', 'Joseph Stiglitz', ...]
    """
    parser = PydanticOutputParser(pydantic_object=PeopleSuggestionResponse)

    prompt = SUGGESTION_PROMPT.partial(format_instructions=parser.get_format_instructions())

    try:
        llm = create_text_llm(provider=llm_provider, temperature=temperature)
        chain = prompt | llm | StrOutputParser()

        raw = chain.invoke(
            {
                "topic": topic,
                "count": count,
            }
        )

        try:
            result = parser.parse(raw)
        except Exception as e:
            logger.warning(f"Initial parse failed for topic '{topic}', attempting retry correction: {e}")
            fix_parser = RetryWithErrorOutputParser.from_llm(
                llm=llm,
                parser=parser,
                max_retries=3,
            )
            correction_prompt = PromptTemplate.from_template(
                "The previous output had validation errors.\n\n"
                "ERROR: {error}\n\n"
                "ORIGINAL OUTPUT:\n{completion}\n\n"
                "Please fix the errors and return valid JSON:\n{format_instructions}"
            )
            result = fix_parser.parse_with_prompt(
                completion=raw,
                prompt_value=correction_prompt.format_prompt(
                    error=str(e),
                    completion=raw,
                    format_instructions=parser.get_format_instructions(),
                ),
            )

        return [person.name for person in result.people]

    except Exception as e:
        logger.error(f"LLM suggestion failed for topic '{topic}': {e}")
        return []


# Language code to full name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
}


DESCRIPTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a biographical writer. Create a 2-3 sentence description of a person.

Requirements:
- Do NOT start with or repeat the person's name (it is already shown separately)
- Include life years (birth-death or birth- if alive)
- Include nationality
- Mention their most important achievements or contributions
- Remove any phonetic pronunciations (like "KRUUG-mən")
- Write directly in {language}
- Target 30-50 words (not less than 30)
- Use only facts from the provided Wikipedia extract
- Use plain text only - NO markdown formatting (no ** or other markup)

Output ONLY the description, nothing else.""",
        ),
        (
            "human",
            """Person: {name}

Wikipedia extract:
{wiki_extract}

Write a biographical description in {language} (30-50 words). Do NOT include the person's name at the start.""",
        ),
    ]
)


def generate_description(
    name: str,
    wiki_extract: str,
    language: str = "en",
    llm_provider: str = "mistral",
) -> str:
    """
    Generate a concise description from Wikipedia extract.

    Creates a 1-2 sentence summary including life years, nationality,
    and key achievements. Generates directly in the target language.

    Args:
        name: Person's name
        wiki_extract: Wikipedia extract to use as source
        language: Target language code (e.g., "en", "es", "fr")
        llm_provider: LLM provider to use

    Returns:
        Concise description in the target language
    """
    lang_name = LANGUAGE_NAMES.get(language.lower(), language)

    try:
        llm = create_text_llm(provider=llm_provider, temperature=0.2)
        chain = DESCRIPTION_PROMPT | llm

        result = chain.invoke(
            {
                "name": name,
                "wiki_extract": wiki_extract,
                "language": lang_name,
            }
        )
        return result.content.strip()

    except Exception as e:
        logger.error(f"Description generation failed for '{name}': {e}")
        # Fallback: return cleaned extract (remove pronunciations)
        import re

        cleaned = re.sub(r"\s*\([^)]*[ˈˌ][^)]*\)\s*", " ", wiki_extract)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:200] + "..." if len(cleaned) > 200 else cleaned


if __name__ == "__main__":
    import os

    # Ensure API keys are available
    if not os.getenv("MISTRAL_API_KEY"):
        print("⚠️  MISTRAL_API_KEY not set. Please set it to test.")
        exit(1)

    topic = "international trade policy and tariffs"
    print(f"🔍 Suggesting people for topic: '{topic}'\n")
    print("-" * 60)

    names = suggest_people(topic, count=5, llm_provider="mistral")

    print(f"\n📋 Suggested {len(names)} people:")
    for i, name in enumerate(names, 1):
        print(f"   {i}. {name}")

    # Test description generation
    print("\n" + "-" * 60)
    print("\n📝 Testing description generation...")

    test_extract = """Paul Robin Krugman ( KRUUG-mən; born February 28, 1953) is an American economist who is the Distinguished Professor of Economics at the Graduate Center of the City University of New York. He was a columnist for The New York Times from 2000 to 2024. In 2008, Krugman was the sole winner of the Nobel Memorial Prize in Economic Sciences for his contributions to new trade theory and new economic geography."""

    print("\n   Original Wikipedia extract:")
    print(f"   {test_extract[:100]}...")

    print("\n   Generated (English):")
    desc_en = generate_description("Paul Krugman", test_extract, "en", "mistral")
    print(f"   {desc_en}")

    print("\n   Generated (Spanish):")
    desc_es = generate_description("Paul Krugman", test_extract, "es", "mistral")
    print(f"   {desc_es}")
