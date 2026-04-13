"""
HTML formatter agent using the base SectionProcessor pattern.

This agent transforms section theories into structured HTML elements
(paragraphs, accordions, tabs, carousels, etc.).
"""

import re
from typing import Any

from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain_core.output_parsers import StrOutputParser
from LLMs.text2text import create_text_llm, resolve_text_model_name
from pydantic import create_model

from agents.base import SectionProcessor, SectionTask
from workflows.state import CourseState, HtmlElement, Section

from .prompts import correction_prompt, html_generation_prompt


def strip_latex_from_raw_json(raw: str) -> str:
    """
    Strip/fix LaTeX patterns from raw JSON string as last resort fallback.
    Operates on the raw string BEFORE JSON parsing when LLM retries are exhausted.

    Handles patterns like \\frac, \\pi, \\epsilon, \\(, \\), etc. that cause
    JSON parsing errors due to invalid escape sequences.

    Strategy: Process only content INSIDE JSON string values to avoid breaking
    JSON structure. Escapes invalid backslashes rather than removing content.
    """
    # Find JSON content (between first { and last })
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return raw

    prefix = raw[:start]
    json_content = raw[start : end + 1]
    suffix = raw[end + 1 :]

    def fix_string_content(match):
        """Fix backslashes inside a JSON string value."""
        s = match.group(1)
        # Escape backslashes that aren't followed by valid JSON escape sequences
        # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        result = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)
        return f'"{result}"'

    # Process only strings within quotes (to avoid breaking JSON structure)
    # This regex matches JSON string values: "..."
    sanitized = re.sub(r'"((?:[^"\\]|\\.)*)"', fix_string_content, json_content)

    return prefix + sanitized + suffix


# ---- Icon Mapping ----
ICON_MAPPING = {
    "history": "mdi-book-clock",
    "historia": "mdi-book-clock",
    "science": "mdi-flask",
    "ciencia": "mdi-flask",
    "technology": "mdi-laptop",
    "tecnología": "mdi-laptop",
    "math": "mdi-calculator",
    "matemática": "mdi-calculator",
    "art": "mdi-palette",
    "arte": "mdi-palette",
    "business": "mdi-briefcase",
    "negocio": "mdi-briefcase",
    "health": "mdi-heart-pulse",
    "salud": "mdi-heart-pulse",
    "education": "mdi-school",
    "educación": "mdi-school",
    "politics": "mdi-account-group",
    "política": "mdi-account-group",
    "military": "mdi-shield-sword",
    "militar": "mdi-shield-sword",
    "war": "mdi-sword-cross",
    "guerra": "mdi-sword-cross",
    "law": "mdi-gavel",
    "derecho": "mdi-gavel",
    "philosophy": "mdi-head-lightbulb",
    "filosofía": "mdi-head-lightbulb",
    "default": "mdi-information",
}


def select_icon(section_title: str) -> str:
    """
    Select an appropriate Material Design Icon based on section title keywords.

    Args:
        section_title: Title of the section

    Returns:
        Material Design Icon class name
    """
    title_lower = section_title.lower()

    # Check for keyword matches
    for keyword, icon in ICON_MAPPING.items():
        if keyword in title_lower:
            return icon

    # Default icon if no match
    return ICON_MAPPING["default"]


class HtmlProcessor(SectionProcessor):
    """Processor for generating HTML structure for sections."""

    def __init__(self):
        super().__init__(name="html_formatter")

    def create_task_data(
        self,
        course_state: CourseState,
        module_idx: int,
        submodule_idx: int,
        section_idx: int,
        section: Section,
    ) -> dict[str, Any]:
        """Create task data with HTML generation parameters."""
        return {
            "theory": section.theory,
            "section_title": section.title,
            "include_quotes": course_state.config.include_quotes_in_html,
            "include_tables": course_state.config.include_tables_in_html,
            "suggested_icon": select_icon(section.title),
        }

    def process_section(self, task: SectionTask) -> dict[str, Any]:
        """Generate HTML structure for a single section."""
        # Extract task data
        theory = task.extra_data["theory"]
        max_chars = getattr(task.course_state.config, "max_theory_chars_for_llm", 1_000_000)
        if len(theory) > max_chars:
            theory = theory[:max_chars] + "\n\n[... text truncated for length ...]"
        section_title = task.extra_data["section_title"]
        include_quotes = task.extra_data["include_quotes"]
        include_tables = task.extra_data["include_tables"]
        suggested_icon = task.extra_data["suggested_icon"]

        # Extract config
        config = task.course_state.config
        provider = config.text_llm_provider
        max_retries = config.max_retries
        language = task.course_state.config.language

        # Parse allowed formats
        allowed_formats = config.html_formats.split("|")
        allowed_formats_str = ", ".join(f'"{fmt}"' for fmt in allowed_formats)

        # Create LLM
        model_name = resolve_text_model_name(provider)
        llm_kwargs = {"temperature": 0.0}
        if model_name:
            llm_kwargs["model_name"] = model_name
        llm = create_text_llm(provider=provider, **llm_kwargs)

        # Create parser for List[HtmlElement]
        HtmlElementList = create_model("HtmlElementList", elements=(list[HtmlElement], ...))
        parser = PydanticOutputParser(pydantic_object=HtmlElementList)

        # Create fix parser for retry
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=llm,
            parser=parser,
            max_retries=max_retries,
        )

        # Generate using LCEL chain
        chain = html_generation_prompt | llm | StrOutputParser()

        # Build optional instructions
        quote_instruction = ""
        if include_quotes:
            quote_instruction = """- Optionally include quote elements for FAMOUS QUOTES ONLY
  - Use ONLY for: famous sayings, philosophical statements, well-known phrases, memorable historical quotes
  Format: {{"type": "quote", "content": {{"author": "...", "text": "..."}}}}"""

        table_instruction = ""
        if include_tables:
            table_instruction = """- Optionally include table elements with title, headers, and rows
  Format: {{"type": "table", "content": {{"title": "...", "headers": [...], "rows": [[...], [...]]}}}}"""

        raw = chain.invoke(
            {
                "theory": theory,
                "section_title": section_title,
                "language": language,
                "quote_instruction": quote_instruction,
                "table_instruction": table_instruction,
                "suggested_icon": suggested_icon,
                "allowed_formats": allowed_formats_str,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Try to parse, with fallback to retry parser, then LaTeX stripping as last resort
        try:
            result = parser.parse(raw)
            html_array = result.elements
        except Exception as e:
            print(f"  ⚠ Initial parse failed for section {task.section_idx}, attempting LLM correction...")
            try:
                # Let LLM retry with configured max_retries
                result = fix_parser.parse_with_prompt(
                    completion=raw,
                    prompt_value=correction_prompt.format_prompt(
                        error=str(e),
                        completion=raw,
                        format_instructions=parser.get_format_instructions(),
                    ),
                )
                html_array = result.elements
            except Exception:
                # All LLM retries exhausted - last resort: strip LaTeX and try once more
                print(f"  ⚠ LLM retries exhausted for section {task.section_idx}, stripping LaTeX as fallback...")
                stripped_raw = strip_latex_from_raw_json(raw)
                try:
                    result = parser.parse(stripped_raw)
                    html_array = result.elements
                    print(f"  ✓ Recovered section {task.section_idx} by stripping LaTeX")
                except Exception as final_error:
                    print(f"  ✗ All recovery attempts failed for section {task.section_idx}: {final_error}")
                    raise final_error

        # Apply random format override if configured
        if config.select_html == "random":
            import random

            # Deterministic seed based on global seed + section position
            seed = config.html_random_seed + task.module_idx * 1000 + task.submodule_idx * 100 + task.section_idx
            random.seed(seed)

            # Randomly select format from allowed formats
            selected_format = random.choice(allowed_formats)

            # Override the format type in the HTML structure
            for element in html_array:
                # Find the interactive format element (not p, ul, quote, table)
                if element.type in ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]:
                    element.type = selected_format
                    print(f"  🎲 Random override: {element.type} → {selected_format}")
                    break

        return {"html_structure": html_array}

    def reduce_result(self, section: Section, result: dict[str, Any]) -> None:
        """Apply HTML structure to section."""
        section.html = result["html_structure"]


# Module-level processor instance
_processor = HtmlProcessor()


def generate_all_section_html(
    course_state: CourseState,
    concurrency: int = 8,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> CourseState:
    """
    Main function to generate all section HTML structures using LangGraph Send pattern.

    Args:
        course_state: CourseState with theories filled
        concurrency: Number of concurrent requests
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff

    Returns:
        Updated CourseState with all HTML structures filled
    """
    return _processor.process_all(
        course_state,
        concurrency=concurrency,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
    )


# Keep the old function names for backward compatibility
def build_html_generation_graph(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """Build the HTML generation subgraph (backward compatibility wrapper)."""
    return _processor.build_graph(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
    )
