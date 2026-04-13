"""
Section theory generator agent using the base SectionProcessor pattern.

This agent generates educational theory content for each section,
with optional reflection-based fact verification.
"""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from LLMs.text2text import create_text_llm, resolve_text_model_name
from pydantic import BaseModel, Field

from agents.base import SectionProcessor, SectionTask
from tools.websearch import create_web_search
from workflows.audience_profiles import build_audience_guidelines
from workflows.state import CourseState, Section

from .prompts import (
    STYLE_CONTINUATION,
    STYLE_COURSE_INTRO,
    STYLE_DEEP_DIVE,
    STYLE_MODULE_START,
    STYLE_SUBMODULE_START,
    query_generation_prompt,
    reflection_prompt,
    regeneration_prompt,
    section_theory_prompt,
)


# ---- Pydantic models for structured outputs ----
class QueryList(BaseModel):
    """List of search queries for fact verification"""

    queries: list[str] = Field(
        description="List of search queries to verify facts, formulas, laws, dates, and concepts"
    )


class Reflection(BaseModel):
    """Reflection and critique of section content"""

    critique: str = Field(
        description="Detailed critique identifying factual errors, outdated info, or improvements needed"
    )
    needs_revision: bool = Field(description="Whether the content needs to be regenerated based on findings")


# ---- Helper functions ----
def select_style_guidelines(
    module_idx: int, submodule_idx: int, section_idx: int, total_sections_in_submodule: int
) -> str:
    """
    Select appropriate style guidelines based on section position in the course structure.

    Args:
        module_idx: Index of the current module
        submodule_idx: Index of the current submodule
        section_idx: Index of the current section
        total_sections_in_submodule: Total number of sections in the current submodule

    Returns:
        Style guidelines string to inject into the prompt
    """
    # Course introduction - very first section
    if module_idx == 0 and submodule_idx == 0 and section_idx == 0:
        return STYLE_COURSE_INTRO

    # Module start - first section of a new module (but not the first module)
    if submodule_idx == 0 and section_idx == 0:
        return STYLE_MODULE_START

    # Submodule start - first section of a new submodule (but not first of module)
    if section_idx == 0:
        return STYLE_SUBMODULE_START

    # Deep dive - later sections in a submodule (last half)
    if section_idx >= total_sections_in_submodule // 2 and total_sections_in_submodule > 2:
        return STYLE_DEEP_DIVE

    # Default continuation - middle sections
    return STYLE_CONTINUATION


def build_course_context(
    course_state: CourseState,
    current_module_idx: int,
    current_submodule_idx: int,
    max_outline_chars: int = 600,
    max_sections_chars: int = 800,
) -> tuple[str, str]:
    """
    Build course-wide context for anti-repetition.

    Returns:
        (course_outline, same_module_sections) tuple:
        - course_outline: Condensed view of all modules and their submodules
        - same_module_sections: Section titles from OTHER submodules in the same module
    """
    # Build course outline: "Module 1: Submod A, Submod B | Module 2: ..."
    outline_parts = []
    for m_idx, module in enumerate(course_state.modules):
        if m_idx == current_module_idx:
            # Mark current module
            submod_names = ", ".join(sm.title for sm in module.submodules)
            outline_parts.append(f"[CURRENT] {module.title}: {submod_names}")
        else:
            submod_names = ", ".join(sm.title for sm in module.submodules)
            outline_parts.append(f"{module.title}: {submod_names}")

    course_outline = " | ".join(outline_parts)

    # Truncate if too long
    if len(course_outline) > max_outline_chars:
        truncated = course_outline[:max_outline_chars]
        # Find last complete module separator
        last_sep = truncated.rfind(" | ")
        if last_sep > 0:
            truncated = truncated[:last_sep]
        remaining_modules = len(course_state.modules) - truncated.count(":")
        course_outline = f"{truncated} | ... and {remaining_modules} more modules"

    # Build same-module sections (from OTHER submodules in current module)
    current_module = course_state.modules[current_module_idx]
    section_parts = []

    for sm_idx, submodule in enumerate(current_module.submodules):
        if sm_idx == current_submodule_idx:
            continue  # Skip current submodule (handled by sibling_summaries)

        section_titles = [s.title for s in submodule.sections]
        section_parts.append(f"{submodule.title}: {', '.join(section_titles)}")

    same_module_sections = "\n".join(section_parts) if section_parts else "None (only one submodule in this module)"

    # Truncate if too long
    if len(same_module_sections) > max_sections_chars:
        truncated = same_module_sections[:max_sections_chars]
        # Find last complete line
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        # Count remaining sections
        total_sections = sum(
            len(sm.sections)
            for sm in current_module.submodules
            if sm != current_module.submodules[current_submodule_idx]
        )
        shown_sections = truncated.count(",") + truncated.count("\n") + 1
        remaining = max(0, total_sections - shown_sections)
        same_module_sections = f"{truncated}\n... and {remaining} more sections in this module"

    return course_outline, same_module_sections


def reflect_and_improve(
    theory: str,
    section_title: str,
    module_title: str,
    submodule_title: str,
    sibling_summaries: str,
    language: str,
    n_words: int,
    num_queries: int,
    provider: str,
    web_search_provider: str,
    target_audience: str | None = None,
) -> str:
    """
    Apply reflection pattern: generate queries → parallel search → reflect → regenerate if needed

    Args:
        theory: Current theory content to improve
        section_title: Title of this section
        module_title: Title of the parent module
        submodule_title: Title of the parent submodule
        sibling_summaries: Formatted string of sibling sections with their summaries
        language: Target language for content
        n_words: Target word count
        num_queries: Number of verification queries to generate
        provider: LLM provider name
        web_search_provider: Web search provider name
        target_audience: Target audience for content adaptation
    """
    try:
        # Create LLM with specified provider
        model_name = resolve_text_model_name(provider)
        llm_kwargs = {"temperature": 0}
        if model_name:
            llm_kwargs["model_name"] = model_name
        llm = create_text_llm(provider=provider, **llm_kwargs)

        # Create web search function with specified provider
        web_search = create_web_search(web_search_provider)

        # Step 1: Generate verification queries
        query_chain = query_generation_prompt | llm.with_structured_output(QueryList)
        query_list = query_chain.invoke(
            {
                "theory": theory,
                "section_title": section_title,
                "module_title": module_title,
                "submodule_title": submodule_title,
                "k": num_queries,
            }
        )

        # Step 2: Execute all web searches in parallel
        def safe_search(query: str, idx: int) -> str:
            try:
                result = web_search(query, max_results=num_queries)
                return f"Query {idx}: {query}\nResult: {result}\n"
            except Exception as e:
                return f"Query {idx}: {query}\nResult: Search failed - {e!s}\n"

        parallel_searches = {
            f"search_{i}": RunnableLambda(lambda x, q=q, idx=i + 1: safe_search(q, idx))
            for i, q in enumerate(query_list.queries)
        }

        search_results = RunnableParallel(**parallel_searches).invoke({})
        all_search_results = "\n".join(search_results.values())

        # Step 3: Reflect on content quality
        reflection_chain = reflection_prompt | llm.with_structured_output(Reflection)
        reflection_result = reflection_chain.invoke(
            {
                "theory": theory,
                "section_title": section_title,
                "module_title": module_title,
                "submodule_title": submodule_title,
                "search_results": all_search_results,
            }
        )

        # Step 4: Regenerate if needed
        if reflection_result.needs_revision:
            audience_guidelines = build_audience_guidelines(target_audience, context="theory")
            regeneration_chain = regeneration_prompt | llm | StrOutputParser()
            improved_theory = regeneration_chain.invoke(
                {
                    "theory": theory,
                    "section_title": section_title,
                    "module_title": module_title,
                    "submodule_title": submodule_title,
                    "sibling_sections": sibling_summaries,
                    "reflection": reflection_result.critique,
                    "search_results": all_search_results,
                    "language": language,
                    "n_words": n_words,
                    "audience_guidelines": audience_guidelines,
                }
            ).strip()

            print("  ↻ Reflection suggested improvements, regenerated content")
            return improved_theory
        else:
            print("  ✓ Reflection: content is accurate")
            return theory

    except Exception as e:
        print(f"  ⚠ Reflection failed: {e!s}, using original content")
        return theory


# ---- Theory Processor using SectionProcessor pattern ----
class TheoryProcessor(SectionProcessor):
    """Processor for generating theory content for sections."""

    def __init__(self):
        super().__init__(name="theory_generator")

    def create_task_data(
        self,
        course_state: CourseState,
        module_idx: int,
        submodule_idx: int,
        section_idx: int,
        section: Section,
    ) -> dict[str, Any]:
        """Create task-specific data for theory generation."""
        return {
            "section_title": section.title,
            "use_reflection": course_state.config.use_reflection,
            "num_queries": course_state.config.num_reflection_queries,
        }

    def process_section(self, task: SectionTask) -> dict[str, Any]:
        """Generate theory for a single section."""
        # Extract context from course state
        module = task.course_state.modules[task.module_idx]
        submodule = module.submodules[task.submodule_idx]
        provider = task.config.text_llm_provider

        # Get task-specific data
        section_title = task.extra_data["section_title"]
        use_reflection = task.extra_data["use_reflection"]
        num_queries = task.extra_data["num_queries"]

        # Calculate target word count
        total_sections = sum(
            len(submodule.sections) for module in task.course_state.modules for submodule in module.submodules
        )
        n_words = task.config.total_pages * task.config.words_per_page // total_sections

        # Create LLM with specified provider
        model_name = resolve_text_model_name(provider)
        llm_kwargs = {"temperature": 0}
        if model_name:
            llm_kwargs["model_name"] = model_name
        llm = create_text_llm(provider=provider, **llm_kwargs)

        # Determine style guidelines based on position
        style_guidelines = select_style_guidelines(
            module_idx=task.module_idx,
            submodule_idx=task.submodule_idx,
            section_idx=task.section_idx,
            total_sections_in_submodule=len(submodule.sections),
        )

        # Build sibling section context with summaries to avoid content repetition
        sibling_parts = []
        for s in submodule.sections:
            if s.title != section_title:
                if s.summary:
                    sibling_parts.append(f"- {s.title}:\n  {s.summary}")
                else:
                    sibling_parts.append(f"- {s.title}: (no summary available)")
        sibling_summaries = (
            "\n".join(sibling_parts) if sibling_parts else "None (this is the only section in this submodule)"
        )

        # Build course-wide context to avoid repetition across modules/submodules
        course_outline, same_module_sections = build_course_context(
            task.course_state, task.module_idx, task.submodule_idx
        )

        # Build audience guidelines block
        audience_guidelines = build_audience_guidelines(task.config.target_audience, context="theory")

        # Generate initial content using LCEL chain (retry handled by LangGraph)
        section_chain = section_theory_prompt | llm | StrOutputParser()
        theory = section_chain.invoke(
            {
                "course_title": task.course_state.title,
                "module_title": module.title,
                "submodule_title": submodule.title,
                "section_title": section_title,
                "language": task.language,
                "n_words": n_words,
                "style_guidelines": style_guidelines,
                "sibling_summaries": sibling_summaries,
                "course_outline": course_outline,
                "same_module_sections": same_module_sections,
                "audience_guidelines": audience_guidelines,
            }
        ).strip()

        # Apply reflection pattern if enabled
        if use_reflection:
            theory = reflect_and_improve(
                theory=theory,
                section_title=section_title,
                module_title=module.title,
                submodule_title=submodule.title,
                sibling_summaries=sibling_summaries,
                language=task.language,
                n_words=n_words,
                num_queries=num_queries,
                provider=provider,
                web_search_provider=task.config.web_search_provider,
                target_audience=task.config.target_audience,
            )

        return {"theory": theory}

    def reduce_result(self, section: Section, result: dict[str, Any]) -> None:
        """Apply theory result to section."""
        section.theory = result["theory"]


def generate_all_section_theories(
    course_state: CourseState,
    concurrency: int = 8,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    use_reflection: bool = False,
    num_reflection_queries: int = 3,
) -> CourseState:
    """
    Main function to generate all section theories using LangGraph Send pattern.

    Args:
        course_state: CourseState with skeleton structure (empty theories)
        concurrency: Maximum number of concurrent section generations
        max_retries: Maximum number of retry attempts for each LLM call
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        use_reflection: Whether to use reflection pattern for fact verification (default: False)
        num_reflection_queries: Number of verification queries to generate (default: 3)

    Returns:
        Updated CourseState with all theories filled
    """
    # Note: use_reflection and num_reflection_queries are read from course_state.config
    # These parameters are kept for backward compatibility but config values take precedence
    reflection_enabled = course_state.config.use_reflection
    reflection_queries = course_state.config.num_reflection_queries

    reflection_msg = f", reflection={'ON' if reflection_enabled else 'OFF'}"
    if reflection_enabled:
        reflection_msg += f" (queries={reflection_queries})"

    print(
        f"🚀 Starting parallel generation with max_retries={max_retries}, "
        f"initial_delay={initial_delay}s, backoff_factor={backoff_factor}{reflection_msg}"
    )

    processor = TheoryProcessor()
    return processor.process_all(
        course_state=course_state,
        concurrency=concurrency,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
    )
