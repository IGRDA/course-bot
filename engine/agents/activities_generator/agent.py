"""
Activities generator agent using the base SectionProcessor pattern.

This agent generates quiz activities, meta elements (glossary, key concepts),
and application activities for each section.
"""

import random
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from workflows.state import CourseState, Section, GlossaryTerm, Activity, FinalActivity, MetaElements, ActivitiesSection
from workflows.audience_profiles import build_audience_guidelines
from agents.base import SectionProcessor, SectionTask
from LLMs.text2text import create_text_llm, resolve_text_model_name
from .prompts import activities_generation_prompt, meta_only_prompt, correction_prompt


# ---- Activity Selection ----
def select_activity_types(mode: str, section_idx: int) -> tuple[List[str], List[str]]:
    """
    Select activity types based on mode (random or deterministic).
    
    Args:
        mode: "random" or "deterministic"
        section_idx: Section index for deterministic cycling
        
    Returns:
        Tuple of (quiz_types, final_types)
    """
    quiz_types = ["order_list", "fill_gaps", "swipper", "linking_terms"]
    final_types = ["group_activity", "discussion_forum", "individual_project", "open_ended_quiz"]
    
    if mode == "random":
        # Random selection - pick 2 from the pool plus always include multiple_choice, multi_selection
        selected_quiz = random.sample(quiz_types, 2)
        selected_final = [random.choice(final_types)]
    else:  # deterministic - cycle through types, pick 2 based on section index
        selected_quiz = []
        for i in range(2):
            type_idx = (section_idx + i) % len(quiz_types)
            selected_quiz.append(quiz_types[type_idx])
        # Cycle through final types
        final_idx = section_idx % len(final_types)
        selected_final = [final_types[final_idx]]
    
    # Always add multiple_choice and multi_selection
    selected_quiz.extend(["multiple_choice", "multi_selection"])
    
    return selected_quiz, selected_final


# ---- Output Models ----
class ActivitiesOutput(BaseModel):
    """Complete activities output for a section (with activities)"""
    glossary: List[GlossaryTerm] = Field(..., min_length=1, max_length=4, description="1-4 glossary terms")
    key_concept: str = Field(..., description="One sentence key concept summary")
    interesting_fact: str = Field(default="", description="Interesting fact related to the section")
    quote: Optional[dict] = Field(default=None, description="Quote with author and text")
    activities: List[Activity] = Field(..., min_length=1, description="List of activities")
    final_activities: List[FinalActivity] = Field(..., min_length=1, description="List of final activities")

    class Config:
        populate_by_name = True


class MetaOnlyOutput(BaseModel):
    """Meta elements only output for sections without activities"""
    glossary: List[GlossaryTerm] = Field(..., min_length=1, max_length=4, description="1-4 glossary terms")
    key_concept: str = Field(..., description="One sentence key concept summary")
    interesting_fact: str = Field(default="", description="Interesting fact related to the section")
    quote: Optional[dict] = Field(default=None, description="Quote with author and text")

    class Config:
        populate_by_name = True


# ---- Activities Processor using SectionProcessor pattern ----
class ActivitiesProcessor(SectionProcessor):
    """Processor for generating activities for sections."""
    
    def __init__(self):
        super().__init__(name="activities_generator")
        self._global_section_idx = 0  # Track global section index for activity type cycling
    
    def create_task_data(
        self,
        course_state: CourseState,
        module_idx: int,
        submodule_idx: int,
        section_idx: int,
        section: Section,
    ) -> dict[str, Any]:
        """Create task-specific data for activities generation."""
        # Get activity selection settings from config
        mode = course_state.config.activity_selection_mode
        freq = course_state.config.sections_per_activity
        
        # Determine if this section should have activities
        # Based on section index within the submodule (0, freq, 2*freq, etc.)
        should_generate_activities = (section_idx % freq == 0)
        
        # Select activity types using global section index for proper cycling
        # Only needed if generating activities
        if should_generate_activities:
            quiz_types, final_types = select_activity_types(mode, self._global_section_idx)
            self._global_section_idx += 1
        else:
            quiz_types, final_types = [], []
        
        return {
            "section_title": section.title,
            "theory": section.theory,
            "should_generate_activities": should_generate_activities,
            "activity_types": quiz_types,
            "final_activity_types": final_types,
        }
    
    def process_section(self, task: SectionTask) -> dict[str, Any]:
        """Generate activities (or meta-only) for a single section."""
        # Extract context
        provider = task.config.text_llm_provider
        max_retries = task.config.max_retries
        
        # Get task-specific data
        section_title = task.extra_data["section_title"]
        theory = task.extra_data["theory"]
        max_chars = getattr(task.config, "max_theory_chars_for_llm", 10_000_000)
        if len(theory) > max_chars:
            theory = theory[:max_chars] + "\n\n[... text truncated for length ...]"
        should_generate_activities = task.extra_data["should_generate_activities"]
        activity_types = task.extra_data["activity_types"]
        final_activity_types = task.extra_data["final_activity_types"]
        
        # Create LLM
        model_name = resolve_text_model_name(provider)
        llm_kwargs = {"temperature": 0.5}  # Slightly higher temp for creative activities
        if model_name:
            llm_kwargs["model_name"] = model_name
        llm = create_text_llm(provider=provider, **llm_kwargs)
        
        # Build audience guidelines block
        audience_guidelines = build_audience_guidelines(
            task.config.target_audience,
            context="activities"
        )
        
        if should_generate_activities:
            # Full activities generation
            parser = PydanticOutputParser(pydantic_object=ActivitiesOutput)
            fix_parser = RetryWithErrorOutputParser.from_llm(
                llm=llm,
                parser=parser,
                max_retries=max_retries,
            )
            
            chain = activities_generation_prompt | llm | StrOutputParser()
            activity_desc = ", ".join(activity_types)
            final_desc = ", ".join(final_activity_types)
            
            raw = chain.invoke({
                "theory": theory,
                "section_title": section_title,
                "language": task.language,
                "activity_types": activity_desc,
                "final_activity_types": final_desc,
                "format_instructions": parser.get_format_instructions(),
                "audience_guidelines": audience_guidelines,
            })
            
            try:
                result = parser.parse(raw)
            except Exception as e:
                print(f"  ⚠ Initial parse failed for section {task.section_idx}, attempting correction...")
                result = fix_parser.parse_with_prompt(
                    completion=raw,
                    prompt_value=correction_prompt.format_prompt(
                        error=str(e),
                        completion=raw,
                        format_instructions=parser.get_format_instructions(),
                    ),
                )
            
            return {
                "should_generate_activities": True,
                "glossary": result.glossary,
                "key_concept": result.key_concept,
                "interesting_fact": result.interesting_fact,
                "quote": result.quote,
                "activities": result.activities,
                "final_activities": result.final_activities,
            }
        else:
            # Meta-only generation (no activities)
            parser = PydanticOutputParser(pydantic_object=MetaOnlyOutput)
            fix_parser = RetryWithErrorOutputParser.from_llm(
                llm=llm,
                parser=parser,
                max_retries=max_retries,
            )
            
            chain = meta_only_prompt | llm | StrOutputParser()
            
            raw = chain.invoke({
                "theory": theory,
                "section_title": section_title,
                "language": task.language,
                "format_instructions": parser.get_format_instructions(),
                "audience_guidelines": audience_guidelines,
            })
            
            try:
                result = parser.parse(raw)
            except Exception as e:
                print(f"  ⚠ Initial parse failed for section {task.section_idx} (meta-only), attempting correction...")
                result = fix_parser.parse_with_prompt(
                    completion=raw,
                    prompt_value=correction_prompt.format_prompt(
                        error=str(e),
                        completion=raw,
                        format_instructions=parser.get_format_instructions(),
                    ),
                )
            
            return {
                "should_generate_activities": False,
                "glossary": result.glossary,
                "key_concept": result.key_concept,
                "interesting_fact": result.interesting_fact,
                "quote": result.quote,
            }
    
    def reduce_result(self, section: Section, result: dict[str, Any]) -> None:
        """Apply activities result to section."""
        # Always create MetaElements with glossary and metadata
        section.meta_elements = MetaElements(
            glossary=result["glossary"],
            key_concept=result["key_concept"],
            interesting_fact=result.get("interesting_fact", ""),
            quote=result.get("quote")
        )
        
        # Only create ActivitiesSection if activities were generated
        if result.get("should_generate_activities", False):
            section.activities = ActivitiesSection(
                quiz=result["activities"],
                application=result["final_activities"]
            )
        # else: section.activities remains None


def generate_all_section_activities(
    course_state: CourseState,
    concurrency: int = 8,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> CourseState:
    """
    Main function to generate all section activities using LangGraph Send pattern.
    
    Args:
        course_state: CourseState with theories filled
        concurrency: Number of concurrent requests
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
        
    Returns:
        Updated CourseState with all activities filled
    """
    print(f"🚀 Starting parallel activities generation with max_retries={max_retries}, "
          f"initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
    
    processor = ActivitiesProcessor()
    return processor.process_all(
        course_state=course_state,
        concurrency=concurrency,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
    )
