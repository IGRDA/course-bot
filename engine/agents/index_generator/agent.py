from typing import List, Optional, Union, Annotated
import json
import logging
import concurrent.futures
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy

# Configure logger for this module
logger = logging.getLogger(__name__)
from workflows.state import CourseState, CourseConfig, Module, Submodule, Section, CourseResearch
from workflows.audience_profiles import build_audience_guidelines, AudienceType


# Helper for state accumulation
def add(a: list, b: list) -> list:
    """Accumulator for list fields in LangGraph state."""
    return a + b
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.websearch import create_web_search
from .prompts import (
    gen_prompt, 
    retry_prompt,
    modules_only_prompt,
    modules_only_with_research_prompt,
    add_submodules_prompt,
    add_sections_prompt,
    expand_descriptions_prompt,
    query_generation_prompt, 
    research_synthesis_prompt,
    gen_with_research_prompt,
    summary_generation_prompt,
    correction_prompt
)
from .utils import compute_layout


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence (```json or ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove closing fence
        return "\n".join(lines)
    return text


# -------------------------------------------------------
# Simplified Skeleton Models (for LLM output parsing)
# -------------------------------------------------------
class SkeletonSection(BaseModel):
    """Lightweight section model with only structural fields"""
    title: str = Field(..., description="Title of the section")
    index: int = Field(default=0, description="Section index within submodule")
    description: str = Field(default="", description="Description of the section")

class SkeletonSubmodule(BaseModel):
    """Lightweight submodule model with only structural fields"""
    title: str = Field(..., description="Title of the submodule")
    index: int = Field(default=0, description="Submodule index within module")
    description: str = Field(default="", description="Description of the submodule")
    sections: List[SkeletonSection] = Field(..., description="List of sections in this submodule")

class SkeletonModule(BaseModel):
    """Lightweight module model with only structural fields"""
    title: str = Field(..., description="Title of the module")
    index: int = Field(default=0, description="Module index in course")
    description: str = Field(default="", description="Description of the module")
    submodules: List[SkeletonSubmodule] = Field(..., description="List of submodules in this module")

class SkeletonCourse(BaseModel):
    """Lightweight course model for skeleton generation"""
    title: str = Field(..., description="Title of the course")
    modules: List[SkeletonModule] = Field(..., description="Course structure with all modules")

# -------------------------------------------------------
# Hierarchical Models (for 3-step generation)
# -------------------------------------------------------

# Step 1: Modules only (no submodules)
class ModuleOnly(BaseModel):
    """Module with title only, no submodules"""
    title: str = Field(..., description="Title of the module")

class ModulesOnlyCourse(BaseModel):
    """Course with module titles only"""
    title: str = Field(..., description="Title of the course")
    modules: List[ModuleOnly] = Field(..., description="List of modules (titles only)")

# Step 2: Modules with submodules (no sections)
class SubmoduleOnly(BaseModel):
    """Submodule with title only, no sections"""
    title: str = Field(..., description="Title of the submodule")

class ModuleWithSubmodules(BaseModel):
    """Module with submodules, no sections"""
    title: str = Field(..., description="Title of the module")
    submodules: List[SubmoduleOnly] = Field(..., description="List of submodules")

class ModulesWithSubmodulesCourse(BaseModel):
    """Course with modules and submodules, no sections"""
    title: str = Field(..., description="Title of the course")
    modules: List[ModuleWithSubmodules] = Field(..., description="Modules with submodules")

# Step 3: Complete structure with sections (titles only)
class TitleOnlySection(BaseModel):
    """Section with title only"""
    title: str = Field(..., description="Title of the section")

class TitleOnlySubmodule(BaseModel):
    """Submodule with sections"""
    title: str = Field(..., description="Title of the submodule")
    sections: List[TitleOnlySection] = Field(..., description="List of sections")

class TitleOnlyModule(BaseModel):
    """Module with submodules and sections"""
    title: str = Field(..., description="Title of the module")
    submodules: List[TitleOnlySubmodule] = Field(..., description="List of submodules")

class TitleOnlyCourse(BaseModel):
    """Complete course structure with titles only"""
    title: str = Field(..., description="Title of the course")
    modules: List[TitleOnlyModule] = Field(..., description="Complete structure")

# Parser for skeleton course (with descriptions)
skeleton_parser = PydanticOutputParser(pydantic_object=SkeletonCourse)

# Parsers for hierarchical generation
modules_only_parser = PydanticOutputParser(pydantic_object=ModulesOnlyCourse)
modules_with_submodules_parser = PydanticOutputParser(pydantic_object=ModulesWithSubmodulesCourse)
titles_only_parser = PydanticOutputParser(pydantic_object=TitleOnlyCourse)

# Parser for research synthesis
research_parser = PydanticOutputParser(pydantic_object=CourseResearch)


# -------------------------------------------------------
# LangGraph State Models for Skeleton Generation Subgraph
# -------------------------------------------------------
class SkeletonGenerationState(BaseModel):
    """State for skeleton generation subgraph (4 sequential steps)."""
    # Inputs (set once at start)
    title: str
    language: str
    research: Optional[CourseResearch] = None
    n_modules: int
    n_submodules: int
    n_sections: int
    provider: str
    target_audience: AudienceType = None  # Target audience for content adaptation
    
    # Progressive outputs (each step populates the next)
    modules_course: Optional[ModulesOnlyCourse] = None
    course_with_submodules: Optional[ModulesWithSubmodulesCourse] = None
    titles_course: Optional[TitleOnlyCourse] = None
    skeleton_course: Optional[SkeletonCourse] = None


# -------------------------------------------------------
# LangGraph State Models for Summary Generation Subgraph
# -------------------------------------------------------
class ModuleSummaryTask(BaseModel):
    """Input for processing one module's summaries via Send."""
    course_title: str
    module_idx: int
    module_title: str
    module_description: str
    sections_list: str  # Pre-formatted section list for prompt
    language: str
    provider: str
    target_audience: AudienceType = None


class SummaryGenerationState(BaseModel):
    """State for summary generation subgraph (Send pattern)."""
    course_state: CourseState
    language: str
    provider: str
    # Accumulated results from parallel module processing
    completed_summaries: Annotated[list[dict], add] = Field(default_factory=list)
    total_modules: int = 0


def validate_structure_counts(
    course: Union[SkeletonCourse, TitleOnlyCourse],
    expected_modules: int,
    expected_submodules: int,
    expected_sections: int,
) -> tuple[bool, str]:
    """
    Validate that the course structure has exactly the expected counts.
    Works with both SkeletonCourse and TitleOnlyCourse.
    
    Returns:
        (is_valid, error_message) - True if valid, False with description if not
    """
    actual_modules = len(course.modules)
    if actual_modules != expected_modules:
        return False, f"Expected {expected_modules} modules, got {actual_modules}"
    
    for i, module in enumerate(course.modules):
        actual_submodules = len(module.submodules)
        if actual_submodules != expected_submodules:
            return False, f"Module {i+1} has {actual_submodules} submodules, expected {expected_submodules}"
        
        for j, submodule in enumerate(module.submodules):
            actual_sections = len(submodule.sections)
            if actual_sections != expected_sections:
                return False, f"Module {i+1}, Submodule {j+1} has {actual_sections} sections, expected {expected_sections}"
    
    return True, ""


# -------------------------------------------------------
# Three-Step Hierarchical Generation Functions
# -------------------------------------------------------

def generate_modules_step(
    title: str,
    n_modules: int,
    language: str,
    provider: str,
    max_retries: int,
    research: Optional[CourseResearch] = None,
) -> ModulesOnlyCourse:
    """
    Step 1: Generate module titles only (smallest output).
    Validates count and retries if wrong.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Choose prompt based on research availability
    if research is not None:
        chain = modules_only_with_research_prompt | llm | StrOutputParser()
        invoke_params = {
            "course_title": title,
            "language": language,
            "key_topics": "\n".join(f"- {topic}" for topic in research.key_topics),
            "learning_objectives": "\n".join(f"- {obj}" for obj in research.learning_objectives),
            "n_modules": n_modules,
            "format_instructions": modules_only_parser.get_format_instructions(),
        }
    else:
        chain = modules_only_prompt | llm | StrOutputParser()
        invoke_params = {
            "course_title": title,
            "language": language,
            "n_modules": n_modules,
            "format_instructions": modules_only_parser.get_format_instructions(),
        }
    
    # Retry loop with validation
    for attempt in range(max_retries):
        print(f"   Step 1 - Attempt {attempt + 1}/{max_retries}: Generating module titles...")
        
        raw = chain.invoke(invoke_params)
        
        try:
            modules_course = modules_only_parser.parse(raw)
        except Exception as e:
            logger.warning(f"Step 1 Attempt {attempt + 1}: Failed to parse: {e}")
            continue
        
        # Validate module count
        actual_modules = len(modules_course.modules)
        if actual_modules == n_modules:
            print(f"   ✓ Step 1 complete: {n_modules} modules generated")
            return modules_course
        else:
            logger.warning(f"Step 1 Attempt {attempt + 1}: Expected {n_modules} modules, got {actual_modules}")
            print(f"   ⚠ Expected {n_modules} modules, got {actual_modules}. Retrying...")
    
    raise ValueError(f"Step 1 failed after {max_retries} attempts. Expected {n_modules} modules.")


def generate_submodules_step(
    modules_course: ModulesOnlyCourse,
    n_submodules: int,
    language: str,
    provider: str,
    max_retries: int,
) -> ModulesWithSubmodulesCourse:
    """
    Step 2: Add submodules to each module.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    modules_json = modules_course.model_dump_json(indent=2)
    chain = add_submodules_prompt | llm | StrOutputParser()
    
    for attempt in range(max_retries):
        print(f"   Step 2 - Attempt {attempt + 1}/{max_retries}: Adding submodules...")
        
        raw = chain.invoke({
            "course_title": modules_course.title,
            "language": language,
            "modules_structure": modules_json,
            "n_submodules": n_submodules,
            "format_instructions": modules_with_submodules_parser.get_format_instructions(),
        })
        
        try:
            result = modules_with_submodules_parser.parse(raw)
            print(f"   ✓ Step 2 complete: {n_submodules} submodules per module")
            return result
        except Exception as e:
            logger.warning(f"Step 2 Attempt {attempt + 1}: Failed to parse: {e}")
            continue
    
    raise ValueError(f"Step 2 failed after {max_retries} attempts.")


def generate_sections_step(
    course_with_submodules: ModulesWithSubmodulesCourse,
    n_sections: int,
    language: str,
    provider: str,
    max_retries: int,
) -> TitleOnlyCourse:
    """
    Step 3: Add sections to each submodule.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    structure_json = course_with_submodules.model_dump_json(indent=2)
    chain = add_sections_prompt | llm | StrOutputParser()
    
    for attempt in range(max_retries):
        print(f"   Step 3 - Attempt {attempt + 1}/{max_retries}: Adding sections...")
        
        raw = chain.invoke({
            "course_title": course_with_submodules.title,
            "language": language,
            "structure_with_submodules": structure_json,
            "n_sections": n_sections,
            "format_instructions": titles_only_parser.get_format_instructions(),
        })
        
        try:
            result = titles_only_parser.parse(raw)
            print(f"   ✓ Step 3 complete: {n_sections} sections per submodule")
            return result
        except Exception as e:
            logger.warning(f"Step 3 Attempt {attempt + 1}: Failed to parse: {e}")
            continue
    
    raise ValueError(f"Step 3 failed after {max_retries} attempts.")


def generate_titles_phase(
    title: str,
    n_modules: int,
    n_submodules: int,
    n_sections: int,
    language: str,
    provider: str,
    max_retries: int,
    research: Optional[CourseResearch] = None,
) -> TitleOnlyCourse:
    """
    Generate course structure with titles only using 3-step hierarchical approach.
    Each step outputs complete structure, maintaining context for coherence.
    """
    # Step 1: Generate module titles (with validation + retry)
    modules_course = generate_modules_step(
        title=title,
        n_modules=n_modules,
        language=language,
        provider=provider,
        max_retries=max_retries,
        research=research,
    )
    
    # Step 2: Add submodules to all modules
    course_with_submodules = generate_submodules_step(
        modules_course=modules_course,
        n_submodules=n_submodules,
        language=language,
        provider=provider,
        max_retries=max_retries,
    )
    
    # Step 3: Add sections to all submodules
    complete_titles = generate_sections_step(
        course_with_submodules=course_with_submodules,
        n_sections=n_sections,
        language=language,
        provider=provider,
        max_retries=max_retries,
    )
    
    return complete_titles


def expand_descriptions_phase(
    titles_course: TitleOnlyCourse,
    language: str,
    provider: str,
    max_retries: int,
) -> SkeletonCourse:
    """
    Final step: Expand titles with descriptions.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    titles_json = titles_course.model_dump_json(indent=2)
    chain = expand_descriptions_prompt | llm | StrOutputParser()
    
    for attempt in range(max_retries):
        print(f"   Adding descriptions - Attempt {attempt + 1}/{max_retries}...")
        
        raw = chain.invoke({
            "course_title": titles_course.title,
            "language": language,
            "titles_structure": titles_json,
            "format_instructions": skeleton_parser.get_format_instructions(),
        })
        
        try:
            skeleton_course = skeleton_parser.parse(raw)
            print(f"   ✓ Descriptions added")
            return skeleton_course
        except Exception as e:
            logger.warning(f"Descriptions Attempt {attempt + 1}: Failed to parse: {e}")
            continue
    
    # Fallback: create skeleton with empty descriptions
    print(f"   ⚠ Description generation failed, using empty descriptions")
    return _titles_to_skeleton(titles_course)


def _titles_to_skeleton(titles_course: TitleOnlyCourse) -> SkeletonCourse:
    """Convert TitleOnlyCourse to SkeletonCourse with empty descriptions."""
    modules = []
    for m in titles_course.modules:
        submodules = []
        for sm in m.submodules:
            sections = [
                SkeletonSection(title=s.title, description="")
                for s in sm.sections
            ]
            submodules.append(SkeletonSubmodule(
                title=sm.title, description="", sections=sections
            ))
        modules.append(SkeletonModule(
            title=m.title, description="", submodules=submodules
        ))
    return SkeletonCourse(title=titles_course.title, modules=modules)


# -------------------------------------------------------
# Skeleton Subgraph Node Functions (LangGraph nodes)
# -------------------------------------------------------
def skeleton_generate_modules_node(state: SkeletonGenerationState) -> dict:
    """
    Node 1: Generate module titles only.
    Raises exception on parse failure (RetryPolicy handles retry).
    """
    print(f"   Step 1: Generating module titles...")
    
    model_name = resolve_text_model_name(state.provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=state.provider, **llm_kwargs)
    
    # Build audience guidelines block
    audience_guidelines = build_audience_guidelines(state.target_audience, context="index")
    
    # Choose prompt based on research availability
    if state.research is not None:
        chain = modules_only_with_research_prompt | llm | StrOutputParser()
        invoke_params = {
            "course_title": state.title,
            "language": state.language,
            "key_topics": "\n".join(f"- {topic}" for topic in state.research.key_topics),
            "learning_objectives": "\n".join(f"- {obj}" for obj in state.research.learning_objectives),
            "n_modules": state.n_modules,
            "format_instructions": modules_only_parser.get_format_instructions(),
            "audience_guidelines": audience_guidelines,
        }
    else:
        chain = modules_only_prompt | llm | StrOutputParser()
        invoke_params = {
            "course_title": state.title,
            "language": state.language,
            "n_modules": state.n_modules,
            "format_instructions": modules_only_parser.get_format_instructions(),
            "audience_guidelines": audience_guidelines,
        }
    
    raw = chain.invoke(invoke_params)
    raw = strip_markdown_fences(raw)  # Pre-process to remove markdown fences
    
    # Try to parse, with fallback to retry parser
    try:
        modules_course = modules_only_parser.parse(raw)
    except Exception as e:
        print(f"   ⚠ Parse failed, attempting correction...")
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=llm, parser=modules_only_parser, max_retries=3
        )
        modules_course = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=correction_prompt.format_prompt(
                error=str(e),
                completion=raw,
                format_instructions=modules_only_parser.get_format_instructions()
            )
        )
    
    # Validate module count
    actual_modules = len(modules_course.modules)
    if actual_modules != state.n_modules:
        raise ValueError(f"Expected {state.n_modules} modules, got {actual_modules}")
    
    print(f"   ✓ Step 1 complete: {state.n_modules} modules generated")
    return {"modules_course": modules_course}


def skeleton_generate_submodules_node(state: SkeletonGenerationState) -> dict:
    """
    Node 2: Add submodules to each module.
    Raises exception on parse failure (RetryPolicy handles retry).
    """
    print(f"   Step 2: Adding submodules...")
    
    model_name = resolve_text_model_name(state.provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=state.provider, **llm_kwargs)
    
    modules_json = state.modules_course.model_dump_json(indent=2)
    chain = add_submodules_prompt | llm | StrOutputParser()
    
    raw = chain.invoke({
        "course_title": state.modules_course.title,
        "language": state.language,
        "modules_structure": modules_json,
        "n_submodules": state.n_submodules,
        "format_instructions": modules_with_submodules_parser.get_format_instructions(),
    })
    raw = strip_markdown_fences(raw)  # Pre-process to remove markdown fences
    
    # Try to parse, with fallback to retry parser
    try:
        result = modules_with_submodules_parser.parse(raw)
    except Exception as e:
        print(f"   ⚠ Parse failed, attempting correction...")
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=llm, parser=modules_with_submodules_parser, max_retries=3
        )
        result = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=correction_prompt.format_prompt(
                error=str(e),
                completion=raw,
                format_instructions=modules_with_submodules_parser.get_format_instructions()
            )
        )
    
    print(f"   ✓ Step 2 complete: {state.n_submodules} submodules per module")
    return {"course_with_submodules": result}


def skeleton_generate_sections_node(state: SkeletonGenerationState) -> dict:
    """
    Node 3: Add sections to each submodule.
    Raises exception on parse failure (RetryPolicy handles retry).
    """
    print(f"   Step 3: Adding sections...")
    
    model_name = resolve_text_model_name(state.provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=state.provider, **llm_kwargs)
    
    structure_json = state.course_with_submodules.model_dump_json(indent=2)
    chain = add_sections_prompt | llm | StrOutputParser()
    
    raw = chain.invoke({
        "course_title": state.course_with_submodules.title,
        "language": state.language,
        "structure_with_submodules": structure_json,
        "n_sections": state.n_sections,
        "format_instructions": titles_only_parser.get_format_instructions(),
    })
    raw = strip_markdown_fences(raw)  # Pre-process to remove markdown fences
    
    # Try to parse, with fallback to retry parser
    try:
        result = titles_only_parser.parse(raw)
    except Exception as e:
        print(f"   ⚠ Parse failed, attempting correction...")
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=llm, parser=titles_only_parser, max_retries=3
        )
        result = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=correction_prompt.format_prompt(
                error=str(e),
                completion=raw,
                format_instructions=titles_only_parser.get_format_instructions()
            )
        )
    
    print(f"   ✓ Step 3 complete: {state.n_sections} sections per submodule")
    return {"titles_course": result}


def skeleton_generate_descriptions_node(state: SkeletonGenerationState) -> dict:
    """
    Node 4: Expand titles with descriptions.
    On persistent failure, falls back to empty descriptions.
    """
    print(f"   Step 4: Adding descriptions...")
    
    model_name = resolve_text_model_name(state.provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=state.provider, **llm_kwargs)
    
    audience_guidelines = build_audience_guidelines(state.target_audience, context="index")
    titles_json = state.titles_course.model_dump_json(indent=2)
    chain = expand_descriptions_prompt | llm | StrOutputParser()
    
    raw = chain.invoke({
        "course_title": state.titles_course.title,
        "language": state.language,
        "titles_structure": titles_json,
        "format_instructions": skeleton_parser.get_format_instructions(),
        "audience_guidelines": audience_guidelines,
    })
    raw = strip_markdown_fences(raw)  # Pre-process to remove markdown fences
    
    # Try to parse, with fallback to retry parser
    try:
        skeleton_course = skeleton_parser.parse(raw)
    except Exception as e:
        print(f"   ⚠ Parse failed, attempting correction...")
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=llm, parser=skeleton_parser, max_retries=3
        )
        skeleton_course = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=correction_prompt.format_prompt(
                error=str(e),
                completion=raw,
                format_instructions=skeleton_parser.get_format_instructions()
            )
        )
    
    print(f"   ✓ Step 4 complete: Descriptions added")
    return {"skeleton_course": skeleton_course}


def skeleton_fallback_descriptions_node(state: SkeletonGenerationState) -> dict:
    """
    Fallback node: Create skeleton with empty descriptions if all retries fail.
    """
    print(f"   ⚠ Description generation failed, using empty descriptions")
    skeleton_course = _titles_to_skeleton(state.titles_course)
    return {"skeleton_course": skeleton_course}


def build_skeleton_graph(max_retries: int = 3):
    """
    Build the skeleton generation subgraph with 4 sequential nodes.
    Uses RetryPolicy for automatic retry on parse failures.
    """
    graph = StateGraph(SkeletonGenerationState)
    
    # Configure retry policy for parse failures
    retry = RetryPolicy(max_attempts=max_retries)
    
    # Add nodes with retry policy
    graph.add_node("generate_modules", skeleton_generate_modules_node, retry=retry)
    graph.add_node("generate_submodules", skeleton_generate_submodules_node, retry=retry)
    graph.add_node("generate_sections", skeleton_generate_sections_node, retry=retry)
    graph.add_node("generate_descriptions", skeleton_generate_descriptions_node, retry=retry)
    
    # Sequential edges
    graph.add_edge(START, "generate_modules")
    graph.add_edge("generate_modules", "generate_submodules")
    graph.add_edge("generate_submodules", "generate_sections")
    graph.add_edge("generate_sections", "generate_descriptions")
    graph.add_edge("generate_descriptions", END)
    
    return graph.compile()


# -------------------------------------------------------
# Research Phase: Generate search queries and synthesize findings
# -------------------------------------------------------
def research_topic(
    title: str,
    description: str | None = None,
    provider: str = "mistral",
    web_search_provider: str = "ddg",
    max_queries: int = 5,
    max_results_per_query: int = 3,
) -> CourseResearch:
    """
    Research the course topic using web search and synthesize findings.
    
    NOTE: All research (queries, searches, synthesis) is conducted in ENGLISH
    for better search results. The final course index will be generated in the
    target language using this English research as context.
    
    Args:
        title: Course title
        description: Course description or context
        provider: LLM provider for query generation and synthesis
        web_search_provider: Web search provider (ddg | tavily | wikipedia)
        max_queries: Maximum number of search queries to generate
        max_results_per_query: Maximum results per search query
    
    Returns:
        CourseResearch with synthesized research findings (in English)
    """
    print(f"🔬 Starting research phase for: {title}")
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.3}  # Slightly higher temp for creative query generation
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Step 1: Generate search queries (always in English for better results)
    print(f"   📝 Generating {max_queries} search queries (in English)...")
    query_chain = query_generation_prompt | llm | StrOutputParser()
    
    raw_queries = query_chain.invoke({
        "course_title": title,
        "course_description": description or "",
        "max_queries": max_queries,
    })
    
    # Parse queries from JSON array
    try:
        # Clean up potential markdown code fences
        clean_queries = raw_queries.strip()
        if clean_queries.startswith("```"):
            clean_queries = clean_queries.split("```")[1]
            if clean_queries.startswith("json"):
                clean_queries = clean_queries[4:]
        queries = json.loads(clean_queries)
        if not isinstance(queries, list):
            queries = [str(queries)]
    except json.JSONDecodeError:
        # Fallback: split by newlines or use as single query
        queries = [q.strip().strip('"') for q in raw_queries.strip().split('\n') if q.strip()]
        if not queries:
            queries = [f"{title} course content"]
    
    queries = queries[:max_queries]  # Limit to max_queries
    print(f"   ✓ Generated queries: {queries}")
    
    # Step 2: Execute searches in parallel
    print(f"   🔍 Executing web searches using {web_search_provider}...")
    web_search = create_web_search(web_search_provider)
    
    all_results = []
    failed_queries = []
    
    def execute_search(query: str) -> tuple[str, bool]:
        """Execute a search and return (result_string, success_flag)."""
        try:
            result = web_search(query, max_results_per_query)
            # Check if result contains error indicators
            result_str = str(result)
            is_error = (
                "Error:" in result_str or 
                "failed:" in result_str or
                "'error':" in result_str or
                "Unauthorized" in result_str
            )
            formatted = f"Query: {query}\n\nResults:\n{result}\n\n{'='*60}\n"
            return formatted, not is_error
        except Exception as e:
            formatted = f"Query: {query}\n\nError: {str(e)}\n\n{'='*60}\n"
            return formatted, False
    
    # Execute searches in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
        futures = {executor.submit(execute_search, q): q for q in queries}
        for future in concurrent.futures.as_completed(futures):
            query = futures[future]
            result, success = future.result()
            all_results.append(result)
            if not success:
                failed_queries.append(query)
    
    raw_research = "\n".join(all_results)
    
    # Log warning if any searches failed
    if failed_queries:
        logger.warning(
            "%d of %d search queries failed. Results may be incomplete. Failed queries: %s",
            len(failed_queries),
            len(queries),
            failed_queries
        )
        print(f"   ⚠ {len(failed_queries)} of {len(queries)} searches failed (check logs for details)")
    
    print(f"   ✓ Collected {len(all_results)} search results ({len(raw_research)} chars)")
    
    # Step 3: Synthesize research into structured output (in English)
    print(f"   🧪 Synthesizing research findings (in English)...")
    
    # Reset temperature for synthesis
    llm_kwargs["temperature"] = 0
    llm_synthesis = create_text_llm(provider=provider, **llm_kwargs)
    
    synthesis_chain = research_synthesis_prompt | llm_synthesis | StrOutputParser()
    
    raw_synthesis = synthesis_chain.invoke({
        "course_title": title,
        "course_description": description or "",
        "raw_research": raw_research[:15000],  # Limit context size
        "format_instructions": research_parser.get_format_instructions(),
    })
    
    # Parse synthesis result
    try:
        research = research_parser.parse(raw_synthesis)
        # Store the raw research for reference
        research.raw_research = raw_research[:10000]  # Truncate for storage
    except Exception as e:
        print(f"   ⚠ Synthesis parsing failed: {e}, using defaults with raw research")
        # Create minimal research output with raw data
        research = CourseResearch(
            course_summary=f"Course about {title}. {description or ''}",
            learning_objectives=[f"Understand {title}"],
            assumed_prerequisites=["Basic knowledge of the subject area"],
            out_of_scope=["Advanced implementation details"],
            key_topics=[title],
            raw_research=raw_research[:10000],
        )
    
    print(f"   ✅ Research complete!")
    print(f"      Summary: {research.course_summary[:100]}...")
    print(f"      Learning objectives: {len(research.learning_objectives)}")
    print(f"      Key topics: {len(research.key_topics)}")
    
    return research

# -------------------------------------------------------
# Conversion function: Skeleton -> Full CourseState
# -------------------------------------------------------
def convert_skeleton_to_course_state(
    skeleton: SkeletonCourse, 
    research: Optional[CourseResearch] = None
) -> CourseState:
    """Convert lightweight skeleton models to full CourseState with empty content fields"""
    
    full_modules = []
    for skel_module in skeleton.modules:
        full_submodules = []
        for skel_submodule in skel_module.submodules:
            full_sections = []
            for skel_section in skel_submodule.sections:
                # Create full Section with empty content fields
                full_section = Section(
                    title=skel_section.title,
                    index=skel_section.index,
                    description=skel_section.description,
                    theory="",  # Empty, to be filled by section_theory_generator
                    html=None,  # None, to be filled by html_formatter
                    meta_elements=None,  # None, to be filled later
                    activities=None,  # None, to be filled by activities_generator
                )
                full_sections.append(full_section)
            
            # Create full Submodule
            full_submodule = Submodule(
                title=skel_submodule.title,
                index=skel_submodule.index,
                description=skel_submodule.description,
                duration=0.0,  # Will be calculated later in workflow
                sections=full_sections,
            )
            full_submodules.append(full_submodule)
        
        # Create full Module
        full_module = Module(
            title=skel_module.title,
            id="",  # Will be set later in workflow
            index=skel_module.index,
            description=skel_module.description,
            duration=0.0,  # Will be calculated later in workflow
            type="module",
            submodules=full_submodules,
        )
        full_modules.append(full_module)
    
    # Return CourseState with empty config and optional research (will be merged in workflow)
    return CourseState(
        config=CourseConfig(),
        research=research,
        title=skeleton.title,
        modules=full_modules
    )

# -------------------------------------------------------
# Summary Generation: Generate section summaries per module
# -------------------------------------------------------
def generate_module_summaries(
    module: Module,
    course_title: str,
    language: str,
    provider: str = "mistral",
    target_audience: AudienceType = None,
) -> dict[str, str]:
    """
    Generate 3-line summaries for all sections in a module.
    
    Processes all sections across all submodules in a single LLM call
    to ensure non-repetitive, complementary summaries.
    
    Args:
        module: Module containing sections to summarize
        course_title: Title of the course for context
        language: Target language for summaries
        provider: LLM provider
        target_audience: Target audience for content adaptation
    
    Returns:
        Dictionary mapping section titles to their 3-line summaries
    """
    # Collect all sections from all submodules
    sections_list_parts = []
    for submodule in module.submodules:
        for section in submodule.sections:
            sections_list_parts.append(
                f"- Title: \"{section.title}\"\n  Description: {section.description}"
            )
    
    sections_list = "\n".join(sections_list_parts)
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.2}  # Slight creativity for varied summaries
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Build audience guidelines
    audience_guidelines = build_audience_guidelines(target_audience, context="index")
    
    # Generate summaries
    chain = summary_generation_prompt | llm | StrOutputParser()
    
    raw_output = chain.invoke({
        "course_title": course_title,
        "module_title": module.title,
        "module_description": module.description,
        "language": language,
        "sections_list": sections_list,
        "audience_guidelines": audience_guidelines,
    })
    
    # Parse JSON output
    try:
        # Clean up potential markdown code fences
        clean_output = raw_output.strip()
        if clean_output.startswith("```"):
            clean_output = clean_output.split("```")[1]
            if clean_output.startswith("json"):
                clean_output = clean_output[4:]
        summaries = json.loads(clean_output)
        if not isinstance(summaries, dict):
            raise ValueError("Expected a dictionary")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse summaries for module '{module.title}': {e}")
        # Return empty dict - sections will have empty summaries
        summaries = {}
    
    return summaries


def generate_all_summaries(
    course_state: CourseState,
    language: str,
    provider: str = "mistral",
) -> None:
    """
    Generate summaries for all sections in all modules, processing modules in parallel.
    
    Updates section.summary fields in-place.
    
    Args:
        course_state: CourseState with modules to process
        language: Target language for summaries
        provider: LLM provider
    """
    print(f"📝 Generating section summaries for {len(course_state.modules)} modules in parallel...")
    
    def process_module(module: Module) -> tuple[str, dict[str, str]]:
        """Process a single module and return (module_title, summaries_dict)."""
        summaries = generate_module_summaries(
            module=module,
            course_title=course_state.title,
            language=language,
            provider=provider,
            target_audience=course_state.config.target_audience if course_state.config else None,
        )
        return module.title, summaries
    
    # Process all modules in parallel
    module_summaries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(course_state.modules), 5)) as executor:
        futures = {executor.submit(process_module, m): m for m in course_state.modules}
        for future in concurrent.futures.as_completed(futures):
            module = futures[future]
            try:
                module_title, summaries = future.result()
                module_summaries[module_title] = summaries
                print(f"   ✓ Generated summaries for module: {module_title}")
            except Exception as e:
                logger.error(f"Failed to generate summaries for module '{module.title}': {e}")
                print(f"   ⚠ Failed summaries for module: {module.title}")
    
    # Apply summaries to sections
    sections_updated = 0
    for module in course_state.modules:
        summaries = module_summaries.get(module.title, {})
        for submodule in module.submodules:
            for section in submodule.sections:
                if section.title in summaries:
                    section.summary = summaries[section.title]
                    sections_updated += 1
    
    print(f"   ✅ Updated {sections_updated} section summaries")


# -------------------------------------------------------
# Summary Subgraph Node Functions (LangGraph Send pattern)
# -------------------------------------------------------
def summary_plan_modules_node(state: SummaryGenerationState) -> dict:
    """
    Plan node: Count modules and update state.
    The actual Send dispatch happens in the routing function.
    """
    print(f"📝 Generating section summaries for {len(state.course_state.modules)} modules in parallel...")
    return {"total_modules": len(state.course_state.modules)}


def summary_continue_to_modules(state: SummaryGenerationState) -> list[Send]:
    """
    Routing function: Create Send tasks for each module to process in parallel.
    Called by conditional edge after plan node.
    """
    sends = []
    for idx, module in enumerate(state.course_state.modules):
        # Collect all sections from all submodules (same logic as original)
        sections_list_parts = []
        for submodule in module.submodules:
            for section in submodule.sections:
                sections_list_parts.append(
                    f"- Title: \"{section.title}\"\n  Description: {section.description}"
                )
        sections_list = "\n".join(sections_list_parts)
        
        task = ModuleSummaryTask(
            course_title=state.course_state.title,
            module_idx=idx,
            module_title=module.title,
            module_description=module.description,
            sections_list=sections_list,
            language=state.language,
            provider=state.provider,
            target_audience=state.course_state.config.target_audience if state.course_state.config else None,
        )
        sends.append(Send("generate_module_summary", task))
    
    return sends


def summary_generate_module_node(state: ModuleSummaryTask) -> dict:
    """
    Generate node: Process one module's summaries.
    EXACT logic from generate_module_summaries() function.
    """
    # Create LLM (same as original)
    model_name = resolve_text_model_name(state.provider)
    llm_kwargs = {"temperature": 0.2}  # Slight creativity for varied summaries
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=state.provider, **llm_kwargs)
    
    # Build audience guidelines
    audience_guidelines = build_audience_guidelines(state.target_audience, context="index")
    
    # Generate summaries (same chain as original)
    chain = summary_generation_prompt | llm | StrOutputParser()
    
    raw_output = chain.invoke({
        "course_title": state.course_title,
        "module_title": state.module_title,
        "module_description": state.module_description,
        "language": state.language,
        "sections_list": state.sections_list,
        "audience_guidelines": audience_guidelines,
    })
    
    # Parse JSON output (same logic as original)
    try:
        clean_output = raw_output.strip()
        if clean_output.startswith("```"):
            clean_output = clean_output.split("```")[1]
            if clean_output.startswith("json"):
                clean_output = clean_output[4:]
        summaries = json.loads(clean_output)
        if not isinstance(summaries, dict):
            raise ValueError("Expected a dictionary")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse summaries for module '{state.module_title}': {e}")
        summaries = {}
    
    print(f"   ✓ Generated summaries for module: {state.module_title}")
    
    return {
        "completed_summaries": [{
            "module_idx": state.module_idx,
            "module_title": state.module_title,
            "summaries": summaries,
        }]
    }


def summary_reduce_node(state: SummaryGenerationState) -> dict:
    """
    Reduce node: Apply all summaries to course_state sections.
    EXACT logic from generate_all_summaries() apply section.
    """
    # Build module_summaries dict from completed_summaries
    module_summaries = {}
    for result in state.completed_summaries:
        module_summaries[result["module_title"]] = result["summaries"]
    
    # Apply summaries to sections (same logic as original)
    sections_updated = 0
    for module in state.course_state.modules:
        summaries = module_summaries.get(module.title, {})
        for submodule in module.submodules:
            for section in submodule.sections:
                if section.title in summaries:
                    section.summary = summaries[section.title]
                    sections_updated += 1
    
    print(f"   ✅ Updated {sections_updated} section summaries")
    
    return {"course_state": state.course_state}


def build_summary_graph(max_concurrency: int = 5):
    """
    Build the summary generation subgraph with Send pattern.
    Processes modules in parallel via Send, then reduces results.
    """
    graph = StateGraph(SummaryGenerationState)
    
    # Add nodes
    graph.add_node("plan_modules", summary_plan_modules_node)
    graph.add_node("generate_module_summary", summary_generate_module_node)
    graph.add_node("reduce_summaries", summary_reduce_node)
    
    # Edges: START -> plan -> (conditional Send) -> generate_module_summary -> reduce -> END
    graph.add_edge(START, "plan_modules")
    graph.add_conditional_edges("plan_modules", summary_continue_to_modules, ["generate_module_summary"])
    graph.add_edge("generate_module_summary", "reduce_summaries")
    graph.add_edge("reduce_summaries", END)
    
    return graph.compile()


# -------------------------------------------------------
# Chain: Generate course index/state with retry-on-parse-failure
# -------------------------------------------------------
def generate_course_state(
    title: str,
    total_pages: int,
    description: str | None = None,
    language: str = "English",
    max_retries: int = 3,
    words_per_page: int = 400,
    provider: str = "mistral",
    # Research configuration
    enable_research: bool = True,
    web_search_provider: str = "ddg",
    research_max_queries: int = 5,
    research_max_results_per_query: int = 3,
    research: Optional[CourseResearch] = None,
    # Audience configuration
    target_audience: AudienceType = None,
) -> CourseState:
    """
    Generate course skeleton and return CourseState with embedded config.
    
    If enable_research is True and research is None, will first conduct
    topic research using web search to inform the course structure.
    
    Args:
        title: Course title
        total_pages: Total pages for the course
        description: Optional course description
        language: Target language for the course
        max_retries: Maximum retries for LLM parsing
        words_per_page: Target words per page
        provider: LLM provider
        enable_research: Whether to conduct research before index generation
        web_search_provider: Web search provider (ddg | tavily | wikipedia)
        research_max_queries: Maximum search queries to generate
        research_max_results_per_query: Maximum results per query
        research: Pre-computed research (if None and enable_research is True, will generate)
    
    Returns:
        CourseState with course structure and optional research
    """
    # Step 1: Conduct research if enabled and not provided (always in English)
    if enable_research and research is None:
        print("📚 Research phase enabled, conducting topic research (in English)...")
        research = research_topic(
            title=title,
            description=description,
            provider=provider,
            web_search_provider=web_search_provider,
            max_queries=research_max_queries,
            max_results_per_query=research_max_results_per_query,
        )
    elif research is not None:
        print("📚 Using provided research data...")
    else:
        print("📚 Research disabled, generating index without research context...")
    
    # Step 2: Compute layout
    n_modules, n_submodules, n_sections = compute_layout(total_pages)
    print(f"🏗️ Target structure: {n_modules} modules × {n_submodules} submodules × {n_sections} sections")

    # Step 3: Generate skeleton via LangGraph subgraph (4 sequential steps with RetryPolicy)
    print("🏗️ Generating course structure (hierarchical via LangGraph)...")
    skeleton_graph = build_skeleton_graph(max_retries=max_retries)
    
    skeleton_initial_state = SkeletonGenerationState(
        title=title,
        language=language,
        research=research,
        n_modules=n_modules,
        n_submodules=n_submodules,
        n_sections=n_sections,
        provider=provider,
        target_audience=target_audience,
    )
    
    skeleton_result = skeleton_graph.invoke(skeleton_initial_state)
    skeleton_course = skeleton_result["skeleton_course"]

    # Step 4: Convert skeleton to full CourseState with research
    course_state = convert_skeleton_to_course_state(skeleton_course, research=research)
    
    # Step 5: Generate section summaries via LangGraph subgraph (Send pattern)
    summary_graph = build_summary_graph(max_concurrency=min(len(course_state.modules), 5))
    
    summary_initial_state = SummaryGenerationState(
        course_state=course_state,
        language=language,
        provider=provider,
        total_modules=len(course_state.modules),
    )
    
    summary_result = summary_graph.invoke(
        summary_initial_state,
        config={"max_concurrency": min(len(course_state.modules), 5)}
    )
    # The course_state is mutated in-place during reduce, but also returned
    course_state = summary_result["course_state"]
    
    print("✅ Course structure generated successfully!")
    return course_state


if __name__ == "__main__":
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    course_title = "Introduction to Modern Data Engineering"
    pages = 100  # total pages desired across the whole course
    desc = "A practical course covering data ingestion, warehousing, orchestration, and observability."

    course_state: CourseState = generate_course_state(
        title=course_title,
        total_pages=pages,
        description=desc,
        max_retries=5,
        words_per_page=400,
        provider="mistral",
        # Research configuration
        enable_research=True,
        web_search_provider="ddg",
        research_max_queries=5,
        research_max_results_per_query=3,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATED COURSE STATE:")
    print("="*80)
    
    if course_state.research:
        print("\n📚 RESEARCH SUMMARY:")
        print(f"   {course_state.research.course_summary[:200]}...")
        print(f"\n📋 LEARNING OBJECTIVES:")
        for obj in course_state.research.learning_objectives:
            print(f"   • {obj}")
        print(f"\n🔑 KEY TOPICS:")
        for topic in course_state.research.key_topics:
            print(f"   • {topic}")
    
    print("\n📖 COURSE STRUCTURE:")
    print(course_state.model_dump_json(indent=2, exclude={"research": {"raw_research"}}))