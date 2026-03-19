"""
PDF Index Generator Agent

Extracts course structure from PDF syllabus using a 3-step hierarchical approach:
1. Extract modules (titles + durations)
2. Extract submodules (topic headings)
3. Extract sections (bullet points)

Research is used only for enriching descriptions and summaries, not for index generation.
"""

from typing import List, Optional, Annotated
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy

from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from workflows.state import CourseState, CourseConfig, Module, Submodule, Section, CourseResearch
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.pdf2md.docling.converter import convert_pdf_to_markdown
from tools.websearch import create_web_search

from .prompts import (
    extract_modules_prompt,
    extract_submodules_prompt,
    extract_sections_prompt,
    enrich_descriptions_prompt,
    pdf_query_generation_prompt,
    pdf_research_synthesis_prompt,
    pdf_summary_generation_prompt,
    retry_prompt,
)

# Configure logger
logger = logging.getLogger(__name__)

# Helper for state accumulation
def add(a: list, b: list) -> list:
    """Accumulator for list fields in LangGraph state."""
    return a + b


# -------------------------------------------------------
# Pydantic Models for PDF Extraction Steps
# -------------------------------------------------------

# Step 1: Module extraction (titles + durations + descriptions from OBJETIVO)
class PDFModuleExtracted(BaseModel):
    """Module extracted from PDF with title, duration, and description from OBJETIVO"""
    title: str = Field(..., description="Module title exactly as in PDF")
    duration_hours: float = Field(default=0.0, description="Duration in hours (0 if not specified)")
    description: str = Field(default="", description="Module description from OBJETIVO section")


class PDFModulesExtraction(BaseModel):
    """Result of Step 1: All modules extracted from PDF"""
    course_title: str = Field(..., description="Course title from PDF")
    modules: List[PDFModuleExtracted] = Field(..., description="List of extracted modules")


# Step 2: Submodule extraction
class PDFSubmoduleExtracted(BaseModel):
    """Submodule extracted from PDF"""
    title: str = Field(..., description="Submodule title (topic heading)")


class PDFModuleWithSubmodules(BaseModel):
    """Module with its submodules"""
    title: str = Field(..., description="Module title")
    duration_hours: float = Field(default=0.0, description="Duration in hours")
    description: str = Field(default="", description="Module description from OBJETIVO")
    submodules: List[PDFSubmoduleExtracted] = Field(..., description="Submodules in this module")


class PDFSubmodulesExtraction(BaseModel):
    """Result of Step 2: Modules with submodules"""
    course_title: str = Field(..., description="Course title")
    modules: List[PDFModuleWithSubmodules] = Field(..., description="Modules with submodules")


# Step 3: Section extraction
class PDFSectionExtracted(BaseModel):
    """Section extracted from PDF"""
    title: str = Field(..., description="Section title (specific learning point)")


class PDFSubmoduleWithSections(BaseModel):
    """Submodule with its sections"""
    title: str = Field(..., description="Submodule title")
    sections: List[PDFSectionExtracted] = Field(..., description="Sections in this submodule")


class PDFModuleComplete(BaseModel):
    """Complete module with submodules and sections"""
    title: str = Field(..., description="Module title")
    duration_hours: float = Field(default=0.0, description="Duration in hours")
    description: str = Field(default="", description="Module description from OBJETIVO")
    submodules: List[PDFSubmoduleWithSections] = Field(..., description="Submodules with sections")


class PDFSectionsExtraction(BaseModel):
    """Result of Step 3: Complete course structure"""
    course_title: str = Field(..., description="Course title")
    modules: List[PDFModuleComplete] = Field(..., description="Complete modules")


# Final enriched structure with descriptions
class PDFSectionEnriched(BaseModel):
    """Section with description added"""
    title: str = Field(..., description="Section title")
    description: str = Field(default="", description="Section description")


class PDFSubmoduleEnriched(BaseModel):
    """Submodule with description added"""
    title: str = Field(..., description="Submodule title")
    description: str = Field(default="", description="Submodule description")
    sections: List[PDFSectionEnriched] = Field(..., description="Sections with descriptions")


class PDFModuleEnriched(BaseModel):
    """Module with descriptions at all levels"""
    title: str = Field(..., description="Module title")
    duration_hours: float = Field(default=0.0, description="Duration in hours")
    description: str = Field(default="", description="Module description")
    submodules: List[PDFSubmoduleEnriched] = Field(..., description="Submodules with descriptions")


class PDFCourseEnriched(BaseModel):
    """Complete enriched course structure"""
    course_title: str = Field(..., description="Course title")
    modules: List[PDFModuleEnriched] = Field(..., description="Enriched modules")


# Parsers for each step
modules_parser = PydanticOutputParser(pydantic_object=PDFModulesExtraction)
submodules_parser = PydanticOutputParser(pydantic_object=PDFSubmodulesExtraction)
sections_parser = PydanticOutputParser(pydantic_object=PDFSectionsExtraction)
enriched_parser = PydanticOutputParser(pydantic_object=PDFCourseEnriched)
research_parser = PydanticOutputParser(pydantic_object=CourseResearch)


# -------------------------------------------------------
# LangGraph State Models for PDF Extraction Subgraph
# -------------------------------------------------------
class PDFExtractionState(BaseModel):
    """State for PDF extraction subgraph (3 sequential steps + enrichment)."""
    # Inputs (set once at start)
    pdf_markdown: str
    language: str
    provider: str
    
    # Progressive outputs (each step populates the next)
    modules_extraction: Optional[PDFModulesExtraction] = None
    submodules_extraction: Optional[PDFSubmodulesExtraction] = None
    sections_extraction: Optional[PDFSectionsExtraction] = None
    enriched_course: Optional[PDFCourseEnriched] = None
    
    # Research (optional, for enrichment)
    research: Optional[CourseResearch] = None


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


class SummaryGenerationState(BaseModel):
    """State for summary generation subgraph (Send pattern)."""
    course_state: CourseState
    language: str
    provider: str
    # Accumulated results from parallel module processing
    completed_summaries: Annotated[list[dict], add] = Field(default_factory=list)
    total_modules: int = 0


# -------------------------------------------------------
# Step 1: Extract Modules from PDF
# -------------------------------------------------------
def extract_modules_step(
    pdf_markdown: str,
    language: str,
    provider: str,
    max_retries: int = 3,
) -> PDFModulesExtraction:
    """
    Step 1: Extract module titles and durations from PDF.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    chain = extract_modules_prompt | llm | StrOutputParser()
    
    for attempt in range(max_retries):
        print(f"   Step 1 - Attempt {attempt + 1}/{max_retries}: Extracting modules...")
        
        raw = chain.invoke({
            "pdf_markdown": pdf_markdown,
            "language": language,
            "format_instructions": modules_parser.get_format_instructions(),
        })
        
        try:
            result = modules_parser.parse(raw)
            print(f"   ✓ Step 1 complete: {len(result.modules)} modules extracted")
            return result
        except Exception as e:
            logger.warning(f"Step 1 Attempt {attempt + 1}: Failed to parse: {e}")
            continue
    
    raise ValueError(f"Step 1 failed after {max_retries} attempts.")


# -------------------------------------------------------
# Step 2: Extract Submodules for each Module
# -------------------------------------------------------
def extract_submodules_step(
    pdf_markdown: str,
    modules_extraction: PDFModulesExtraction,
    language: str,
    provider: str,
    max_retries: int = 3,
) -> PDFSubmodulesExtraction:
    """
    Step 2: Extract submodules (topic headings) for each module.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    modules_json = modules_extraction.model_dump_json(indent=2)
    chain = extract_submodules_prompt | llm | StrOutputParser()
    
    for attempt in range(max_retries):
        print(f"   Step 2 - Attempt {attempt + 1}/{max_retries}: Extracting submodules...")
        
        raw = chain.invoke({
            "pdf_markdown": pdf_markdown,
            "modules_structure": modules_json,
            "language": language,
            "format_instructions": submodules_parser.get_format_instructions(),
        })
        
        try:
            result = submodules_parser.parse(raw)
            total_submodules = sum(len(m.submodules) for m in result.modules)
            print(f"   ✓ Step 2 complete: {total_submodules} submodules extracted")
            return result
        except Exception as e:
            logger.warning(f"Step 2 Attempt {attempt + 1}: Failed to parse: {e}")
            continue
    
    raise ValueError(f"Step 2 failed after {max_retries} attempts.")


# -------------------------------------------------------
# Step 3: Extract Sections for each Submodule
# -------------------------------------------------------
def extract_sections_step(
    pdf_markdown: str,
    submodules_extraction: PDFSubmodulesExtraction,
    language: str,
    provider: str,
    max_retries: int = 3,
) -> PDFSectionsExtraction:
    """
    Step 3: Extract sections (bullet points/learning outcomes) for each submodule.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    structure_json = submodules_extraction.model_dump_json(indent=2)
    chain = extract_sections_prompt | llm | StrOutputParser()
    
    for attempt in range(max_retries):
        print(f"   Step 3 - Attempt {attempt + 1}/{max_retries}: Extracting sections...")
        
        raw = chain.invoke({
            "pdf_markdown": pdf_markdown,
            "structure_with_submodules": structure_json,
            "language": language,
            "format_instructions": sections_parser.get_format_instructions(),
        })
        
        try:
            result = sections_parser.parse(raw)
            total_sections = sum(
                len(s.sections) 
                for m in result.modules 
                for s in m.submodules
            )
            print(f"   ✓ Step 3 complete: {total_sections} sections extracted")
            return result
        except Exception as e:
            logger.warning(f"Step 3 Attempt {attempt + 1}: Failed to parse: {e}")
            continue
    
    raise ValueError(f"Step 3 failed after {max_retries} attempts.")


# -------------------------------------------------------
# Research Phase (for enrichment only)
# -------------------------------------------------------
def research_pdf_topic(
    course_title: str,
    modules: List[PDFModuleComplete],
    provider: str = "mistral",
    web_search_provider: str = "ddg",
    max_queries: int = 5,
    max_results_per_query: int = 3,
) -> CourseResearch:
    """
    Research the course topic to enrich descriptions and summaries.
    
    NOTE: Research is conducted in ENGLISH for better search results.
    """
    import concurrent.futures
    
    print(f"🔬 Starting research phase for: {course_title}")
    
    # Build context from extracted modules
    module_topics = "\n".join([f"- {m.title}" for m in modules])
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.3}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Step 1: Generate search queries
    print(f"   📝 Generating {max_queries} search queries (in English)...")
    query_chain = pdf_query_generation_prompt | llm | StrOutputParser()
    
    raw_queries = query_chain.invoke({
        "course_title": course_title,
        "module_topics": module_topics,
        "max_queries": max_queries,
    })
    
    # Parse queries from JSON array
    try:
        clean_queries = raw_queries.strip()
        if clean_queries.startswith("```"):
            clean_queries = clean_queries.split("```")[1]
            if clean_queries.startswith("json"):
                clean_queries = clean_queries[4:]
        queries = json.loads(clean_queries)
        if not isinstance(queries, list):
            queries = [str(queries)]
    except json.JSONDecodeError:
        queries = [q.strip().strip('"') for q in raw_queries.strip().split('\n') if q.strip()]
        if not queries:
            queries = [f"{course_title} course content"]
    
    queries = queries[:max_queries]
    print(f"   ✓ Generated queries: {queries}")
    
    # Step 2: Execute searches in parallel
    print(f"   🔍 Executing web searches using {web_search_provider}...")
    web_search = create_web_search(web_search_provider)
    
    all_results = []
    failed_queries = []
    
    def execute_search(query: str) -> tuple[str, bool]:
        try:
            result = web_search(query, max_results_per_query)
            result_str = str(result)
            is_error = any(x in result_str for x in ["Error:", "failed:", "'error':", "Unauthorized"])
            formatted = f"Query: {query}\n\nResults:\n{result}\n\n{'='*60}\n"
            return formatted, not is_error
        except Exception as e:
            return f"Query: {query}\n\nError: {str(e)}\n\n{'='*60}\n", False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
        futures = {executor.submit(execute_search, q): q for q in queries}
        for future in concurrent.futures.as_completed(futures):
            query = futures[future]
            result, success = future.result()
            all_results.append(result)
            if not success:
                failed_queries.append(query)
    
    raw_research = "\n".join(all_results)
    
    if failed_queries:
        logger.warning(f"{len(failed_queries)} of {len(queries)} searches failed")
        print(f"   ⚠ {len(failed_queries)} of {len(queries)} searches failed")
    
    print(f"   ✓ Collected {len(all_results)} search results ({len(raw_research)} chars)")
    
    # Step 3: Synthesize research
    print(f"   🧪 Synthesizing research findings...")
    
    llm_kwargs["temperature"] = 0
    llm_synthesis = create_text_llm(provider=provider, **llm_kwargs)
    
    synthesis_chain = pdf_research_synthesis_prompt | llm_synthesis | StrOutputParser()
    
    raw_synthesis = synthesis_chain.invoke({
        "course_title": course_title,
        "module_topics": module_topics,
        "raw_research": raw_research[:15000],
        "format_instructions": research_parser.get_format_instructions(),
    })
    
    try:
        research = research_parser.parse(raw_synthesis)
        research.raw_research = raw_research[:10000]
    except Exception as e:
        print(f"   ⚠ Synthesis parsing failed: {e}, using defaults")
        research = CourseResearch(
            course_summary=f"Course about {course_title}.",
            learning_objectives=[f"Understand {course_title}"],
            assumed_prerequisites=["Basic knowledge of the subject area"],
            out_of_scope=["Advanced implementation details"],
            key_topics=[course_title],
            raw_research=raw_research[:10000],
        )
    
    print(f"   ✅ Research complete!")
    return research


# -------------------------------------------------------
# Enrich Structure with Descriptions
# -------------------------------------------------------
def enrich_descriptions_step(
    sections_extraction: PDFSectionsExtraction,
    research: Optional[CourseResearch],
    language: str,
    provider: str,
    max_retries: int = 3,
) -> PDFCourseEnriched:
    """
    Add descriptions to all modules, submodules, and sections using research context.
    """
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    structure_json = sections_extraction.model_dump_json(indent=2)
    
    # Prepare research context
    research_context = ""
    if research:
        research_context = f"""
RESEARCH CONTEXT:
Course Summary: {research.course_summary}

Key Topics: {', '.join(research.key_topics)}

Learning Objectives:
{chr(10).join('- ' + obj for obj in research.learning_objectives)}
"""
    
    chain = enrich_descriptions_prompt | llm | StrOutputParser()
    
    for attempt in range(max_retries):
        print(f"   Adding descriptions - Attempt {attempt + 1}/{max_retries}...")
        
        raw = chain.invoke({
            "course_title": sections_extraction.course_title,
            "structure_json": structure_json,
            "research_context": research_context,
            "language": language,
            "format_instructions": enriched_parser.get_format_instructions(),
        })
        
        try:
            result = enriched_parser.parse(raw)
            print(f"   ✓ Descriptions added")
            return result
        except Exception as e:
            logger.warning(f"Descriptions Attempt {attempt + 1}: Failed to parse: {e}")
            continue
    
    # Fallback: create enriched structure with empty descriptions
    print(f"   ⚠ Description generation failed, using empty descriptions")
    return _sections_to_enriched(sections_extraction)


def _sections_to_enriched(sections_extraction: PDFSectionsExtraction) -> PDFCourseEnriched:
    """Convert PDFSectionsExtraction to PDFCourseEnriched, preserving module descriptions from PDF."""
    enriched_modules = []
    for m in sections_extraction.modules:
        enriched_submodules = []
        for sm in m.submodules:
            enriched_sections = [
                PDFSectionEnriched(title=s.title, description="")
                for s in sm.sections
            ]
            enriched_submodules.append(PDFSubmoduleEnriched(
                title=sm.title, description="", sections=enriched_sections
            ))
        enriched_modules.append(PDFModuleEnriched(
            title=m.title,
            duration_hours=m.duration_hours,
            description=m.description,  # Preserve description from PDF (OBJETIVO)
            submodules=enriched_submodules
        ))
    return PDFCourseEnriched(
        course_title=sections_extraction.course_title,
        modules=enriched_modules
    )


# -------------------------------------------------------
# LangGraph Subgraph Node Functions for PDF Extraction
# -------------------------------------------------------
def pdf_extract_modules_node(state: PDFExtractionState) -> dict:
    """Node 1: Extract modules from PDF."""
    print(f"   Step 1: Extracting modules from PDF...")
    
    result = extract_modules_step(
        pdf_markdown=state.pdf_markdown,
        language=state.language,
        provider=state.provider,
        max_retries=3,
    )
    
    return {"modules_extraction": result}


def pdf_extract_submodules_node(state: PDFExtractionState) -> dict:
    """Node 2: Extract submodules for each module."""
    print(f"   Step 2: Extracting submodules...")
    
    result = extract_submodules_step(
        pdf_markdown=state.pdf_markdown,
        modules_extraction=state.modules_extraction,
        language=state.language,
        provider=state.provider,
        max_retries=3,
    )
    
    return {"submodules_extraction": result}


def pdf_extract_sections_node(state: PDFExtractionState) -> dict:
    """Node 3: Extract sections for each submodule."""
    print(f"   Step 3: Extracting sections...")
    
    result = extract_sections_step(
        pdf_markdown=state.pdf_markdown,
        submodules_extraction=state.submodules_extraction,
        language=state.language,
        provider=state.provider,
        max_retries=3,
    )
    
    return {"sections_extraction": result}


def pdf_enrich_descriptions_node(state: PDFExtractionState) -> dict:
    """Node 4: Enrich structure with descriptions."""
    print(f"   Step 4: Adding descriptions...")
    
    result = enrich_descriptions_step(
        sections_extraction=state.sections_extraction,
        research=state.research,
        language=state.language,
        provider=state.provider,
        max_retries=3,
    )
    
    return {"enriched_course": result}


def build_pdf_extraction_graph(max_retries: int = 3):
    """
    Build the PDF extraction subgraph with 4 sequential nodes.
    """
    graph = StateGraph(PDFExtractionState)
    
    retry = RetryPolicy(max_attempts=max_retries)
    
    graph.add_node("extract_modules", pdf_extract_modules_node, retry=retry)
    graph.add_node("extract_submodules", pdf_extract_submodules_node, retry=retry)
    graph.add_node("extract_sections", pdf_extract_sections_node, retry=retry)
    graph.add_node("enrich_descriptions", pdf_enrich_descriptions_node, retry=retry)
    
    graph.add_edge(START, "extract_modules")
    graph.add_edge("extract_modules", "extract_submodules")
    graph.add_edge("extract_submodules", "extract_sections")
    graph.add_edge("extract_sections", "enrich_descriptions")
    graph.add_edge("enrich_descriptions", END)
    
    return graph.compile()


# -------------------------------------------------------
# Summary Generation Subgraph (reused pattern from index_generator)
# -------------------------------------------------------
def summary_plan_modules_node(state: SummaryGenerationState) -> dict:
    """Plan node: Count modules."""
    print(f"📝 Generating section summaries for {len(state.course_state.modules)} modules in parallel...")
    return {"total_modules": len(state.course_state.modules)}


def summary_continue_to_modules(state: SummaryGenerationState) -> list[Send]:
    """Routing function: Create Send tasks for each module."""
    sends = []
    for idx, module in enumerate(state.course_state.modules):
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
        )
        sends.append(Send("generate_module_summary", task))
    
    return sends


def summary_generate_module_node(state: ModuleSummaryTask) -> dict:
    """Generate node: Process one module's summaries."""
    model_name = resolve_text_model_name(state.provider)
    llm_kwargs = {"temperature": 0.2}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=state.provider, **llm_kwargs)
    
    chain = pdf_summary_generation_prompt | llm | StrOutputParser()
    
    raw_output = chain.invoke({
        "course_title": state.course_title,
        "module_title": state.module_title,
        "module_description": state.module_description,
        "language": state.language,
        "sections_list": state.sections_list,
    })
    
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
    """Reduce node: Apply all summaries to course_state sections."""
    module_summaries = {}
    for result in state.completed_summaries:
        module_summaries[result["module_title"]] = result["summaries"]
    
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
    """Build the summary generation subgraph with Send pattern."""
    graph = StateGraph(SummaryGenerationState)
    
    graph.add_node("plan_modules", summary_plan_modules_node)
    graph.add_node("generate_module_summary", summary_generate_module_node)
    graph.add_node("reduce_summaries", summary_reduce_node)
    
    graph.add_edge(START, "plan_modules")
    graph.add_conditional_edges("plan_modules", summary_continue_to_modules, ["generate_module_summary"])
    graph.add_edge("generate_module_summary", "reduce_summaries")
    graph.add_edge("reduce_summaries", END)
    
    return graph.compile()


# -------------------------------------------------------
# Conversion: Enriched PDF Structure -> CourseState
# -------------------------------------------------------
def convert_pdf_to_course_state(
    enriched: PDFCourseEnriched,
    research: Optional[CourseResearch] = None,
) -> CourseState:
    """Convert enriched PDF structure to full CourseState."""
    
    full_modules = []
    for m_idx, pdf_module in enumerate(enriched.modules):
        full_submodules = []
        for sm_idx, pdf_submodule in enumerate(pdf_module.submodules):
            full_sections = []
            for s_idx, pdf_section in enumerate(pdf_submodule.sections):
                full_section = Section(
                    title=pdf_section.title,
                    index=s_idx + 1,
                    description=pdf_section.description,
                    theory="",
                    html=None,
                    meta_elements=None,
                    activities=None,
                )
                full_sections.append(full_section)
            
            full_submodule = Submodule(
                title=pdf_submodule.title,
                index=sm_idx + 1,
                description=pdf_submodule.description,
                duration=0.0,
                sections=full_sections,
            )
            full_submodules.append(full_submodule)
        
        full_module = Module(
            title=pdf_module.title,
            id=str(m_idx + 1),
            index=m_idx + 1,
            description=pdf_module.description,
            duration=pdf_module.duration_hours,
            type="module",
            submodules=full_submodules,
        )
        full_modules.append(full_module)
    
    return CourseState(
        config=CourseConfig(),
        research=research,
        title=enriched.course_title,
        modules=full_modules
    )


# -------------------------------------------------------
# Main Entry Point: Generate CourseState from PDF
# -------------------------------------------------------
def generate_course_state_from_pdf(
    pdf_path: str,
    total_pages: int,
    language: str = "English",
    max_retries: int = 3,
    words_per_page: int = 400,
    provider: str = "openai",
    # Research configuration
    enable_research: bool = True,
    web_search_provider: str = "ddg",
    research_max_queries: int = 5,
    research_max_results_per_query: int = 3,
) -> CourseState:
    """
    Generate course skeleton from PDF syllabus and return CourseState.
    
    Pipeline:
    1. Extract PDF to markdown
    2. Extract modules (titles + durations)
    3. Extract submodules (topic headings)
    4. Extract sections (bullet points)
    5. Research topic (optional, for enrichment)
    6. Enrich with descriptions
    7. Generate section summaries
    
    Args:
        pdf_path: Path to the PDF syllabus file
        total_pages: Total pages for the course content
        language: Language for the course content
        max_retries: Maximum retries for LLM parsing
        words_per_page: Words per page (for downstream use)
        provider: LLM provider to use
        enable_research: Whether to conduct research for enrichment
        web_search_provider: Web search provider
        research_max_queries: Maximum search queries
        research_max_results_per_query: Results per query
    
    Returns:
        CourseState with course structure extracted from PDF
    """
    # Validate PDF path
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF syllabus not found: {pdf_path}")
    
    print(f"📄 Extracting content from PDF: {pdf_path}")
    
    # Step 1: Extract markdown from PDF
    try:
        pdf_markdown = convert_pdf_to_markdown(pdf_path, return_string=True)
        print(f"✓ Extracted {len(pdf_markdown)} characters from PDF")
    except Exception as e:
        raise RuntimeError(f"Failed to extract content from PDF: {e}")
    
    # Step 2-4: Run PDF extraction subgraph (3 steps + descriptions)
    print("🏗️ Extracting course structure from PDF (hierarchical via LangGraph)...")
    
    extraction_graph = build_pdf_extraction_graph(max_retries=max_retries)
    
    extraction_initial_state = PDFExtractionState(
        pdf_markdown=pdf_markdown,
        language=language,
        provider=provider,
    )
    
    # Run extraction without research first
    extraction_result = extraction_graph.invoke(extraction_initial_state)
    sections_extraction = extraction_result["sections_extraction"]
    
    # Step 5: Research (optional, for enrichment)
    research = None
    if enable_research:
        print("📚 Research phase enabled, conducting topic research...")
        research = research_pdf_topic(
            course_title=sections_extraction.course_title,
            modules=sections_extraction.modules,
            provider=provider,
            web_search_provider=web_search_provider,
            max_queries=research_max_queries,
            max_results_per_query=research_max_results_per_query,
        )
    
    # Step 6: Enrich with descriptions (using research context)
    enriched_course = enrich_descriptions_step(
        sections_extraction=sections_extraction,
        research=research,
        language=language,
        provider=provider,
        max_retries=max_retries,
    )
    
    # Step 7: Convert to CourseState
    course_state = convert_pdf_to_course_state(enriched_course, research=research)
    
    # Step 8: Generate section summaries
    print("📝 Generating section summaries...")
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
    course_state = summary_result["course_state"]
    
    print("✅ Course structure extracted successfully from PDF!")
    print(f"   Title: {course_state.title}")
    print(f"   Modules: {len(course_state.modules)}")
    total_sections = sum(len(s.sections) for m in course_state.modules for s in m.submodules)
    print(f"   Total Sections: {total_sections}")
    
    return course_state


if __name__ == "__main__":
    # Test with a sample PDF syllabus
    pdf_syllabus = "example_pdfs/coaching_y_orientacion.pdf"
    pages = 50
    
    course_state: CourseState = generate_course_state_from_pdf(
        pdf_path=pdf_syllabus,
        total_pages=pages,
        language="Español",
        max_retries=5,
        words_per_page=400,
        provider="openai",
        enable_research=True,
        web_search_provider="ddg",
        research_max_queries=5,
        research_max_results_per_query=3,
    )
    
    print("\n" + "="*80)
    print("GENERATED COURSE STATE:")
    print("="*80)
    print(course_state.model_dump_json(indent=2, exclude={"research": {"raw_research"}}))
