"""
Formatting nodes for HTML structure and image generation.

Agent imports are deferred to the node functions that use them.
"""

from typing import Optional
from langchain_core.runnables import RunnableConfig

from workflows.state import CourseState
from .utils import get_output_manager


def generate_html_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate HTML structure for all sections in parallel.
    
    This node transforms section theories into structured HTML elements
    (paragraphs, accordions, tabs, carousels, etc.).
    
    Args:
        state: CourseState with populated section theories.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with all HTML structures filled.
    """
    print("Generating HTML structures in parallel...")
    
    from agents.html_formatter.agent import generate_all_section_html

    updated_state = generate_all_section_html(
        state,
        concurrency=state.config.html_concurrency,
        max_retries=state.config.max_retries
    )
    
    print("All HTML structures generated successfully!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("html", updated_state)
    
    return updated_state


def generate_images_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate images for all HTML blocks using configured image search provider.
    
    This node finds and assigns images to HTML paragraph blocks that have
    image placeholders. Optionally uses vision LLM for ranking.
    
    Args:
        state: CourseState with populated HTML structures.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with all images assigned.
    """
    if not state.config.generate_images:
        print("🖼️ Image generation disabled, skipping...")
        return state
    
    print(f"Generating images for HTML blocks using {state.config.image_search_provider}...")
    if state.config.use_vision_ranking:
        print(f"   Vision ranking enabled: fetching {state.config.num_images_to_fetch} images, ranking with {state.config.vision_llm_provider}")
    else:
        print("   Vision ranking disabled: picking first image result")
    
    from agents.image_search.agent import generate_all_section_images

    updated_state = generate_all_section_images(
        state,
        max_retries=state.config.max_retries,
        use_vision_ranking=state.config.use_vision_ranking,
        num_images_to_fetch=state.config.num_images_to_fetch,
        vision_provider=state.config.vision_llm_provider,
    )
    
    print("All images generated successfully!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("images", updated_state)
    
    return updated_state

