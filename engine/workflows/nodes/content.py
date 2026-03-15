"""
Content generation nodes for theories and activities.

Agent imports are deferred to the node functions that use them.
"""

from typing import Optional
from langchain_core.runnables import RunnableConfig

from workflows.state import CourseState
from .utils import get_output_manager


def generate_theories_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate all section theories in parallel using LangGraph Send.
    
    This node populates the theory field for each section in the course structure.
    Uses reflection pattern if enabled in config for fact verification.
    
    Args:
        state: CourseState with populated course skeleton.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with all section theories filled.
    """
    print("Generating section theories in parallel...")
    
    from agents.section_theory_generator.agent import generate_all_section_theories

    # Use config settings
    concurrency = state.config.concurrency
    max_retries = state.config.max_retries
    use_reflection = state.config.use_reflection
    num_reflection_queries = state.config.num_reflection_queries
    
    # Run theory generation
    updated_state = generate_all_section_theories(
        state, 
        concurrency=concurrency,
        max_retries=max_retries,
        use_reflection=use_reflection,
        num_reflection_queries=num_reflection_queries
    )
    
    print("All section theories generated successfully!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("theories", updated_state)
    
    return updated_state


def generate_activities_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate activities for all sections in parallel.
    
    This node creates quiz activities, application activities, and meta elements
    (glossary, key concepts, etc.) for each section.
    
    Args:
        state: CourseState with populated section theories.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with all section activities filled.
    """
    print("Generating section activities in parallel...")
    
    from agents.activities_generator.agent import generate_all_section_activities

    updated_state = generate_all_section_activities(
        state,
        concurrency=state.config.activities_concurrency,
        max_retries=state.config.max_retries
    )
    
    print("All section activities generated successfully!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("activities", updated_state)
    
    return updated_state

