"""
Metadata calculation node for course structure.
"""

from typing import Optional
from langchain_core.runnables import RunnableConfig

from workflows.state import CourseState
from .utils import get_output_manager


def calculate_metadata_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Calculate IDs, indexes and durations for all course elements.
    
    This node assigns:
    - Unique IDs and indexes to modules, submodules, and sections
    - Default descriptions where missing
    - Duration estimates based on section count
    
    Args:
        state: CourseState with populated course structure.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with all metadata calculated.
    """
    print("Calculating course metadata (IDs, Indexes, Durations)...")
    
    for m_idx, module in enumerate(state.modules):
        # Simple string ID matching index
        module.id = str(m_idx + 1)
        module.index = m_idx + 1
        
        if not module.description:
            module.description = module.title
            
        for sm_idx, submodule in enumerate(module.submodules):
            submodule.index = sm_idx + 1
            
            for s_idx, section in enumerate(submodule.sections):
                section.index = s_idx + 1
            
            # Calculate submodule duration: 0.1 hours per section
            submodule.duration = round(len(submodule.sections) * 0.1, 1)
            
        # If module has duration from PDF, keep it; otherwise calculate
        if module.duration == 0.0:
            module.duration = round(sum(sm.duration for sm in module.submodules), 1)
        
    print("Metadata calculation completed!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("metadata", state)
    
    return state

