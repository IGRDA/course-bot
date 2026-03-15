"""
Generic parallel section processor pattern.

This module provides reusable components for the fan-out/fan-in pattern
used by section-processing agents (theory, activities, HTML, images).

The pattern consists of:
1. Plan - Count sections and prepare for processing
2. Fan-out - Create parallel tasks via LangGraph Send
3. Process - Execute each task independently
4. Fan-in (Reduce) - Aggregate results back into CourseState

Usage:
    class MyProcessor(SectionProcessor):
        def create_task(self, course_state, m_idx, sm_idx, s_idx, section) -> dict:
            # Return task-specific data
            return {"section_title": section.title, "theory": section.theory}
        
        def process_section(self, task: SectionTask) -> dict:
            # Process and return results
            return {"result_key": "result_value"}
        
        def reduce_result(self, section, result: dict) -> None:
            # Update section with result
            section.html = result["html"]
    
    processor = MyProcessor()
    updated_state = processor.process_all(course_state, concurrency=8)
"""

from abc import ABC, abstractmethod
from operator import add
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy

from workflows.state import CourseState, Section


class SectionTask(BaseModel):
    """Base task for processing a single section.
    
    Subclasses can extend this with additional fields as needed.
    The `extra_data` field can hold task-specific data without subclassing.
    """
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    extra_data: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def section(self) -> Section:
        """Get the section this task refers to."""
        return self.course_state.modules[self.module_idx].submodules[self.submodule_idx].sections[self.section_idx]
    
    @property
    def module_title(self) -> str:
        """Get the module title."""
        return self.course_state.modules[self.module_idx].title
    
    @property
    def submodule_title(self) -> str:
        """Get the submodule title."""
        return self.course_state.modules[self.module_idx].submodules[self.submodule_idx].title
    
    @property
    def section_title(self) -> str:
        """Get the section title."""
        return self.section.title
    
    @property
    def config(self):
        """Get the course config."""
        return self.course_state.config
    
    @property
    def language(self) -> str:
        """Get the course language."""
        return self.course_state.language


class SectionProcessorState(BaseModel):
    """State for the section processor subgraph."""
    course_state: CourseState
    completed_results: Annotated[list[dict], add] = Field(default_factory=list)
    total_sections: int = 0


class SectionProcessor(ABC):
    """Abstract base class for parallel section processing.
    
    Subclasses must implement:
    - create_task_data: Prepare task-specific data for each section
    - process_section: Process a single section and return results
    - reduce_result: Apply results to update the section
    
    The base class handles:
    - Planning (counting sections)
    - Fan-out (creating Send tasks)
    - Graph construction
    - Execution with concurrency control
    """
    
    def __init__(self, name: str = "section_processor"):
        """Initialize the processor.
        
        Args:
            name: Name for this processor (used in graph node names).
        """
        self.name = name
    
    @abstractmethod
    def create_task_data(
        self,
        course_state: CourseState,
        module_idx: int,
        submodule_idx: int,
        section_idx: int,
        section: Section,
    ) -> dict[str, Any]:
        """Create task-specific data for a section.
        
        Override this to add any extra data needed for processing.
        
        Args:
            course_state: Full course state.
            module_idx: Index of the module.
            submodule_idx: Index of the submodule.
            section_idx: Index of the section.
            section: The section being processed.
            
        Returns:
            Dictionary of extra data to include in the task.
        """
        pass
    
    @abstractmethod
    def process_section(self, task: SectionTask) -> dict[str, Any]:
        """Process a single section.
        
        Override this to implement the actual processing logic.
        
        Args:
            task: The section task with all context.
            
        Returns:
            Dictionary of results to be used in reduce_result.
        """
        pass
    
    @abstractmethod
    def reduce_result(
        self,
        section: Section,
        result: dict[str, Any],
    ) -> None:
        """Apply processing result to update the section.
        
        Override this to update the section with the processing results.
        
        Args:
            section: The section to update.
            result: The processing result from process_section.
        """
        pass
    
    def plan(self, state: SectionProcessorState) -> dict:
        """Count total sections for processing."""
        section_count = 0
        for module in state.course_state.modules:
            for submodule in module.submodules:
                section_count += len(submodule.sections)
        
        print(f"📋 Planning {section_count} sections for {self.name}")
        return {"total_sections": section_count}
    
    def fan_out(self, state: SectionProcessorState) -> list[Send]:
        """Create Send tasks for parallel processing."""
        sends = []
        
        for m_idx, module in enumerate(state.course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    # Create task with extra data from subclass
                    extra_data = self.create_task_data(
                        state.course_state, m_idx, sm_idx, s_idx, section
                    )
                    
                    task = SectionTask(
                        course_state=state.course_state,
                        module_idx=m_idx,
                        submodule_idx=sm_idx,
                        section_idx=s_idx,
                        extra_data=extra_data,
                    )
                    
                    sends.append(Send(f"process_{self.name}", task))
        
        return sends
    
    def _process_wrapper(self, task: SectionTask) -> dict:
        """Wrapper that calls process_section and formats result."""
        result = self.process_section(task)
        
        print(f"✓ Processed {self.name} for Module {task.module_idx+1}, "
              f"Submodule {task.submodule_idx+1}, Section {task.section_idx+1}")
        
        return {
            "completed_results": [{
                "module_idx": task.module_idx,
                "submodule_idx": task.submodule_idx,
                "section_idx": task.section_idx,
                **result,
            }]
        }
    
    def reduce(self, state: SectionProcessorState) -> dict:
        """Aggregate all results back into the course state."""
        print(f"📦 Reducing {len(state.completed_results)} completed {self.name} results")
        
        for result in state.completed_results:
            m_idx = result["module_idx"]
            sm_idx = result["submodule_idx"]
            s_idx = result["section_idx"]
            
            section = state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx]
            self.reduce_result(section, result)
        
        print(f"✅ All {state.total_sections} sections processed by {self.name}!")
        return {"course_state": state.course_state}
    
    def build_graph(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """Build the LangGraph for this processor.
        
        Args:
            max_retries: Number of retries for each section.
            initial_delay: Initial delay before first retry.
            backoff_factor: Multiplier for exponential backoff.
            
        Returns:
            Compiled LangGraph.
        """
        graph = StateGraph(SectionProcessorState)
        
        retry_policy = RetryPolicy(
            max_attempts=max_retries,
            initial_interval=initial_delay,
            backoff_factor=backoff_factor,
            max_interval=60.0,
        )
        
        # Add nodes
        graph.add_node(f"plan_{self.name}", self.plan)
        graph.add_node(f"process_{self.name}", self._process_wrapper, retry=retry_policy)
        graph.add_node(f"reduce_{self.name}", self.reduce)
        
        # Add edges
        graph.add_edge(START, f"plan_{self.name}")
        graph.add_conditional_edges(
            f"plan_{self.name}",
            self.fan_out,
            [f"process_{self.name}"]
        )
        graph.add_edge(f"process_{self.name}", f"reduce_{self.name}")
        graph.add_edge(f"reduce_{self.name}", END)
        
        return graph.compile()
    
    def process_all(
        self,
        course_state: CourseState,
        concurrency: int = 8,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> CourseState:
        """Process all sections in parallel.
        
        Args:
            course_state: CourseState to process.
            concurrency: Maximum concurrent tasks.
            max_retries: Retry attempts per section.
            initial_delay: Initial retry delay.
            backoff_factor: Backoff multiplier.
            
        Returns:
            Updated CourseState with all sections processed.
        """
        print(f"🚀 Starting parallel {self.name} with max_retries={max_retries}, "
              f"initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
        
        graph = self.build_graph(
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
        )
        
        initial_state = SectionProcessorState(course_state=course_state)
        result = graph.invoke(initial_state, config={"max_concurrency": concurrency})
        
        return result["course_state"]


def build_section_processor_graph(
    name: str,
    create_task_data: Callable[[CourseState, int, int, int, Section], dict[str, Any]],
    process_section: Callable[[SectionTask], dict[str, Any]],
    reduce_result: Callable[[Section, dict[str, Any]], None],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
):
    """Functional alternative to subclassing SectionProcessor.
    
    Use this when you prefer functions over classes.
    
    Args:
        name: Name for this processor.
        create_task_data: Function to create task-specific data.
        process_section: Function to process a single section.
        reduce_result: Function to apply results to section.
        max_retries: Retry attempts per section.
        initial_delay: Initial retry delay.
        backoff_factor: Backoff multiplier.
        
    Returns:
        Compiled LangGraph.
    
    Example:
        def create_data(state, m, sm, s, section):
            return {"theory": section.theory}
        
        def process(task):
            return {"html": generate_html(task.extra_data["theory"])}
        
        def reduce(section, result):
            section.html = result["html"]
        
        graph = build_section_processor_graph(
            "html_generator",
            create_data,
            process,
            reduce,
        )
    """
    
    class FunctionalProcessor(SectionProcessor):
        def create_task_data(self, course_state, m_idx, sm_idx, s_idx, section):
            return create_task_data(course_state, m_idx, sm_idx, s_idx, section)
        
        def process_section(self, task):
            return process_section(task)
        
        def reduce_result(self, section, result):
            return reduce_result(section, result)
    
    processor = FunctionalProcessor(name=name)
    return processor.build_graph(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
    )

