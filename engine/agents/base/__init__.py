"""
Base classes and utilities for agents.

This module provides reusable abstractions for common agent patterns,
including the fan-out/fan-in pattern for parallel section processing.
"""

from .parallel_processor import (
    SectionTask,
    SectionProcessorState,
    SectionProcessor,
    build_section_processor_graph,
)

__all__ = [
    "SectionTask",
    "SectionProcessorState",
    "SectionProcessor",
    "build_section_processor_graph",
]

