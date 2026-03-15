"""Markdown content digitalizer agent.

Parses a folder of structured markdown files into a CourseState,
preserving content faithfully and extracting local image references.
Optionally restructures the parsed content with LLM assistance.
"""

from .parser import parse_markdown_folder
from .restructurer import restructure_course

__all__ = ["parse_markdown_folder", "restructure_course"]
