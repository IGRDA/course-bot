"""Video HTML Generator Agent.

This agent simplifies module JSON files by extracting only essential fields
for video generation: title, index, and the first HTML element from each section.
"""

from .agent import simplify_module, simplify_module_from_path

__all__ = ["simplify_module", "simplify_module_from_path"]

