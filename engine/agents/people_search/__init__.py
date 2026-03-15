"""
People Search Agent.

Searches for relevant people for course modules using Wikipedia validation.
"""

from .agent import (
    generate_module_people,
    generate_course_people,
    generate_people_node,
)

__all__ = [
    "generate_module_people",
    "generate_course_people",
    "generate_people_node",
]

