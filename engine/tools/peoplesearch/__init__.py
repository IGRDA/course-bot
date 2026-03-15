"""
People Search Tool

Find notable people relevant to a topic using LLM + Wikipedia.

Example:
    >>> from tools.peoplesearch import search_relevant_people
    >>> 
    >>> people = search_relevant_people(
    ...     topic="quantum physics",
    ...     max_results=3,
    ...     language="es"
    ... )
    >>> for person in people:
    ...     print(f"{person.name}: {person.wikiUrl}")
"""

from .models import PersonResult
from .search import search_relevant_people

__all__ = [
    "search_relevant_people",
    "PersonResult",
]
