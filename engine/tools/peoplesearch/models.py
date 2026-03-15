"""
Pydantic models for people search results.

Defines structured output for relevant people found via Wikipedia.
"""

from pydantic import BaseModel, Field


class PersonResult(BaseModel):
    """A person relevant to a topic with Wikipedia information.
    
    Attributes:
        name: Person's full name
        description: Brief description of the person and their relevance
        wikiUrl: URL to the person's Wikipedia page
        image: URL to the person's Wikipedia image/thumbnail
    """
    name: str = Field(..., description="Person's full name")
    description: str = Field(..., description="Brief description of the person")
    wikiUrl: str = Field(..., description="Wikipedia page URL")
    image: str = Field(..., description="Wikipedia image URL")


class PersonSuggestion(BaseModel):
    """LLM-generated suggestion of a person relevant to a topic.
    
    Used internally before Wikipedia validation.
    """
    name: str = Field(..., description="Person's full name as it would appear on Wikipedia")
    reason: str = Field(..., description="Brief reason why this person is relevant to the topic")


class PeopleSuggestionResponse(BaseModel):
    """Response from LLM containing list of suggested people."""
    people: list[PersonSuggestion] = Field(
        ..., 
        description="List of people relevant to the topic"
    )

