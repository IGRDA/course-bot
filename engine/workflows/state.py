"""
Course state and domain models.

This module contains all the Pydantic models for representing course structure
and content. The CourseConfig is imported from workflows.config for organization.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Import CourseConfig from the new location (re-export for backward compatibility)
from workflows.config import CourseConfig


# ---- Research Models ----
class CourseResearch(BaseModel):
    """Research output from the topic research phase"""

    course_summary: str = Field(default="", description="Comprehensive summary of the course topic")
    learning_objectives: list[str] = Field(default_factory=list, description="What students will achieve")
    assumed_prerequisites: list[str] = Field(default_factory=list, description="Required prior knowledge")
    out_of_scope: list[str] = Field(default_factory=list, description="Topics explicitly excluded from the course")
    key_topics: list[str] = Field(default_factory=list, description="Canonical domain topics to cover")
    raw_research: str = Field(default="", description="Concatenated raw search results for reference")


# ---- Bibliography Models ----
class BookReference(BaseModel):
    """Single book reference with APA 7 citation."""

    title: str = Field(..., description="Book title")
    authors: list[str] = Field(
        default_factory=list, description="Authors in APA format: ['Last, F. M.', 'Last2, F. M.']"
    )
    year: int | str | None = Field(default=None, description="Publication year or 'n.d.' if unknown")
    publisher: str | None = Field(default=None, description="Publisher name")
    isbn: str | None = Field(default=None, description="ISBN-10 identifier")
    isbn_13: str | None = Field(default=None, description="ISBN-13 identifier")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    url: str | None = Field(default=None, description="Open Library or Google Books URL")
    edition: str | None = Field(default=None, description="Edition (e.g., '2nd ed.')")
    apa_citation: str = Field(default="", description="Pre-formatted APA 7 citation string")

    def get_dedup_key(self) -> str:
        """Generate a key for deduplication based on ISBN or title+author."""
        if self.isbn_13:
            return f"isbn13:{self.isbn_13}"
        if self.isbn:
            return f"isbn:{self.isbn}"
        # Fallback to title + first author
        author_key = self.authors[0].lower() if self.authors else "unknown"
        title_key = self.title.lower().strip()
        return f"title:{title_key}|author:{author_key}"


class ModuleBibliography(BaseModel):
    """Bibliography for a single module."""

    module_index: int = Field(..., description="Module index (1-based)")
    module_title: str = Field(..., description="Module title")
    books: list[BookReference] = Field(default_factory=list, description="Books recommended for this module")


class CourseBibliography(BaseModel):
    """Course-level bibliography with per-module breakdowns and deduplication."""

    modules: list[ModuleBibliography] = Field(default_factory=list, description="Per-module bibliographies")
    all_books: list[BookReference] = Field(default_factory=list, description="Deduplicated master list of all books")

    def get_all_dedup_keys(self) -> set[str]:
        """Get all deduplication keys from the master list."""
        return {book.get_dedup_key() for book in self.all_books}


# ---- Video Models ----
class VideoReference(BaseModel):
    """Single video reference with metadata from YouTube."""

    title: str = Field(..., description="Video title")
    url: str = Field(..., description="YouTube video URL")
    duration: int = Field(default=0, description="Video duration in seconds")
    published_at: int = Field(default=0, description="Publication timestamp in milliseconds")
    thumbnail: str = Field(default="", description="Thumbnail URL")
    channel: str = Field(default="", description="Channel name")
    views: int = Field(default=0, description="View count")
    likes: int = Field(default=0, description="Like count")


class ModuleVideos(BaseModel):
    """Videos for a single module."""

    module_index: int = Field(..., description="Module index (1-based)")
    module_title: str = Field(..., description="Module title")
    query: str = Field(default="", description="Search query used to find videos")
    videos: list[VideoReference] = Field(default_factory=list, description="Video recommendations for this module")


class CourseVideos(BaseModel):
    """Course-level video recommendations with per-module breakdowns."""

    modules: list[ModuleVideos] = Field(default_factory=list, description="Per-module video recommendations")


# ---- Module-Embedded Video Model ----
class ModuleVideoEmbed(BaseModel):
    """Video data embedded within a Module ."""

    type: Literal["video"] = Field(default="video", description="Type identifier")
    query: str = Field(default="", description="Search query used to find videos")
    content: list[VideoReference] = Field(default_factory=list, description="Video recommendations")


# ---- Module-Embedded Bibliography Model ----
class BibliographyItem(BaseModel):
    """Bibliography item for module embedding with full APA 7 citation."""

    title: str = Field(..., description="Book/article title")
    url: str = Field(default="", description="URL to the resource")
    apa_citation: str = Field(default="", description="Full APA 7 citation string")
    item_type: Literal["book", "article"] = Field(default="book", description="Type: book or article")


class ModuleBibliographyEmbed(BaseModel):
    """Bibliography data embedded within a Module ."""

    type: Literal["biblio"] = Field(default="biblio", description="Type identifier")
    query: str = Field(default="", description="Search query used")
    content: list[BibliographyItem] = Field(default_factory=list, description="Bibliography items")


# ---- Person Reference Model ----
class PersonReference(BaseModel):
    """A relevant person with Wikipedia information (for module embedding)."""

    name: str = Field(..., description="Person's full name")
    description: str = Field(..., description="Brief description of the person and their relevance")
    wikiUrl: str = Field(..., description="Wikipedia page URL")
    image: str = Field(default="", description="Wikipedia image URL")


# ---- Mind Map Models ----
class MindmapNodeData(BaseModel):
    """Data for a mind map node."""

    label: str = Field(..., description="Concept name (short, noun-based)")


class MindmapNode(BaseModel):
    """A node in the mind map."""

    id: str = Field(..., description="Unique node ID (e.g., 'root', 'n1', 'n2')")
    level: int = Field(..., description="Hierarchy level (0 for root, 1+ for children)")
    data: MindmapNodeData = Field(..., description="Node data with label")


class MindmapRelationData(BaseModel):
    """Data for a mind map relation."""

    label: str = Field(..., description="Linking phrase/connector (verb or short phrase)")


class MindmapRelation(BaseModel):
    """A relation between two nodes in the mind map."""

    id: str = Field(..., description="Unique relation ID (e.g., 'r1', 'r2')")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    data: MindmapRelationData = Field(..., description="Relation data with linking phrase")


class ModuleMindmap(BaseModel):
    """Mind map for a module (Novak's concept map methodology)."""

    moduleIdx: int = Field(..., description="Module index (1-based)")
    title: str = Field(..., description="Module title")
    nodes: list[MindmapNode] = Field(..., description="List of concept nodes")
    relations: list[MindmapRelation] = Field(..., description="List of relations between nodes")


# ---- Activity Models ----
class GlossaryTerm(BaseModel):
    term: str = Field(..., description="Glossary term")
    explanation: str = Field(..., description="Explanation of the term")


class OrderListContent(BaseModel):
    question: str = Field(..., description="Question for order list activity")
    solution: list[str] = Field(..., min_length=2, description="Ordered list of items")


class FillGapsContent(BaseModel):
    question: str = Field(..., description="Question with *gap* placeholders")
    solution: list[str] = Field(..., min_length=2, description="List of words to fill gaps")


class SwipperContent(BaseModel):
    question: str = Field(..., description="Question for swipper activity")
    solution: dict = Field(..., description="Dictionary with 'true' and 'false' arrays")


class LinkingTermsContent(BaseModel):
    question: str = Field(..., description="Question for linking terms")
    solution: list[dict] = Field(..., min_length=2, description="List of concept-related pairs")


class MultipleChoiceContent(BaseModel):
    question: str = Field(..., description="Multiple choice question")
    solution: str = Field(..., description="Correct answer")
    other_options: list[str] = Field(..., min_length=3, description="Incorrect options")


class MultiSelectionContent(BaseModel):
    question: str = Field(..., description="Multi-selection question")
    solution: list[str] = Field(..., min_length=1, description="Correct answers")
    other_options: list[str] = Field(..., min_length=1, description="Incorrect options")


class Activity(BaseModel):
    type: Literal["order_list", "fill_gaps", "swipper", "linking_terms", "multiple_choice", "multi_selection"] = Field(
        ..., description="Type of activity"
    )
    content: (
        OrderListContent
        | FillGapsContent
        | SwipperContent
        | LinkingTermsContent
        | MultipleChoiceContent
        | MultiSelectionContent
    ) = Field(..., description="Activity content")


class FinalActivityContent(BaseModel):
    question: str = Field(..., description="Final activity question or task description")


class FinalActivity(BaseModel):
    type: Literal["group_activity", "discussion_forum", "individual_project", "open_ended_quiz"] = Field(
        ..., description="Type of final activity"
    )
    content: FinalActivityContent = Field(..., description="Final activity content")


# ---- Meta Elements ----
class MetaElements(BaseModel):
    glossary: list[GlossaryTerm] = Field(default_factory=list, description="Glossary terms for the section")
    key_concept: str = Field(default="", description="Key concept summary for the section")
    interesting_fact: str = Field(default="", description="Interesting fact related to the section")
    quote: dict | None = Field(default=None, description="Quote with author and text fields")


# ---- Activities Section ----
class ActivitiesSection(BaseModel):
    quiz: list[Activity] = Field(default_factory=list, description="Quiz activities for the section")
    application: list[FinalActivity] = Field(default_factory=list, description="Application activities for the section")


# ---- HTML Models ----
class HtmlElement(BaseModel):
    """HTML element with type-specific content structure.

    Content types by element type:
    - 'p': str (paragraph text)
    - 'ul': list[str] (unordered list items)
    - 'quote', 'table': dict[str, Any] (structured data)
    - 'paragraphs', 'accordion', 'tabs', 'carousel', 'flip', 'timeline', 'conversation': list[ParagraphBlock]
      (all interactive formats use the same block structure)
    """

    type: Literal[
        "p", "ul", "quote", "table", "paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"
    ] = Field(..., description="Type of HTML element")
    content: str | list[str] | dict[str, Any] | list["ParagraphBlock"] = Field(
        ..., description="Content of the element"
    )

    @model_validator(mode="after")
    def validate_content_type(self) -> "HtmlElement":
        """Ensure content type matches element type."""
        element_type = self.type
        content = self.content

        # Simple text paragraph
        if element_type == "p":
            if not isinstance(content, str):
                raise ValueError(f"Element type 'p' requires content to be a string, got {type(content).__name__}")

        # Unordered list
        elif element_type == "ul":
            if not isinstance(content, list) or not all(isinstance(item, str) for item in content):
                raise ValueError("Element type 'ul' requires content to be a list of strings")

        # Structured elements (quote, table)
        elif element_type in ["quote", "table"]:
            if not isinstance(content, dict):
                raise ValueError(f"Element type '{element_type}' requires content to be a dictionary")

        # Interactive formats (ALL use the same ParagraphBlock structure)
        elif element_type in ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]:
            if not isinstance(content, list):
                raise ValueError(f"Interactive format '{element_type}' requires content to be a list of ParagraphBlock")
            # Check that items are ParagraphBlock (will be validated by Pydantic)
            for idx, block in enumerate(content):
                if not isinstance(block, (dict, ParagraphBlock)):
                    raise ValueError(f"Interactive format '{element_type}' block {idx} must be a ParagraphBlock")

        return self


class ParagraphBlock(BaseModel):
    title: str = Field(..., description="Title of the paragraph block")
    icon: str = Field(..., description="Material Design Icon class")
    image: dict | None = Field(None, description="Image with type, query, and content URL")
    elements: list[HtmlElement] = Field(..., description="List of HTML elements within this block")


# Update forward references for recursive definition
HtmlElement.model_rebuild()


# ---- Section level ----
class Section(BaseModel):
    title: str = Field(..., description="Title of the section")
    index: int = Field(default=0, description="Section index within submodule")
    description: str = Field(default="", description="Description of the section")
    summary: str = Field(default="", description="3-line summary of what this section will cover")

    theory: str = Field(default="", description="Theory text content for the section")

    html: list[HtmlElement] | None = Field(
        default=None, description="Structured HTML format as direct array of elements"
    )

    meta_elements: MetaElements | None = Field(
        default=None, description="Metadata elements: glossary, key concepts, facts, quotes"
    )

    activities: ActivitiesSection | None = Field(
        default=None, description="Activities section with quiz and application activities"
    )

    source_images: list[dict] | None = Field(
        default=None,
        description=(
            "Local image references extracted from markdown, with positional context. "
            "Each entry: {'alt': str, 'path': str, 'preceding_text': str} "
            "where preceding_text is a snippet of text before the image for inline placement."
        ),
    )


# ---- Submodule level ----
class Submodule(BaseModel):
    title: str = Field(..., description="Title of the submodule")
    index: int = Field(default=0, description="Submodule index within module")
    description: str = Field(default="", description="Description of the submodule")
    duration: float = Field(default=0.0, description="Duration in hours")

    sections: list[Section] = Field(..., description="List of sections in this submodule")


# ---- Module level ----
class Module(BaseModel):
    title: str = Field(..., description="Title of the module")
    id: str = Field(default="", description="Simple string identifier matching index")
    index: int = Field(default=0, description="Module index in course")
    description: str = Field(default="", description="Description of the module")
    duration: float = Field(default=0.0, description="Duration in hours")
    type: Literal["module"] = Field(default="module", description="Type identifier")

    submodules: list[Submodule] = Field(..., description="List of submodules in this module")

    # Module-embedded enrichment data
    video: ModuleVideoEmbed | None = Field(default=None, description="Video recommendations embedded in module")
    bibliography: ModuleBibliographyEmbed | None = Field(default=None, description="Bibliography embedded in module")
    relevant_people: list[PersonReference] | None = Field(
        default=None, description="Relevant people for this module's topic"
    )
    mindmap: ModuleMindmap | None = Field(default=None, description="Concept mind map for this module")


# ---- Course State ----
class CourseState(BaseModel):
    """Main course state - agents should only modify content fields"""

    # Configuration (should not be modified by agents)
    config: CourseConfig = Field(..., description="Course generation configuration")

    # Research output (populated by research phase)
    research: CourseResearch | None = Field(default=None, description="Research output from topic analysis phase")

    # Content fields (can be modified by agents)
    title: str = Field(..., description="Title of the course (initialized from config, can be refined by agents)")
    modules: list[Module] = Field(default_factory=list, description="Full course structure with all modules")

    # Bibliography output (populated by bibliography generator)
    bibliography: CourseBibliography | None = Field(
        default=None, description="Course bibliography with per-module book recommendations"
    )

    # Video recommendations (populated by video generator)
    videos: CourseVideos | None = Field(default=None, description="Video recommendations per module")

    @property
    def description(self) -> str:
        return self.config.description

    @property
    def language(self) -> str:
        return self.config.language

    @property
    def max_retries(self) -> int:
        return self.config.max_retries

    @property
    def concurrency(self) -> int:
        return self.config.concurrency
