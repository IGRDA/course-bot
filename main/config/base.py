"""
Main CourseConfig class that composes all sub-configurations.

This module provides backward-compatible property aliases so existing code
continues to work while allowing the new nested config structure.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .research import ResearchConfig
from .activities import ActivitiesConfig
from .html import HtmlConfig
from .image import ImageConfig
from .podcast import PodcastConfig
from .bibliography import BibliographyConfig
from .video import VideoConfig
from .people import PeopleConfig
from .mindmap import MindmapConfig


# Mapping of flat field names to (nested_config_name, nested_field_name)
_FLAT_TO_NESTED: dict[str, tuple[str, str]] = {
    # Research
    "enable_research": ("research", "enabled"),
    "research_max_queries": ("research", "max_queries"),
    "research_max_results_per_query": ("research", "max_results_per_query"),
    # Activities
    "activities_concurrency": ("activities", "concurrency"),
    "activity_selection_mode": ("activities", "selection_mode"),
    "sections_per_activity": ("activities", "sections_per_activity"),
    # HTML
    "html_concurrency": ("html", "concurrency"),
    "select_html": ("html", "select_mode"),
    "html_formats": ("html", "formats"),
    "html_random_seed": ("html", "random_seed"),
    "include_quotes_in_html": ("html", "include_quotes"),
    "include_tables_in_html": ("html", "include_tables"),
    # Image
    "image_search_provider": ("image", "search_provider"),
    "use_vision_ranking": ("image", "use_vision_ranking"),
    "num_images_to_fetch": ("image", "num_to_fetch"),
    "vision_llm_provider": ("image", "vision_llm_provider"),
    "image_concurrency": ("image", "concurrency"),
    "imagetext2text_concurrency": ("image", "imagetext2text_concurrency"),
    "vision_ranking_batch_size": ("image", "vision_ranking_batch_size"),
    # Podcast
    "podcast_target_words": ("podcast", "target_words"),
    "podcast_tts_engine": ("podcast", "tts_engine"),
    "podcast_speaker_map": ("podcast", "speaker_map"),
    # Bibliography
    "generate_bibliography": ("bibliography", "enabled"),
    "bibliography_books_per_module": ("bibliography", "books_per_module"),
    "bibliography_articles_per_module": ("bibliography", "articles_per_module"),
    "book_search_provider": ("bibliography", "search_provider"),
    "article_search_provider": ("bibliography", "article_search_provider"),
    # Video
    "generate_videos": ("video", "enabled"),
    "videos_per_module": ("video", "videos_per_module"),
    "video_search_provider": ("video", "search_provider"),
    # People
    "generate_people": ("people", "enabled"),
    "people_per_module": ("people", "people_per_module"),
    "people_llm_provider": ("people", "llm_provider"),
    "people_concurrency": ("people", "concurrency"),
    # Mindmap
    "generate_mindmap": ("mindmap", "enabled"),
    "mindmap_max_nodes": ("mindmap", "max_nodes"),
    "mindmap_llm_provider": ("mindmap", "llm_provider"),
}


class CourseConfig(BaseModel):
    """Configuration parameters for course generation.
    
    This config should not be modified by agents after initialization.
    It uses nested config objects for organization, but also provides
    flat property aliases for backward compatibility.
    """
    
    # ---- Core Settings ----
    title: str = Field(default="", description="Initial title of the course")
    text_llm_provider: str = Field(
        default="mistral",
        description="LLM provider for text generation (mistral | gemini | groq | openai | deepseek)"
    )
    web_search_provider: str = Field(
        default="ddg",
        description="Web search provider (ddg | tavily | wikipedia)"
    )
    total_pages: int = Field(default=50, description="Total number of pages for the course")
    words_per_page: int = Field(default=400, description="Target words per page for content estimation")
    description: str = Field(default="", description="Optional description or context for the course")
    language: str = Field(default="English", description="Language for the content generation")
    target_audience: Literal["kids", "general", "advanced"] | None = Field(
        default=None, 
        description="Target audience for content adaptation"
    )
    pdf_syllabus_path: str = Field(default="", description="Path to PDF syllabus file")
    md_source_path: str = Field(default="", description="Path to local markdown folder for content digitalization")
    generate_html_output: bool = Field(default=False, description="Generate interactive HTML output (digitalizer workflow)")
    generate_podcast: bool = Field(default=False, description="Generate podcast audio (digitalizer workflow)")
    generate_pdf: bool = Field(default=False, description="Generate PDF book output (digitalizer workflow)")
    generate_activities: bool = Field(default=True, description="Generate activities, glossary, and meta-elements (digitalizer workflow)")
    max_retries: int = Field(default=3, description="Maximum number of retries for generation")
    concurrency: int = Field(default=8, description="Number of concurrent section theory generations")
    use_reflection: bool = Field(
        default=False,
        description="Whether to use reflection pattern for fact verification"
    )
    num_reflection_queries: int = Field(
        default=5,
        description="Number of verification queries to generate during reflection"
    )
    theory_only: bool = Field(
        default=False,
        description="If True, no web search (research and reflection off); content from theory only.",
    )
    max_theory_chars_for_llm: int = Field(
        default=500_000,
        description="Max characters of section theory to send to LLM (avoids context length errors; ~70k tokens at 4 chars/token). Truncated text gets a suffix.",
    )
    
    # ---- Nested Configurations ----
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    activities: ActivitiesConfig = Field(default_factory=ActivitiesConfig)
    html: HtmlConfig = Field(default_factory=HtmlConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    podcast: PodcastConfig = Field(default_factory=PodcastConfig)
    bibliography: BibliographyConfig = Field(default_factory=BibliographyConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    people: PeopleConfig = Field(default_factory=PeopleConfig)
    mindmap: MindmapConfig = Field(default_factory=MindmapConfig)
    
    @model_validator(mode="before")
    @classmethod
    def _convert_flat_to_nested(cls, data: Any) -> Any:
        """Convert flat field names to nested config structure for backward compatibility.
        
        This allows existing code like:
            CourseConfig(enable_research=True, research_max_queries=5)
        to work with the new nested structure.
        """
        if not isinstance(data, dict):
            return data
        
        # Group flat fields by their nested config
        nested_updates: dict[str, dict[str, Any]] = {}
        keys_to_remove: list[str] = []
        
        for flat_key, (nested_name, nested_field) in _FLAT_TO_NESTED.items():
            if flat_key in data:
                if nested_name not in nested_updates:
                    nested_updates[nested_name] = {}
                nested_updates[nested_name][nested_field] = data[flat_key]
                keys_to_remove.append(flat_key)
        
        # Remove flat keys from data
        for key in keys_to_remove:
            del data[key]
        
        # Merge nested updates with existing nested configs
        for nested_name, updates in nested_updates.items():
            if nested_name in data:
                # Merge with existing nested config
                existing = data[nested_name]
                if isinstance(existing, dict):
                    existing.update(updates)
                else:
                    # It's already a config object, convert to dict and update
                    data[nested_name] = {**existing.model_dump(), **updates}
            else:
                data[nested_name] = updates
        
        # When theory_only=True, force no web search (research and reflection off)
        if data.get("theory_only") is True:
            if "research" not in data or isinstance(data["research"], dict):
                research = data.get("research") or {}
                if isinstance(research, dict):
                    research = {**research, "enabled": False}
                else:
                    research = {"enabled": False}
                data["research"] = research
            elif hasattr(data["research"], "model_dump"):
                data["research"] = {**data["research"].model_dump(), "enabled": False}
            data["use_reflection"] = False
        return data
    
    # ========================================================================
    # BACKWARD COMPATIBILITY ALIASES
    # These properties allow existing code to use flat attribute access
    # while internally using the nested config structure.
    # ========================================================================
    
    # ---- Research aliases ----
    @property
    def enable_research(self) -> bool:
        return self.research.enabled
    
    @property
    def research_max_queries(self) -> int:
        return self.research.max_queries
    
    @property
    def research_max_results_per_query(self) -> int:
        return self.research.max_results_per_query
    
    # ---- Activities aliases ----
    @property
    def activities_concurrency(self) -> int:
        return self.activities.concurrency
    
    @property
    def activity_selection_mode(self) -> Literal["random", "deterministic"]:
        return self.activities.selection_mode
    
    @property
    def sections_per_activity(self) -> int:
        return self.activities.sections_per_activity
    
    # ---- HTML aliases ----
    @property
    def html_concurrency(self) -> int:
        return self.html.concurrency
    
    @property
    def select_html(self) -> Literal["LLM", "random"]:
        return self.html.select_mode
    
    @property
    def html_formats(self) -> str:
        return self.html.formats
    
    @property
    def html_random_seed(self) -> int:
        return self.html.random_seed
    
    @property
    def include_quotes_in_html(self) -> bool:
        return self.html.include_quotes
    
    @property
    def include_tables_in_html(self) -> bool:
        return self.html.include_tables
    
    # ---- Image aliases ----
    @property
    def image_search_provider(self) -> str:
        return self.image.search_provider
    
    @property
    def use_vision_ranking(self) -> bool:
        return self.image.use_vision_ranking
    
    @property
    def num_images_to_fetch(self) -> int:
        return self.image.num_to_fetch
    
    @property
    def vision_llm_provider(self) -> str:
        return self.image.vision_llm_provider
    
    @property
    def image_concurrency(self) -> int:
        return self.image.concurrency
    
    @property
    def imagetext2text_concurrency(self) -> int:
        return self.image.imagetext2text_concurrency
    
    @property
    def vision_ranking_batch_size(self) -> int:
        return self.image.vision_ranking_batch_size
    
    # ---- Podcast aliases ----
    @property
    def podcast_target_words(self) -> int:
        return self.podcast.target_words
    
    @property
    def podcast_tts_engine(self) -> Literal["edge", "coqui", "elevenlabs", "chatterbox", "openai_tts"]:
        return self.podcast.tts_engine
    
    @property
    def podcast_speaker_map(self) -> dict[str, str] | None:
        return self.podcast.speaker_map
    
    # ---- Bibliography aliases ----
    @property
    def generate_bibliography(self) -> bool:
        return self.bibliography.enabled
    
    @property
    def bibliography_books_per_module(self) -> int:
        return self.bibliography.books_per_module
    
    @property
    def bibliography_articles_per_module(self) -> int:
        return self.bibliography.articles_per_module
    
    @property
    def book_search_provider(self) -> str:
        return self.bibliography.search_provider
    
    @property
    def article_search_provider(self) -> str:
        return self.bibliography.article_search_provider
    
    # ---- Video aliases ----
    @property
    def generate_videos(self) -> bool:
        return self.video.enabled
    
    @property
    def videos_per_module(self) -> int:
        return self.video.videos_per_module
    
    @property
    def video_search_provider(self) -> str:
        return self.video.search_provider
    
    # ---- People aliases ----
    @property
    def generate_people(self) -> bool:
        return self.people.enabled
    
    @property
    def people_per_module(self) -> int:
        return self.people.people_per_module
    
    @property
    def people_llm_provider(self) -> str:
        return self.people.llm_provider
    
    @property
    def people_concurrency(self) -> int:
        return self.people.concurrency
    
    # ---- Mindmap aliases ----
    @property
    def generate_mindmap(self) -> bool:
        return self.mindmap.enabled
    
    @property
    def mindmap_max_nodes(self) -> int:
        return self.mindmap.max_nodes
    
    @property
    def mindmap_llm_provider(self) -> str:
        return self.mindmap.llm_provider

