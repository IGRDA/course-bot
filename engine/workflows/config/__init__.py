"""
Course configuration models.

This module provides modular configuration classes for different aspects
of course generation. The main CourseConfig class composes these together.
"""

from .activities import ActivitiesConfig
from .base import CourseConfig
from .bibliography import BibliographyConfig
from .html import HtmlConfig
from .image import ImageConfig
from .mindmap import MindmapConfig
from .people import PeopleConfig
from .podcast import PodcastConfig
from .research import ResearchConfig
from .video import VideoConfig

__all__ = [
    "ActivitiesConfig",
    "BibliographyConfig",
    "CourseConfig",
    "HtmlConfig",
    "ImageConfig",
    "MindmapConfig",
    "PeopleConfig",
    "PodcastConfig",
    "ResearchConfig",
    "VideoConfig",
]
