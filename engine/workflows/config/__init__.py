"""
Course configuration models.

This module provides modular configuration classes for different aspects
of course generation. The main CourseConfig class composes these together.
"""

from .research import ResearchConfig
from .activities import ActivitiesConfig
from .html import HtmlConfig
from .image import ImageConfig
from .podcast import PodcastConfig
from .bibliography import BibliographyConfig
from .video import VideoConfig
from .people import PeopleConfig
from .mindmap import MindmapConfig
from .base import CourseConfig

__all__ = [
    "ResearchConfig",
    "ActivitiesConfig",
    "HtmlConfig",
    "ImageConfig",
    "PodcastConfig",
    "BibliographyConfig",
    "VideoConfig",
    "PeopleConfig",
    "MindmapConfig",
    "CourseConfig",
]

