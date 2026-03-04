"""
Shared workflow node functions.

This module provides reusable node implementations for LangGraph workflows.
All workflow files should import nodes from here to avoid duplication.

Node sub-modules are imported lazily so that heavy transitive dependencies
(docling, torch, playwright, etc.) are only loaded when a specific node
function is actually called.
"""

from .utils import get_output_manager


def __getattr__(name: str):
    """Lazy module-level attribute access for node functions."""
    # Index generation
    if name in ("generate_index_node", "generate_index_from_pdf_node"):
        from .index import generate_index_node, generate_index_from_pdf_node
        return generate_index_node if name == "generate_index_node" else generate_index_from_pdf_node

    # Content generation
    if name in ("generate_theories_node", "generate_activities_node"):
        from .content import generate_theories_node, generate_activities_node
        return generate_theories_node if name == "generate_theories_node" else generate_activities_node

    # Formatting
    if name in ("generate_html_node", "generate_images_node"):
        from .formatting import generate_html_node, generate_images_node
        return generate_html_node if name == "generate_html_node" else generate_images_node

    # Metadata
    if name == "calculate_metadata_node":
        from .metadata import calculate_metadata_node
        return calculate_metadata_node

    # Digitalize
    if name in ("parse_markdown_folder_node", "inject_local_images_node", "restructure_parsed_content_node"):
        from .digitalize import parse_markdown_folder_node, inject_local_images_node, restructure_parsed_content_node
        _dmap = {
            "parse_markdown_folder_node": parse_markdown_folder_node,
            "inject_local_images_node": inject_local_images_node,
            "restructure_parsed_content_node": restructure_parsed_content_node,
        }
        return _dmap[name]

    # Extras
    _extras = {
        "generate_bibliography_node",
        "generate_videos_node",
        "generate_podcasts_node",
        "generate_pdf_book_node",
        "generate_people_node",
        "generate_mindmap_node",
    }
    if name in _extras:
        from .extras import (
            generate_bibliography_node,
            generate_videos_node,
            generate_podcasts_node,
            generate_pdf_book_node,
            generate_people_node,
            generate_mindmap_node,
        )
        _map = {
            "generate_bibliography_node": generate_bibliography_node,
            "generate_videos_node": generate_videos_node,
            "generate_podcasts_node": generate_podcasts_node,
            "generate_pdf_book_node": generate_pdf_book_node,
            "generate_people_node": generate_people_node,
            "generate_mindmap_node": generate_mindmap_node,
        }
        return _map[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Utils
    "get_output_manager",
    # Index generation
    "generate_index_node",
    "generate_index_from_pdf_node",
    # Content generation
    "generate_theories_node",
    "generate_activities_node",
    # Formatting
    "generate_html_node",
    "generate_images_node",
    # Metadata
    "calculate_metadata_node",
    # Extras
    "generate_bibliography_node",
    "generate_videos_node",
    "generate_people_node",
    "generate_mindmap_node",
    "generate_podcasts_node",
    "generate_pdf_book_node",
    # Digitalize
    "parse_markdown_folder_node",
    "inject_local_images_node",
    "restructure_parsed_content_node",
]
