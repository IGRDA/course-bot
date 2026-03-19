"""
Image Generator Agent using Gemini Batch API.

This agent generates images from query fields in module JSON files using
Google's Gemini Batch API, saves them to an images folder, and updates
the JSON with base64-encoded image content.
"""

from .agent import (
    extract_image_queries,
    generate_images_batch,
    process_module_images,
    ImageLocation,
)

__all__ = [
    "extract_image_queries",
    "generate_images_batch",
    "process_module_images",
    "ImageLocation",
]




