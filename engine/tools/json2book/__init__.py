"""
JSON-to-Book Generator Tool.

Converts course.json to professional academic PDF books via LaTeX.
"""

from .generator import generate_pdf_book
from .utils import download_image, escape_latex, generate_bibtex

__all__ = [
    "download_image",
    "escape_latex",
    "generate_bibtex",
    "generate_pdf_book",
]
