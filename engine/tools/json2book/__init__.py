"""
JSON-to-Book Generator Tool.

Converts course.json to professional academic PDF books via LaTeX.
"""

from .generator import generate_pdf_book
from .utils import escape_latex, download_image, generate_bibtex

__all__ = [
    "generate_pdf_book",
    "escape_latex",
    "download_image", 
    "generate_bibtex",
]

