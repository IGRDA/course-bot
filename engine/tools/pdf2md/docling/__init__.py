"""Docling PDF-to-Markdown converter.

The actual converter module is imported lazily so that heavy dependencies
(docling, docling_core, easyocr, torch) are only loaded when the
convert_pdf_to_markdown function is actually called.
"""


def __getattr__(name: str):
    if name == "convert_pdf_to_markdown":
        from .converter import convert_pdf_to_markdown
        return convert_pdf_to_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["convert_pdf_to_markdown"]
