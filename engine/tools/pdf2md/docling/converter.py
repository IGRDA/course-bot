"""
PDF to Markdown converter using the Docling library.

All heavy imports (docling, docling_core, easyocr, torch) are deferred
to function bodies so that importing this module does not pull in ~2 GB
of transitive dependencies at module-load time.
"""

import os
import platform
import tempfile
from pathlib import Path
import logging

# Configure logging to see conversion details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language mappings for different OCR engines
# Each engine uses different language codes
LANGUAGE_MAP = {
    "ocrmac": {
        "es": ["es-ES"],
        "en": ["en-US"],
        "fr": ["fr-FR"],
        "de": ["de-DE"],
        "pt": ["pt-PT"],
        "it": ["it-IT"],
    },
    "easyocr": {
        "es": ["es"],
        "en": ["en"],
        "fr": ["fr"],
        "de": ["de"],
        "pt": ["pt"],
        "it": ["it"],
    },
    "tesseract": {
        "es": ["spa"],
        "en": ["eng"],
        "fr": ["fra"],
        "de": ["deu"],
        "pt": ["por"],
        "it": ["ita"],
    },
}


def convert_pdf_to_markdown(
    pdf_path: str | Path,
    return_string: bool = False,
    output_dir: str | Path | None = None,
    ocr_engine: str = "auto",  # Auto-detect best engine for platform
    language: str = "es",
    images_scale: float = 2.0,
    extract_tables: bool = True,
    extract_images: bool = True,
    table_mode: str = "accurate",
    force_ocr: bool = True,
    table_text_handling: str = "hybrid",
):
    """
    Convert PDF to Markdown with configurable OCR and extraction options.
    
    Optimized for Spanish documents by default. Uses force_full_page_ocr to bypass
    potentially corrupted PDF text layers and extract text via OCR instead.
    
    Args:
        pdf_path: Path to the PDF file
        return_string: If True, return markdown as string; if False, save to file and return path
        output_dir: Optional directory for the output markdown file. If None, uses
                    output/docling/{doc_name}/ under the current working directory.
        ocr_engine: OCR engine to use. Options:
                    - 'ocrmac': macOS native OCR (fastest on Mac, default)
                    - 'easyocr': Deep learning based (high accuracy, slower)
                    - 'tesseract': Google's Tesseract OCR (good accuracy, widely supported)
                    - 'auto': Let docling choose best for platform
        language: Language code for OCR. Options: 'es' (Spanish, default), 'en', 'fr', 'de', 'pt', 'it'
                  This helps OCR engines better recognize language-specific characters.
        images_scale: DPI scale for images (default 2.0 = 2x quality, higher = better quality but slower)
        extract_tables: Whether to extract tables with structure (default True)
        extract_images: Whether to extract images from the PDF and embed them in markdown (default True)
        table_mode: Table extraction mode - 'fast' or 'accurate' (default 'accurate')
        force_ocr: If True (default), forces full-page OCR to bypass potentially corrupted
                   PDF text layers. Set to False only if you trust the PDF's embedded text.
        table_text_handling: How to handle text inside tables:
                            - 'structure': Keep markdown table formatting (may have encoding issues in cells)
                            - 'ocr': Disable table detection, full OCR (correct text, no table formatting)
                            - 'hybrid': (default) Tables enabled with force_ocr for best balance
    
    Returns:
        str: Markdown content if return_string=True, otherwise path to saved markdown file
    
    Examples:
        # Default: Spanish document with OCR
        >>> md = convert_pdf_to_markdown("document.pdf", return_string=True)
        
        # English document
        >>> md = convert_pdf_to_markdown("document.pdf", language="en")
        
        # Prioritize correct text over table formatting
        >>> md = convert_pdf_to_markdown("document.pdf", table_text_handling="ocr")
        
        # Trust PDF's embedded text (no OCR)
        >>> md = convert_pdf_to_markdown("document.pdf", force_ocr=False)
        
        # Write to a custom directory
        >>> path = convert_pdf_to_markdown("document.pdf", output_dir="course/md_course")
    """
    # Setup paths
    pdf_path = Path(pdf_path).resolve()
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Extract document name from path
    doc_name = pdf_path.stem

    # Output directory: custom if provided, else default
    if output_dir is not None:
        output_base = Path(output_dir).resolve()
    else:
        output_base = Path.cwd() / "output" / "docling" / doc_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {pdf_path}")
    logger.info(f"Output directory: {output_base}")
    logger.info(f"OCR Engine: {ocr_engine}, Language: {language}, Force OCR: {force_ocr}")
    logger.info(f"Table extraction: {extract_tables} (mode: {table_mode}, handling: {table_text_handling})")

    # --- Lazy imports of heavy docling dependencies ---
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.base_models import InputFormat

    # Step 1: Configure pipeline with optimized settings
    pipeline_options = PdfPipelineOptions()
    
    # Image quality settings
    pipeline_options.images_scale = images_scale
    pipeline_options.generate_picture_images = extract_images
    pipeline_options.generate_table_images = False
    
    # Table extraction settings based on table_text_handling strategy
    if table_text_handling == "ocr":
        pipeline_options.do_table_structure = False
        logger.info("Table structure disabled - OCR will handle all text including tables")
    elif extract_tables:
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
        )
        pipeline_options.table_structure_options.do_cell_matching = True
    else:
        pipeline_options.do_table_structure = False
    
    # OCR engine configuration
    pipeline_options.do_ocr = True
    ocr_options = _get_ocr_options(ocr_engine, language, force_ocr)
    if ocr_options is not None:
        pipeline_options.ocr_options = ocr_options
    
    # Step 2: Convert document (with automatic repair on failure)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    import time
    repaired_path = None
    try:
        start_time = time.time()
        result = converter.convert(pdf_path)
        elapsed = time.time() - start_time
        logger.info(f"Conversion completed in {elapsed:.2f} seconds")
    except Exception as first_error:
        logger.warning(f"Initial conversion failed: {first_error}")
        logger.info("Attempting PDF repair with PyMuPDF and retrying...")
        try:
            repaired_path = _repair_pdf(pdf_path, output_base)
            start_time = time.time()
            result = converter.convert(repaired_path)
            elapsed = time.time() - start_time
            logger.info(f"Conversion succeeded after repair in {elapsed:.2f} seconds")
        except Exception as second_error:
            logger.error(f"Conversion failed even after PDF repair: {second_error}")
            raise RuntimeError(
                f"PDF conversion failed.\n"
                f"  Original error: {first_error}\n"
                f"  After repair: {second_error}"
            ) from second_error

    # Step 3: Export markdown (with or without images)
    markdown_path = output_base / f"{doc_name}.md"

    if extract_images:
        from docling_core.types.doc import ImageRefMode
        result.document.save_as_markdown(
            markdown_path,
            image_mode=ImageRefMode.REFERENCED,
        )
        logger.info(f"✓ Markdown with referenced images saved to: {markdown_path}")
    else:
        markdown_content = result.document.export_to_markdown()
        markdown_path.write_text(markdown_content, encoding="utf-8")
        logger.info(f"✓ Markdown saved to: {markdown_path}")

    markdown_content = markdown_path.read_text(encoding="utf-8")
    logger.info(f"✓ Markdown: {len(markdown_content)} characters")

    # Clean up temporary repaired PDF
    if repaired_path and repaired_path.exists():
        try:
            repaired_path.unlink()
            logger.debug(f"Cleaned up repaired PDF: {repaired_path}")
        except OSError:
            pass

    if return_string:
        return markdown_content
    else:
        return str(markdown_path)


def _repair_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Re-save a PDF via PyMuPDF to normalise page trees and metadata.

    Some PDFs (e.g. those produced by XeLaTeX) store /MediaBox only in the
    parent Pages node.  Docling's parser expects it on every page and fails
    with ``could not find the page-dimensions``.  Re-saving through PyMuPDF
    forces explicit page geometry on every page object, fixing this class of
    issues.

    Args:
        pdf_path: Path to the original PDF.
        output_dir: Directory to write the repaired file.

    Returns:
        Path to the repaired PDF.

    Raises:
        ImportError: If PyMuPDF is not installed.
        RuntimeError: If the repair itself fails.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF repair but is not installed. "
            "Install it with: pip install pymupdf"
        )

    repaired_path = output_dir / f"{pdf_path.stem}_repaired.pdf"
    logger.info(f"Repairing PDF: {pdf_path} → {repaired_path}")

    try:
        src = fitz.open(str(pdf_path))
        dst = fitz.open()

        for page in src:
            # insert_pdf copies pages with fully resolved attributes
            dst.insert_pdf(src, from_page=page.number, to_page=page.number)

        dst.save(str(repaired_path), garbage=4, deflate=True)
        dst.close()
        src.close()

        logger.info(
            f"PDF repaired successfully "
            f"({repaired_path.stat().st_size / 1024:.0f} KB)"
        )
        return repaired_path

    except Exception as e:
        raise RuntimeError(f"PDF repair failed: {e}") from e


def _get_ocr_options(ocr_engine: str, language: str = "es", force_ocr: bool = True):
    """
    Configure OCR options based on selected engine, language, and force_ocr setting.
    
    Args:
        ocr_engine: OCR engine name ('ocrmac', 'easyocr', 'tesseract', 'auto')
        language: Language code ('es', 'en', 'fr', 'de', 'pt', 'it')
        force_ocr: If True, enables force_full_page_ocr to bypass PDF text layer
    
    Returns:
        OCR options object configured for the specified engine, or None for auto
    """
    if ocr_engine == "auto":
        # Auto-detect the best OCR engine for the current platform
        recommended = get_recommended_ocr_engine()
        logger.info(f"Auto-detected OCR engine: {recommended}")
        return _get_ocr_options(recommended, language, force_ocr)
    
    elif ocr_engine == "ocrmac":
        # macOS native OCR - fastest and most efficient on Mac
        # Best choice for macOS systems, NOT available on Linux/Windows
        if platform.system() != "Darwin":
            logger.warning("ocrmac requested but not on macOS - falling back to easyocr")
            return _get_ocr_options("easyocr", language, force_ocr)
        
        from docling.datamodel.pipeline_options import OcrMacOptions
        lang_codes = LANGUAGE_MAP["ocrmac"].get(language, [f"{language}-{language.upper()}"])
        ocr_options = OcrMacOptions(
            force_full_page_ocr=force_ocr,
            lang=lang_codes
        )
        logger.debug(f"OcrMac configured with lang={lang_codes}, force_full_page_ocr={force_ocr}")
        return ocr_options
    
    elif ocr_engine == "tesseract":
        # Google's Tesseract OCR - good accuracy, widely supported
        from docling.datamodel.pipeline_options import TesseractOcrOptions
        lang_codes = LANGUAGE_MAP["tesseract"].get(language, [language])
        ocr_options = TesseractOcrOptions(
            force_full_page_ocr=force_ocr,
            lang=lang_codes
        )
        logger.debug(f"Tesseract configured with lang={lang_codes}, force_full_page_ocr={force_ocr}")
        return ocr_options
    
    elif ocr_engine == "easyocr":
        # Deep learning based - high accuracy but slower
        from docling.datamodel.pipeline_options import EasyOcrOptions
        lang_codes = LANGUAGE_MAP["easyocr"].get(language, [language])
        ocr_options = EasyOcrOptions(
            force_full_page_ocr=force_ocr,
            lang=lang_codes
        )
        logger.debug(f"EasyOCR configured with lang={lang_codes}, force_full_page_ocr={force_ocr}")
        return ocr_options
    
    else:
        # For unknown engines, use default
        logger.warning(f"Unknown OCR engine '{ocr_engine}', using default options")
        return None


def get_recommended_ocr_engine() -> str:
    """
    Get the recommended OCR engine for the current platform.
    
    Returns:
        str: 'ocrmac' on macOS, 'easyocr' on other platforms
    """
    if platform.system() == "Darwin":
        return "ocrmac"
    else:
        return "easyocr"


def normalize_heading_levels(text: str) -> str:
    """Normalize markdown heading levels based on content patterns.

    Corrects heading levels for chapter files where PDF-to-markdown conversion
    (e.g. Docling) flattened all headings to the same level (typically ``##``).
    Uses numbered prefixes and known patterns to infer the correct hierarchy:

    - ``## Módulo N`` / ``Module N`` / ``Chapter N``  →  ``#``  (module title)
    - ``## N.M Title`` (two-level number)              →  ``##`` (submodule)
    - ``## N.M.K Title`` (three-level number)           →  ``###`` (section)
    - Unnumbered ``##`` headings are left at level 2 by default.

    This function is idempotent: running it on already-normalised text is safe.

    Args:
        text: Markdown text (typically a single chapter/module file).

    Returns:
        Markdown text with corrected heading levels.
    """
    import re

    _module_header = re.compile(
        r"^(#{1,3})\s+"
        r"(?:Módulo|Modulo|Module|Chapter|Capítulo|Capitulo)"
        r"\s+\d+",
        re.IGNORECASE,
    )
    _section_3lvl = re.compile(r"^(#{1,3})\s+(\d+\.\d+\.\d+)\b")
    _submodule_2lvl = re.compile(r"^(#{1,3})\s+(\d+\.\d+)\b")

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        if _module_header.match(line):
            result.append(re.sub(r"^#{1,3}", "#", line))
            continue

        if _section_3lvl.match(line):
            result.append(re.sub(r"^#{1,3}", "###", line))
            continue

        if _submodule_2lvl.match(line):
            result.append(re.sub(r"^#{1,3}", "##", line))
            continue

        result.append(line)

    return "\n".join(result)


def split_markdown_by_chapters(
    markdown_path: str | Path,
    output_dir: str | Path | None = None,
    chapter_pattern: str | None = None,
) -> list[Path]:
    """
    Split a single markdown file into per-chapter/module files.

    Detects chapter boundaries using headers that match patterns like
    ``# Módulo N``, ``## Module N``, ``## Chapter N``, ``## Capítulo N``,
    or standalone ``## N.`` / ``## N `` numbering.  Content before the
    first chapter goes into ``00_frontmatter.md``; the references section
    (if present) goes into a separate ``references.md``.

    Args:
        markdown_path: Path to the source markdown file.
        output_dir: Directory to write chapter files into.  Defaults to a
                    ``chapters/`` subdirectory next to the source file.
        chapter_pattern: Regex that matches chapter-start lines.  The
                        default handles many common patterns
                        (case-insensitive, H1 or H2).

    Returns:
        List of paths to the created chapter files.
    """
    import re
    import unicodedata

    markdown_path = Path(markdown_path).resolve()
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    if output_dir is not None:
        out = Path(output_dir).resolve()
    else:
        out = markdown_path.parent / "chapters"
    out.mkdir(parents=True, exist_ok=True)

    # Default pattern: H1/H2 with Módulo|Module|Chapter|Capítulo + number
    if chapter_pattern is None:
        chapter_pattern = (
            r"^#{1,2}\s+"
            r"(?:Módulo|Modulo|Module|Chapter|Capítulo|Capitulo)"
            r"\s+\d+"
        )

    content = markdown_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Locate chapter boundaries and references
    boundaries: list[tuple[int, str]] = []
    ref_pattern = r"^#{1,2}\s+[Rr]eferencias?"
    for idx, line in enumerate(lines):
        if re.match(chapter_pattern, line, re.IGNORECASE):
            boundaries.append((idx, line.strip()))
        elif re.match(ref_pattern, line, re.IGNORECASE):
            boundaries.append((idx, "Referencias"))

    if not boundaries:
        logger.warning(
            "No chapter boundaries found with default pattern. "
            "Trying fallback: any H1 header as chapter boundary."
        )
        for idx, line in enumerate(lines):
            if re.match(r"^#\s+\S", line) and not re.match(r"^##", line):
                boundaries.append((idx, line.strip()))

    if not boundaries:
        logger.warning("No chapter boundaries found — writing entire file as single chapter")
        dest = out / "01_full.md"
        dest.write_text(content, encoding="utf-8")
        return [dest]

    # Collect artifacts dir (images) so we can fix relative paths
    artifacts_dir = markdown_path.parent / f"{markdown_path.stem}_artifacts"

    created: list[Path] = []

    def _fix_image_paths(text: str, target_file: Path) -> str:
        """Rewrite absolute/relative image paths so they work from target_file."""
        if not artifacts_dir.exists():
            return text

        def _replacer(m: re.Match) -> str:
            alt = m.group(1)
            img_path = Path(m.group(2))
            try:
                rel = os.path.relpath(img_path, target_file.parent)
            except ValueError:
                rel = str(img_path)
            return f"![{alt}]({rel})"

        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replacer, text)

    def _slugify(text: str, max_len: int = 50) -> str:
        # Strip numbered prefixes like "1.1", "2.3.1" before slugifying
        text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii").lower()
        text = re.sub(r"[^\w\s-]", "", text).strip()
        text = re.sub(r"[\s]+", "_", text)
        return text[:max_len].rstrip("_")

    def _write_chunk(filename: str, chunk_lines: list[str], normalize: bool = False) -> Path:
        dest = out / filename
        text = "\n".join(chunk_lines)
        if normalize:
            text = normalize_heading_levels(text)
        text = _fix_image_paths(text, dest)
        dest.write_text(text, encoding="utf-8")
        logger.info(f"  → {dest.name}  ({len(chunk_lines)} lines)")
        return dest

    # Frontmatter (everything before first chapter)
    first_chapter_idx = boundaries[0][0]
    if first_chapter_idx > 0:
        frontmatter = lines[:first_chapter_idx]
        if any(l.strip() for l in frontmatter):
            created.append(_write_chunk("00_frontmatter.md", frontmatter))

    # Chapters
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        chunk = lines[start:end]

        is_references = "referencia" in title.lower()
        if is_references:
            fname = "references.md"
        else:
            num_match = re.search(r"\d+", title)
            num = num_match.group() if num_match else str(i + 1)
            raw_title = re.sub(r"^#+\s*", "", title)
            raw_title = re.sub(
                r"^(?:Módulo|Modulo|Module|Chapter|Capítulo|Capitulo)\s+\d+\s*[:\-–—.]?\s*",
                "", raw_title, flags=re.IGNORECASE,
            )
            raw_title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", raw_title)
            slug = _slugify(raw_title) if raw_title.strip() else ""
            fname = f"{int(num):02d}_{slug}.md" if slug else f"{int(num):02d}_module.md"

        created.append(_write_chunk(fname, chunk, normalize=not is_references))

    logger.info(f"Split into {len(created)} files in {out}")
    return created


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDF to Markdown (optionally split by chapters)")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--language", "-l", default="es", help="OCR language (default: es)")
    parser.add_argument("--split", action="store_true", help="Split output into per-chapter files")
    parser.add_argument("--no-ocr", action="store_true", help="Disable force OCR")
    args = parser.parse_args()

    print(f"\n=== Converting {args.pdf_path} ===")
    result_path = convert_pdf_to_markdown(
        args.pdf_path,
        ocr_engine="auto",
        language=args.language,
        force_ocr=not args.no_ocr,
        images_scale=2.0,
        table_mode="accurate",
        table_text_handling="hybrid",
        output_dir=args.output_dir,
    )
    print(f"Saved markdown to: {result_path}")

    if args.split:
        print("\n=== Splitting into chapters ===")
        chapter_dir = Path(result_path).parent / "chapters"
        files = split_markdown_by_chapters(result_path, output_dir=chapter_dir)
        print(f"Created {len(files)} chapter files:")
        for f in files:
            print(f"  {f}")
