"""
Markdown folder parser for content digitalization.

Reads a local folder of .md files (one per module) and builds a CourseState
with the full heading hierarchy, theory text, and local image references.

Supports two kinds of input:

1. **Well-structured** markdown with explicit heading levels::

       # Module Title
       ## Submodule Title
       ### Section Title
       Body text ...

2. **Flat** markdown (e.g. from Docling PDF conversion) where all headings
   share the same level.  Numbered prefixes (``1.1``, ``1.1.1``) are used
   to reconstruct the hierarchy automatically.
"""

import re
import logging
from pathlib import Path

from main.state import CourseState, CourseConfig, Module, Submodule, Section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

_OBJECTIVES_RE = re.compile(
    r"[▶►]?\s*(?:"
    r"(?:Course\s+)?Learning\s+Objectives"
    r"|Module\s+Objectives"
    r"|Objetivos?\s+(?:del?\s+)?(?:módulo|curso|aprendizaje)"
    r")",
    re.IGNORECASE,
)

_GENERIC_MODULE_TITLE_RE = re.compile(
    r"^(?:Módulo|Modulo|Module|Chapter|Capítulo|Capitulo)\s+\d+\s*$",
    re.IGNORECASE,
)

_OCR_JUNK_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"<!-- formula-not-decoded -->"
    r"|UNDERWEIGHT"
    r"|©\s*\d{4}\s+\w+\.?\s*(?:Todos\s+los\s+derechos\s+reservados|All\s+rights\s+reserved)\.?"
    r"|(?:ATHLETIC|Infection|HIGH\s+FATALITY\s+RATE|HEALTHCARE\s+SYSTEM\s+CAPACITY"
    r"|EXIT\s+OF\s+WORMHOLE|ENTER\s+WORMHOLE)\b[^\n]*"
    r")\s*(?:\n|$)",
    re.IGNORECASE,
)

_STRAY_SINGLE_CHAR_RE = re.compile(r"(?<=\n)\s*[A-Z]\s*(?=\n)")

_NUMBERED_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)+\.?\s+")

_MALFORMED_HEADING_RE = re.compile(
    r"^[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{5,}$"
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_PRECEDING_SNIPPET_LEN = 200

_MIN_SECTION_WORDS = 100


def _strip_numbered_prefix(title: str) -> str:
    """Remove leading numbered prefixes like ``1.1``, ``2.3.1`` from a title."""
    return _NUMBERED_PREFIX_RE.sub("", title).strip()


def _is_malformed_heading(title: str) -> bool:
    """Return True if the title looks like a formula or garbled OCR, not real text."""
    stripped = title.strip()
    if not stripped:
        return True
    if _MALFORMED_HEADING_RE.match(stripped):
        return True
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 10 and alpha_chars / len(stripped) < 0.3:
        return True
    return False


def _derive_image_description(alt: str, preceding_text: str) -> str:
    """Generate a descriptive image query from alt text and preceding context.

    If *alt* is generic (e.g. ``"Image"``), extracts a short description from
    the last ~60 chars of *preceding_text*.
    """
    if alt and alt.lower() not in ("image", "img", "figure", "imagen", "figura", ""):
        return alt
    if not preceding_text:
        return alt or "Image"
    snippet = preceding_text.strip()[-120:]
    words = snippet.split()
    desc_words = words[-10:] if len(words) > 10 else words
    desc = " ".join(desc_words).strip(" .,;:-")
    return desc if desc else (alt or "Image")


def extract_images_with_context(text: str) -> list[dict]:
    """Extract images with the text that precedes them for inline placement.

    Returns a list of dicts::

        [{"alt": str, "path": str, "preceding_text": str}, ...]

    ``preceding_text`` is the last ~200 chars before the image marker,
    stripped of other image syntax.  The injection node uses this to find the
    ParagraphBlock whose content best matches, so each image is rendered
    next to the text that references it.
    """
    results = []
    for m in _IMAGE_RE.finditer(text):
        preceding_raw = text[: m.start()]
        preceding_clean = _IMAGE_RE.sub("", preceding_raw).strip()
        snippet = preceding_clean[-_PRECEDING_SNIPPET_LEN:]
        raw_alt = m.group(1)
        results.append({
            "alt": _derive_image_description(raw_alt, snippet),
            "path": m.group(2),
            "preceding_text": snippet,
        })
    return results


def strip_images(text: str) -> str:
    """Remove ``![alt](path)`` markdown image syntax, leaving the body text clean."""
    return _IMAGE_RE.sub("", text)


def _resolve_image_paths(images: list[dict], md_file: Path, source_folder: Path) -> list[dict]:
    """Resolve relative image paths to absolute paths based on the markdown file location."""
    resolved = []
    for img in images:
        raw_path = img["path"]
        candidate = (md_file.parent / raw_path).resolve()
        if not candidate.exists():
            candidate = (source_folder / raw_path).resolve()
        resolved.append({
            "alt": img["alt"],
            "path": str(candidate),
            "preceding_text": img.get("preceding_text", ""),
        })
    return resolved


# ---------------------------------------------------------------------------
# Heading normalisation (fallback for flat markdown)
# ---------------------------------------------------------------------------

def _normalize_headings_if_needed(md_text: str) -> str:
    """Apply heading-level correction when headings appear to be flattened.

    When all headings use the same markdown level (e.g. all ``##``), this
    detects numbered prefixes (``X.Y``, ``X.Y.Z``) and module keywords to
    infer the correct hierarchy.  Already-correct markdown is returned as-is.
    """
    headings = list(_HEADING_RE.finditer(md_text))
    if not headings:
        return md_text

    levels = {len(h.group(1)) for h in headings}
    if len(levels) > 1:
        return md_text

    _mod_re = re.compile(
        r"^(#{1,3})\s+(?:Módulo|Modulo|Module|Chapter|Capítulo|Capitulo)\s+\d+",
        re.IGNORECASE,
    )
    _sec3_re = re.compile(r"^(#{1,3})\s+\d+\.\d+\.\d+\b")
    _sub2_re = re.compile(r"^(#{1,3})\s+\d+\.\d+\b")

    lines = md_text.split("\n")
    result: list[str] = []
    changed = False

    for line in lines:
        if _mod_re.match(line):
            result.append(re.sub(r"^#{1,3}", "#", line))
            changed = True
        elif _sec3_re.match(line):
            result.append(re.sub(r"^#{1,3}", "###", line))
            changed = True
        elif _sub2_re.match(line):
            result.append(re.sub(r"^#{1,3}", "##", line))
            changed = True
        else:
            result.append(line)

    return "\n".join(result) if changed else md_text


# ---------------------------------------------------------------------------
# Heading tree builder
# ---------------------------------------------------------------------------

def _split_by_headings(md_text: str) -> list[dict]:
    """Split markdown text into a flat list of heading blocks.

    Returns a list of dicts::

        [
            {"level": 1, "title": "Module Title", "body": "..."},
            {"level": 2, "title": "Submodule Title", "body": "..."},
            {"level": 3, "title": "Section Title", "body": "..."},
            ...
        ]

    Text before the first heading is attached to a synthetic level-0 "preamble" block.
    """
    headings = list(_HEADING_RE.finditer(md_text))
    if not headings:
        return [{"level": 0, "title": "", "body": md_text.strip()}]

    blocks: list[dict] = []

    preamble = md_text[: headings[0].start()].strip()
    if preamble:
        blocks.append({"level": 0, "title": "", "body": preamble})

    for i, match in enumerate(headings):
        level = len(match.group(1))
        title = _strip_numbered_prefix(match.group(2).strip())
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(md_text)
        body = md_text[start:end].strip()

        if _is_malformed_heading(title):
            if blocks:
                blocks[-1]["body"] += "\n\n" + body
                logger.info("Merged malformed heading into previous block: %s", title[:60])
            continue

        blocks.append({"level": level, "title": title, "body": body})

    return blocks


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

def _is_objectives_section(title: str) -> bool:
    """Return True if *title* matches a learning-objectives heading pattern."""
    return bool(_OBJECTIVES_RE.search(title))


def _is_generic_module_title(title: str) -> bool:
    """Return True if *title* is just ``Módulo N`` / ``Module N`` without a description."""
    return bool(_GENERIC_MODULE_TITLE_RE.match(title.strip()))


def _extract_title_and_description(preamble: str) -> tuple[str, str]:
    """Extract a descriptive title and description from module preamble text.

    The preamble is the body text after ``# Módulo N``. Convention:
    - First non-empty paragraph  → descriptive title
    - Second non-empty paragraph → module description
    """
    paragraphs = [p.strip() for p in preamble.split("\n\n") if p.strip()]
    title = paragraphs[0] if paragraphs else ""
    description = paragraphs[1] if len(paragraphs) > 1 else ""
    return title, description


def _clean_ocr_artifacts(text: str) -> str:
    """Strip common OCR noise: copyright notices, formula placeholders, stray chars."""
    text = _OCR_JUNK_RE.sub("\n", text)
    text = _STRAY_SINGLE_CHAR_RE.sub("", text)
    # Fix missing word boundaries from OCR (e.g. "ThomasYoung" -> "Thomas Young")
    text = re.sub(r"([a-záéíóúñü])([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_frontmatter_file(filepath: Path) -> bool:
    """Return True if *filepath* looks like a cover / TOC file, not course content."""
    stem = filepath.stem.lower()
    return stem in ("00_frontmatter", "frontmatter", "00_front", "00_cover")


# ---------------------------------------------------------------------------
# Module file parser
# ---------------------------------------------------------------------------

def parse_module_file(filepath: Path, source_folder: Path, module_index: int) -> Module:
    """Parse a single markdown file into a ``Module``.

    Heading convention (after normalisation):

    - ``#``   → Module title (first one wins; falls back to filename)
    - ``##``  → Submodule
    - ``###`` → Section (body becomes theory)

    If all headings share the same markdown level, the parser applies
    automatic heading normalisation based on numbered prefixes before
    building the tree.
    """
    md_text = filepath.read_text(encoding="utf-8")

    md_text = _normalize_headings_if_needed(md_text)

    blocks = _split_by_headings(md_text)

    module_title = _title_from_filename(filepath)
    module_description = ""
    submodules: list[Submodule] = []
    current_submodule_title = ""
    current_sections: list[Section] = []
    preamble_body = ""

    for block in blocks:
        level = block["level"]
        title = block["title"]
        body = block["body"]

        if level <= 1:
            if level == 1 and title:
                module_title = _strip_numbered_prefix(title)
            if body:
                preamble_body += ("\n\n" + body if preamble_body else body)
            continue

        if _is_objectives_section(title):
            logger.debug("Skipping objectives section: %s", title)
            continue

        if level == 2:
            if current_submodule_title or current_sections:
                submodules.append(_build_submodule(
                    current_submodule_title or "General",
                    current_sections,
                    len(submodules) + 1,
                ))
                current_sections = []
            current_submodule_title = title
            if body.strip():
                clean_body = _clean_ocr_artifacts(body)
                section = _body_to_section(title, clean_body, filepath, source_folder, len(current_sections) + 1)
                if section.theory.strip():
                    current_sections.append(section)
            continue

        if level == 3:
            clean_body = _clean_ocr_artifacts(body)
            section = _body_to_section(title, clean_body, filepath, source_folder, len(current_sections) + 1)
            current_sections.append(section)
            continue

    # Use preamble for smarter title / description when the heading is generic
    if preamble_body.strip():
        extracted_title, extracted_desc = _extract_title_and_description(preamble_body)
        if extracted_title and _is_generic_module_title(module_title):
            module_title = extracted_title
        if extracted_desc:
            module_description = extracted_desc

    if current_submodule_title or current_sections:
        submodules.append(_build_submodule(
            current_submodule_title or "General",
            current_sections,
            len(submodules) + 1,
        ))

    if not submodules:
        submodules = [Submodule(title="Content", index=1, sections=[
            Section(title="Content", index=1, theory=md_text),
        ])]

    for sm in submodules:
        for sec in sm.sections:
            word_count = len(sec.theory.split())
            if word_count < _MIN_SECTION_WORDS:
                logger.warning(
                    "Short section (%d words): [%s] > [%s] — consider merging",
                    word_count, sm.title[:40], sec.title[:40],
                )

    return Module(
        title=module_title,
        description=module_description,
        index=module_index,
        id=str(module_index),
        submodules=submodules,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _title_from_filename(filepath: Path) -> str:
    """Derive a human-readable title from a filename like ``01_intro_to_physics.md``."""
    stem = filepath.stem
    stem = re.sub(r"^\d+[_\-.\s]*", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip().title() or filepath.stem


def _body_to_section(
    title: str,
    body: str,
    md_file: Path,
    source_folder: Path,
    index: int,
) -> Section:
    """Build a ``Section`` from a heading title and its body text."""
    images = extract_images_with_context(body)
    resolved_images = _resolve_image_paths(images, md_file, source_folder) if images else None
    clean_theory = strip_images(body).strip()

    return Section(
        title=title,
        index=index,
        theory=clean_theory,
        source_images=resolved_images,
    )


def _build_submodule(title: str, sections: list[Section], index: int) -> Submodule:
    """Build a ``Submodule`` from accumulated sections, re-indexing them."""
    for i, sec in enumerate(sections, 1):
        sec.index = i
    return Submodule(title=title, index=index, sections=sections)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_markdown_folder(
    folder_path: str,
    title: str = "",
    language: str = "English",
    **config_kwargs,
) -> CourseState:
    """Read all ``.md`` files from *folder_path* and build a ``CourseState``.

    Files are sorted alphabetically so numeric prefixes (``01_``, ``02_``) give
    deterministic ordering.  Files that look like frontmatter (cover pages,
    tables of contents) are automatically skipped.

    Parameters
    ----------
    folder_path:
        Path to the local folder containing one ``.md`` file per module.
    title:
        Course title.  If empty, derived from the folder name.
    language:
        Content language (passed through to ``CourseConfig``).
    **config_kwargs:
        Extra keyword arguments forwarded to ``CourseConfig``.
    """
    source = Path(folder_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Markdown source folder not found: {source}")

    md_files = sorted(source.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {source}")

    logger.info("Parsing %d markdown files from %s", len(md_files), source)

    modules: list[Module] = []
    content_index = 0
    for md_file in md_files:
        if _is_frontmatter_file(md_file):
            logger.info("  Skipping frontmatter: %s", md_file.name)
            continue
        content_index += 1
        logger.info("  Parsing module %d: %s", content_index, md_file.name)
        module = parse_module_file(md_file, source, module_index=content_index)
        modules.append(module)

    if not title:
        title = _title_from_filename(Path(folder_path))

    config = CourseConfig(
        title=title,
        language=language,
        md_source_path=str(source),
        **config_kwargs,
    )

    state = CourseState(
        config=config,
        title=title,
        modules=modules,
    )

    total_sections = sum(len(s.sections) for m in modules for s in m.submodules)
    logger.info(
        "Parsed %d modules, %d submodules, %d sections",
        len(modules),
        sum(len(m.submodules) for m in modules),
        total_sections,
    )

    return state
