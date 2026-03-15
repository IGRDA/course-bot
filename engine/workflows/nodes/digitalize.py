"""
Content digitalization nodes for parsing markdown and injecting local images.

Provides LangGraph nodes:
- parse_markdown_folder_node: reads a folder of .md files into CourseState
- restructure_parsed_content_node: LLM-assisted validation/improvement of parsed structure
- inject_local_images_node: places local images into ParagraphBlocks after HTML formatting
"""

import shutil
import logging
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from workflows.state import CourseState, ParagraphBlock
from .utils import get_output_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1: Parse markdown folder
# ---------------------------------------------------------------------------

def parse_markdown_folder_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Parse a local folder of markdown files into the course skeleton.

    Reads ``state.config.md_source_path``, builds modules/submodules/sections
    from heading hierarchy, and populates theory text and image references.
    No LLM calls -- purely structural parsing.
    """
    print("Parsing markdown folder into course structure...")

    from agents.md_digitalizer.parser import parse_markdown_folder

    source_path = state.config.md_source_path
    if not source_path:
        raise ValueError("md_source_path must be set in CourseConfig")

    parsed = parse_markdown_folder(
        folder_path=source_path,
        title=state.title or state.config.title,
        language=state.config.language,
    )

    state.modules = parsed.modules
    if parsed.title:
        state.title = parsed.title

    total_sections = sum(len(s.sections) for m in state.modules for s in m.submodules)
    total_images = sum(
        len(sec.source_images or [])
        for m in state.modules for sm in m.submodules for sec in sm.sections
    )
    print(f"   Modules: {len(state.modules)}")
    print(f"   Total sections: {total_sections}")
    print(f"   Total images found: {total_images}")

    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("parse_markdown", state)

    return state


# ---------------------------------------------------------------------------
# Node 2: LLM-assisted restructuring
# ---------------------------------------------------------------------------

def restructure_parsed_content_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Validate and improve the parsed module structure using an LLM.

    Generates descriptive titles when originals are generic, writes module
    descriptions, removes junk sections, and detects the dominant content
    language (updating ``state.config.language`` when auto-detected).
    """
    print("Restructuring parsed content with LLM...")

    from agents.md_digitalizer.restructurer import restructure_course

    state = restructure_course(state)

    total_sections = sum(len(s.sections) for m in state.modules for s in m.submodules)
    print(f"   After restructuring: {len(state.modules)} modules, {total_sections} sections")

    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("restructure", state)

    return state


def detect_language_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Lightweight language detection without LLM (used when restructuring is skipped)."""
    from agents.md_digitalizer.restructurer import detect_content_language

    detected = detect_content_language(state)
    if detected:
        print(f"   Auto-detected language: {detected}")
        state.config.language = detected

    return state


# ---------------------------------------------------------------------------
# Node 3: Inject local images into ParagraphBlocks
# ---------------------------------------------------------------------------

def _extract_text_from_element(element) -> str:
    """Recursively extract plain text from an HtmlElement."""
    if hasattr(element, "type") and hasattr(element, "content"):
        content = element.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif hasattr(item, "elements"):
                    for sub in item.elements:
                        parts.append(_extract_text_from_element(sub))
                elif hasattr(item, "content"):
                    parts.append(_extract_text_from_element(item))
            return " ".join(parts)
        if isinstance(content, dict):
            return " ".join(str(v) for v in content.values() if isinstance(v, str))
    return ""


def _extract_text_from_block(block: ParagraphBlock) -> str:
    """Get concatenated text from a ParagraphBlock's elements."""
    parts = [block.title]
    for el in block.elements:
        parts.append(_extract_text_from_element(el))
    return " ".join(parts)


def _text_overlap_score(needle: str, haystack: str) -> float:
    """Simple word-overlap score between two strings (0.0 to 1.0)."""
    if not needle or not haystack:
        return 0.0
    needle_words = set(needle.lower().split())
    haystack_words = set(haystack.lower().split())
    if not needle_words:
        return 0.0
    return len(needle_words & haystack_words) / len(needle_words)


def _collect_paragraph_blocks(section) -> list[tuple[int, int, ParagraphBlock]]:
    """Walk a section's html elements and return (element_idx, block_idx, block) tuples."""
    blocks = []
    if not section.html:
        return blocks
    for e_idx, element in enumerate(section.html):
        content = element.content
        if isinstance(content, list):
            for b_idx, item in enumerate(content):
                if isinstance(item, ParagraphBlock):
                    blocks.append((e_idx, b_idx, item))
                elif isinstance(item, dict) and "title" in item:
                    blocks.append((e_idx, b_idx, item))
    return blocks


def _block_has_image(block) -> bool:
    """Check if a ParagraphBlock already has a valid image (e.g. from internet search)."""
    if isinstance(block, ParagraphBlock):
        return bool(block.image and block.image.get("content"))
    if isinstance(block, dict):
        img = block.get("image")
        return bool(img and isinstance(img, dict) and img.get("content"))
    return False


def inject_local_images_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Place local images from ``source_images`` into ParagraphBlocks.

    Runs AFTER ``generate_images`` (internet search).  For each section that
    has ``source_images``, this node finds the ParagraphBlock whose text best
    matches the image's ``preceding_text`` and overwrites the block's image
    with the local file.  Blocks that were not matched to any local image
    keep whatever internet image they already have.

    If no HTML elements exist (e.g. podcast-only run) the node is a no-op.
    """
    if not state.config.generate_html_output and not state.config.generate_pdf:
        print("Image injection skipped (no HTML/PDF output requested)")
        return state

    print("Injecting local markdown images into course...")

    output_mgr = get_output_manager(config)
    images_dir = None
    if output_mgr:
        images_dir = Path(output_mgr.get_run_folder()) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

    injected = 0
    skipped = 0

    for module in state.modules:
        for submodule in module.submodules:
            for section in submodule.sections:
                source_images = section.source_images
                if not source_images or not section.html:
                    continue

                blocks = _collect_paragraph_blocks(section)
                if not blocks:
                    continue

                block_texts = [
                    _extract_text_from_block(b) if isinstance(b, ParagraphBlock)
                    else b.get("title", "")
                    for (_, _, b) in blocks
                ]

                assigned: set[int] = set()
                for img_info in source_images:
                    img_path_str = img_info["path"]
                    alt = img_info.get("alt", "")
                    preceding = img_info.get("preceding_text", "")
                    is_url = img_path_str.startswith(("http://", "https://"))

                    if is_url:
                        dest_path_str = img_path_str
                    else:
                        src_path = Path(img_path_str)
                        if not src_path.exists():
                            logger.warning("Local image not found, skipping: %s", src_path)
                            skipped += 1
                            continue
                        if images_dir:
                            dest_path = images_dir / src_path.name
                            if not dest_path.exists():
                                shutil.copy2(src_path, dest_path)
                        else:
                            dest_path = src_path
                        dest_path_str = str(dest_path)

                    query = alt
                    if not query or query.lower() in ("image", "img", "figure", "imagen", "figura"):
                        snippet = (preceding or "").strip()[-120:]
                        words = snippet.split()[-10:]
                        query = " ".join(words).strip(" .,;:-") or alt or "Image"

                    image_data = {
                        "type": "img",
                        "query": query,
                        "content": dest_path_str,
                    }

                    # Find best matching block by text overlap
                    best_idx = -1
                    best_score = -1.0
                    for idx, text in enumerate(block_texts):
                        if idx in assigned:
                            continue
                        score = _text_overlap_score(preceding, text)
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx < 0:
                        skipped += 1
                        continue

                    assigned.add(best_idx)
                    _, _, block = blocks[best_idx]
                    if isinstance(block, ParagraphBlock):
                        block.image = image_data
                    elif isinstance(block, dict):
                        block["image"] = image_data
                    injected += 1

    print(f"   Local images injected: {injected}, skipped: {skipped}")

    if output_mgr:
        output_mgr.save_step("inject_images", state)

    return state
