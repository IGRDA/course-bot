"""
Unified LLM-assisted structure validator for digitalized content.

After the markdown parser builds an initial CourseState, this module
checks whether the resulting module/submodule tree has a healthy shape.
When pathological structures are detected it extracts compact structural
signals — submodule titles for well-segmented content, or chunk
fingerprints for oversized blobs — and sends them to an LLM which
organises the content into proper course modules and submodules.

The full theory text is never sent to the LLM.  Only compact metadata
(titles, first lines, heading hints, word counts) is used, keeping
the prompt small (~4-8 KB even for 300-page documents).
"""

import json
import logging
import os
import re
import time
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from workflows.state import CourseState, Module, Submodule, Section
from LLMs.text2text import create_text_llm, resolve_text_model_name

logger = logging.getLogger(__name__)

_FALLBACK_PROVIDERS = ["groq", "openai"]

_MAX_SUBMODULES_PER_MODULE = 15
_MAX_SUBMODULES_HARD = 20
_OVERSIZED_SUBMODULE_WORDS = 10_000
_OVERSIZED_MODULE_WORDS = 15_000
_TARGET_MIN_MODULES = 3
_TARGET_MAX_MODULES = 15
_CHUNK_TARGET_WORDS = 1500

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _is_anomalous(modules: list[Module]) -> bool:
    """Return True when the parsed structure needs LLM-assisted restructuring."""
    if not modules:
        return False

    if len(modules) == 1 and len(modules[0].submodules) > _MAX_SUBMODULES_PER_MODULE:
        return True

    if any(len(m.submodules) > _MAX_SUBMODULES_HARD for m in modules):
        return True

    for m in modules:
        for sm in m.submodules:
            words = sum(len(sec.theory.split()) for sec in sm.sections)
            if words > _OVERSIZED_SUBMODULE_WORDS:
                return True

    if len(modules) == 1:
        total_words = sum(
            len(sec.theory.split())
            for m in modules for sm in m.submodules for sec in sm.sections
        )
        if total_words > _OVERSIZED_MODULE_WORDS:
            return True

    if len(modules) == 1:
        total_sections = sum(
            len(sm.sections) for m in modules for sm in m.submodules
        )
        if total_sections > 30:
            return True

    return False


# ---------------------------------------------------------------------------
# Smart paragraph chunking
# ---------------------------------------------------------------------------

def _chunk_by_paragraphs(text: str, target_words: int = _CHUNK_TARGET_WORDS) -> list[str]:
    """Split text into chunks of ~target_words at paragraph boundaries.

    Lossless: ``"".join(result) == text`` always holds.
    """
    breaks = [m.end() for m in re.finditer(r'\n[ \t]*\n', text)]

    if not breaks:
        return [text] if text.strip() else []

    chunks: list[str] = []
    chunk_start = 0
    prev_break = 0
    current_words = 0

    for brk in breaks:
        segment_words = len(text[prev_break:brk].split())

        if current_words + segment_words > target_words and current_words > 0:
            chunks.append(text[chunk_start:prev_break])
            chunk_start = prev_break
            current_words = segment_words
        else:
            current_words += segment_words

        prev_break = brk

    remaining_words = len(text[prev_break:].split())
    if current_words + remaining_words > target_words and current_words > 0:
        chunks.append(text[chunk_start:prev_break])
        chunks.append(text[prev_break:])
    else:
        chunks.append(text[chunk_start:])

    return chunks


# ---------------------------------------------------------------------------
# Fingerprint extraction
# ---------------------------------------------------------------------------

_HEADING_HINT_RE = re.compile(
    r'^(?:'
    r'[A-ZÁÉÍÓÚÑ\s]{8,}'           # ALL CAPS lines (8+ chars)
    r'|(?:Artículo|Art\.?|ITC|CAPÍTULO|Capítulo|Chapter|Módulo|Tema|TEMA|Sección|Section)\s*\S+'
    r'|\d+(?:\.\d+)*\s*[.\-–—]\s*.+'  # Numbered sections like "1.1.- Algo"
    r')$',
    re.MULTILINE,
)


def _extract_fingerprint(chunk: str) -> dict:
    """Extract compact structural signals from a text chunk."""
    lines = chunk.strip().split('\n')
    first_line = lines[0][:150] if lines else ""

    hints = _HEADING_HINT_RE.findall(chunk)
    heading_hints = [h.strip()[:100] for h in hints[:5]]

    return {
        "first_line": first_line,
        "heading_hints": heading_hints,
        "word_count": len(chunk.split()),
    }


# ---------------------------------------------------------------------------
# Content unit expansion
# ---------------------------------------------------------------------------

def _expand_to_content_units(
    modules: list[Module],
) -> tuple[list[dict], dict]:
    """Convert modules into a flat list of content units with structural signals.

    Each submodule becomes either:
    - A single "heading" unit (if small enough), using the submodule title
    - Multiple "chunk" units (if oversized), using fingerprints

    Returns (units, origin_map) where origin_map maps unit index ->
    either {"type": "submodule", "ref": Submodule} or {"type": "chunk", "text": str}.
    """
    units: list[dict] = []
    origin_map: dict[int, dict] = {}
    global_idx = 0

    for module in modules:
        for sm in module.submodules:
            total_words = sum(len(sec.theory.split()) for sec in sm.sections)

            if total_words < _OVERSIZED_SUBMODULE_WORDS:
                global_idx += 1
                unit: dict = {
                    "idx": global_idx,
                    "source": "heading",
                    "title": sm.title,
                    "word_count": total_words,
                }
                section_titles = [sec.title for sec in sm.sections if sec.title]
                if 1 < len(section_titles) <= 6:
                    unit["section_titles"] = section_titles
                units.append(unit)
                origin_map[global_idx] = {"type": "submodule", "ref": sm}
            else:
                all_theory = "\n\n".join(
                    sec.theory for sec in sm.sections if sec.theory
                )
                chunks = _chunk_by_paragraphs(all_theory)
                for chunk_text in chunks:
                    global_idx += 1
                    fp = _extract_fingerprint(chunk_text)
                    unit = {
                        "idx": global_idx,
                        "source": "chunk",
                        **fp,
                    }
                    units.append(unit)
                    origin_map[global_idx] = {"type": "chunk", "text": chunk_text}

    return units, origin_map


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_STRUCTURE_SYSTEM = """\
You are a course-structure expert.  You receive a list of content units \
extracted from a document.  Each unit is either a titled heading (from an \
already-segmented section) or a text chunk fingerprint (first line and \
structural hints from a large block of text).

Your job is to organise these units into a logical course structure with \
modules and submodules.

Rules:
1. Create between {min_modules} and {max_modules} modules.
2. Each module must have between 1 and 10 submodules.
3. Every unit index must appear in exactly one submodule.
4. Groups must be **contiguous** — each submodule's unit indices must be \
consecutive, and modules must follow reading order.
5. For "heading" units: prefer grouping one heading = one submodule \
(unless headings are very small and thematically related).
6. For "chunk" units: group 2-5 thematically related chunks into each \
submodule.
7. Use the titles, first_lines, heading_hints, and section_titles to \
judge topic boundaries.
8. Write all titles and descriptions in the SAME LANGUAGE as the content.
9. Return ONLY a valid JSON array — no markdown fences, no commentary."""

_STRUCTURE_USER = """\
Course title: {course_title}
Total content units: {total_units}
Approximate total words: {total_words}

Organise these content units into {min_modules}-{max_modules} course modules, \
each with well-structured submodules.

Content units:
```json
{units_json}
```

Return a JSON array:
[
  {{
    "title": "Descriptive module title",
    "description": "1-2 sentence module description",
    "submodules": [
      {{
        "title": "Submodule title",
        "unit_indices": [1, 2, 3]
      }}
    ]
  }}
]

Every index from 1 to {total_units} must appear exactly once.  \
All index ranges must be contiguous."""

_structure_prompt = ChatPromptTemplate.from_messages([
    ("system", _STRUCTURE_SYSTEM),
    ("human", _STRUCTURE_USER),
])


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _robust_json_loads(text: str) -> list | dict:
    text = _strip_markdown_fences(text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(
        r'[\x00-\x1f\x7f]',
        lambda m: ' ' if m.group() in ('\n', '\r', '\t') else '',
        text,
    )
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
            return json.loads(fixed)

    raise json.JSONDecodeError("Could not extract valid JSON array", text, 0)


# ---------------------------------------------------------------------------
# LLM call with retries / fallbacks
# ---------------------------------------------------------------------------

def _llm_structure_analysis(
    units: list[dict],
    course_title: str,
    provider: str = "mistral",
    max_retries: int = 3,
    fallback_providers: list[str] | None = None,
) -> list[dict]:
    """Ask an LLM to organise content units into modules and submodules."""
    if fallback_providers is None:
        fallback_providers = list(_FALLBACK_PROVIDERS)

    units_json = json.dumps(units, ensure_ascii=False, indent=2)
    total_units = len(units)
    total_words = sum(u.get("word_count", 0) for u in units)

    providers_to_try = [provider] + [p for p in fallback_providers if p != provider]
    result = None

    for prov in providers_to_try:
        model_name = resolve_text_model_name(prov)
        if not model_name and not os.getenv(f"{prov.upper()}_API_KEY", ""):
            print(f"      [{prov}] No API key, skipping...")
            continue

        llm_kwargs: dict = {"temperature": 0.1}
        if model_name:
            llm_kwargs["model_name"] = model_name

        try:
            llm = create_text_llm(provider=prov, **llm_kwargs)
        except Exception as e:
            logger.warning("Could not create LLM for provider %s: %s", prov, e)
            continue

        chain = _structure_prompt | llm | StrOutputParser()

        for attempt in range(max_retries):
            t0 = time.time()
            print(f"      [{prov}] structure analysis attempt {attempt + 1}/{max_retries} ...")
            try:
                raw = chain.invoke({
                    "course_title": course_title,
                    "total_units": total_units,
                    "total_words": total_words,
                    "units_json": units_json,
                    "min_modules": _TARGET_MIN_MODULES,
                    "max_modules": _TARGET_MAX_MODULES,
                })
                elapsed = time.time() - t0
                parsed = _robust_json_loads(raw)
                if not isinstance(parsed, list) or not parsed:
                    raise ValueError("LLM returned non-list or empty result")

                _validate_structure_response(parsed, total_units)
                result = parsed
                total_mods = len(result)
                total_subs = sum(len(m.get("submodules", [])) for m in result)
                print(
                    f"      [{prov}] attempt {attempt + 1} succeeded ({elapsed:.1f}s) "
                    f"-> {total_mods} modules, {total_subs} submodules"
                )
                break

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                elapsed = time.time() - t0
                logger.warning(
                    "Structure analysis attempt %d/%d with %s failed (%.1fs): %s",
                    attempt + 1, max_retries, prov, elapsed, e,
                )
            except Exception as e:
                elapsed = time.time() - t0
                logger.warning(
                    "Structure analysis attempt %d/%d with %s failed (%.1fs): %s",
                    attempt + 1, max_retries, prov, elapsed, e,
                )

        if result is not None:
            break
        print(f"      All attempts with {prov} failed, trying next provider...")

    if result is None:
        raise RuntimeError(
            f"All structure-validation attempts failed across providers {providers_to_try}"
        )

    return result


def _validate_structure_response(modules: list[dict], total_units: int) -> None:
    """Raise ValueError if the LLM structure response is malformed."""
    seen: set[int] = set()
    prev_max = 0

    for mod in modules:
        if not mod.get("title"):
            raise ValueError("Module missing title")
        submodules = mod.get("submodules", [])
        if not submodules:
            raise ValueError(f"Module '{mod['title']}' has no submodules")

        for sm in submodules:
            indices = sm.get("unit_indices", [])
            if not indices:
                raise ValueError(f"Submodule '{sm.get('title', '?')}' has no unit_indices")

            for idx in indices:
                if not isinstance(idx, int):
                    raise ValueError(f"Non-integer index: {idx}")
                if idx < 1 or idx > total_units:
                    raise ValueError(f"Index {idx} out of range [1, {total_units}]")
                if idx in seen:
                    raise ValueError(f"Duplicate index: {idx}")
                seen.add(idx)

            if min(indices) <= prev_max:
                raise ValueError(
                    f"Non-contiguous: submodule '{sm.get('title')}' "
                    f"starts at {min(indices)} but previous ended at {prev_max}"
                )
            if sorted(indices) != list(range(min(indices), max(indices) + 1)):
                raise ValueError(
                    f"Non-contiguous indices in submodule '{sm.get('title')}': {indices}"
                )
            prev_max = max(indices)

    if seen != set(range(1, total_units + 1)):
        missing = set(range(1, total_units + 1)) - seen
        raise ValueError(f"Missing unit indices: {missing}")


# ---------------------------------------------------------------------------
# Apply structure
# ---------------------------------------------------------------------------

def _apply_structure(
    structure: list[dict],
    units: list[dict],
    origin_map: dict[int, dict],
) -> list[Module]:
    """Build Module/Submodule/Section objects from the LLM's structure."""
    new_modules: list[Module] = []

    for mod_idx, mod_spec in enumerate(structure, 1):
        new_submodules: list[Submodule] = []

        for sub_idx, sub_spec in enumerate(mod_spec.get("submodules", []), 1):
            indices = sub_spec["unit_indices"]
            origins = [origin_map[i] for i in indices]

            if all(o["type"] == "submodule" for o in origins) and len(origins) == 1:
                sm = origins[0]["ref"]
                sm.index = sub_idx
                if sub_spec.get("title"):
                    sm.title = sub_spec["title"]
                new_submodules.append(sm)
            elif all(o["type"] == "submodule" for o in origins):
                merged_sections: list[Section] = []
                sec_idx = 0
                for o in origins:
                    for sec in o["ref"].sections:
                        sec_idx += 1
                        sec.index = sec_idx
                        merged_sections.append(sec)
                new_submodules.append(Submodule(
                    title=sub_spec.get("title", f"Submodule {sub_idx}"),
                    index=sub_idx,
                    description="",
                    sections=merged_sections,
                ))
            else:
                chunk_texts = []
                for o in origins:
                    if o["type"] == "chunk":
                        chunk_texts.append(o["text"])
                    elif o["type"] == "submodule":
                        chunk_texts.append("\n\n".join(
                            sec.theory for sec in o["ref"].sections if sec.theory
                        ))

                combined_theory = "".join(chunk_texts)
                new_submodules.append(Submodule(
                    title=sub_spec.get("title", f"Submodule {sub_idx}"),
                    index=sub_idx,
                    description="",
                    sections=[Section(
                        title=sub_spec.get("title", f"Section {sub_idx}"),
                        index=1,
                        theory=combined_theory,
                    )],
                ))

        new_modules.append(Module(
            title=mod_spec.get("title", f"Module {mod_idx}"),
            description=mod_spec.get("description", ""),
            id=str(mod_idx),
            index=mod_idx,
            submodules=new_submodules,
        ))

    return new_modules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_split(
    state: CourseState,
    provider: str | None = None,
    max_retries: int = 3,
) -> CourseState:
    """Validate parsed course structure and restructure if pathological.

    Uses a unified pipeline that adapts to whatever structural signals
    are available: submodule titles when the content is well-segmented,
    chunk fingerprints when the content is a flat blob, or a mix of both.
    """
    provider = provider or state.config.text_llm_provider
    modules = state.modules

    if not _is_anomalous(modules):
        total_subs = sum(len(m.submodules) for m in modules)
        print(f"   Structure OK ({len(modules)} modules, {total_subs} total submodules) — no re-split needed.")
        return state

    total_subs = sum(len(m.submodules) for m in modules)
    total_sections = sum(len(sm.sections) for m in modules for sm in m.submodules)
    total_words = sum(
        len(sec.theory.split())
        for m in modules for sm in m.submodules for sec in sm.sections
    )
    total_chars = sum(
        len(sec.theory)
        for m in modules for sm in m.submodules for sec in sm.sections
    )

    print(
        f"   Structure anomaly detected: {len(modules)} module(s), "
        f"{total_subs} submodules, {total_sections} sections, ~{total_words:,} words"
    )

    units, origin_map = _expand_to_content_units(modules)
    heading_count = sum(1 for u in units if u["source"] == "heading")
    chunk_count = sum(1 for u in units if u["source"] == "chunk")
    print(f"   Expanded to {len(units)} content units ({heading_count} headings, {chunk_count} chunks)")
    print(f"   Asking LLM to organise into modules and submodules...")

    course_title = state.title or state.config.title or "Untitled"

    structure = _llm_structure_analysis(
        units=units,
        course_title=course_title,
        provider=provider,
        max_retries=max_retries,
    )

    new_modules = _apply_structure(structure, units, origin_map)

    after_chars = sum(
        len(sec.theory)
        for m in new_modules for sm in m.submodules for sec in sm.sections
    )
    if after_chars != total_chars:
        logger.error(
            "Theory text length mismatch after restructuring: %d -> %d (delta %d chars)",
            total_chars, after_chars, total_chars - after_chars,
        )

    state.modules = new_modules

    print(f"   Restructuring complete: {len(new_modules)} modules")
    for m in new_modules:
        secs = sum(len(sm.sections) for sm in m.submodules)
        print(f"      Module {m.index}: {m.title[:70]} ({len(m.submodules)} submodules, {secs} sections)")

    return state
