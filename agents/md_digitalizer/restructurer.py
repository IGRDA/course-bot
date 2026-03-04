"""
LLM-assisted module restructurer for digitalized content.

After the markdown parser builds an initial CourseState, the restructurer
validates and improves each module's structure:

- Generates rich, descriptive titles when originals are generic
- Produces module-level descriptions summarising content
- Detects and removes non-content sections (copyright, stray OCR)
- Merges sections that were incorrectly split
- Ensures language consistency

Processes one module at a time to keep LLM context small and costs low.
"""

import json
import re
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from main.state import CourseState, Module, Submodule, Section
from LLMs.text2text import create_text_llm, resolve_text_model_name

logger = logging.getLogger(__name__)

_NUMBERED_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)+\.?\s+")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_RESTRUCTURE_SYSTEM = """\
You are an expert course-structure editor. You receive a JSON skeleton of one \
course module (titles and short theory snippets) and return an improved version \
as JSON.

Rules:
1. **Preserve all real educational content** — never drop valid theory sections.
2. **Title improvement**: Replace generic or flat titles with engaging, \
descriptive titles. Use question-based or hook-style titles when possible \
(e.g. "¿Por qué falla la física clásica?" instead of "Limitaciones de la \
física clásica"). Keep the language matching the content.
3. **Strip numbered prefixes**: Remove leading numbers like "1.1", "2.3.1" \
from all titles. The hierarchy is already encoded in the structure.
4. **Description**: Write a 1–3 sentence module description summarising the \
key topics. Match the content language.
5. **Remove junk sections**: Set "keep" to false for sections whose theory is \
empty, contains only copyright notices, OCR artifacts, or table-of-contents \
listings. Also flag sections whose theory_snippet contains obviously corrupted \
content (unrelated English words in Spanish text, garbled formulas as prose, \
text like "ATHLETIC", "Infection", "EXIT OF WORMHOLE").
6. **Merge tiny sections**: If a section has fewer than 100 words of theory \
(check theory_length) and the previous section covers the same topic, set \
"keep" to "merge_with_previous" to merge it into the preceding section.
7. **Section summaries**: For each section with keep=true, write a 2-3 \
sentence "summary" capturing the key takeaway from the theory_snippet.
8. **Language**: Detect the dominant language of the theory text and ensure \
titles, descriptions, and summaries use that same language.
9. Return ONLY a valid JSON object — no markdown fences, no commentary."""

_RESTRUCTURE_USER = """\
Course title: {course_title}
Module index: {module_index}

Module skeleton:
```json
{module_skeleton}
```

Return the improved module as a JSON object with this exact schema:
{{
  "title": "improved module title (engaging, question-based when possible, NO numbered prefixes)",
  "description": "module description (1-3 sentences)",
  "language": "detected dominant language (e.g. Español, English)",
  "submodules": [
    {{
      "title": "submodule title (NO numbered prefixes like 1.1 or 2.3)",
      "sections": [
        {{
          "title": "section title (descriptive, NO numbered prefixes)",
          "keep": true,
          "summary": "2-3 sentence summary of the section content"
        }}
      ]
    }}
  ]
}}

Values for "keep":
- true: keep this section as-is
- false: remove (empty, copyright-only, OCR junk, corrupted content)
- "merge_with_previous": merge into the preceding section (for tiny sections <100 words)

When in doubt, keep the section."""

_restructure_prompt = ChatPromptTemplate.from_messages([
    ("system", _RESTRUCTURE_SYSTEM),
    ("human", _RESTRUCTURE_USER),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _robust_json_loads(text: str) -> dict:
    """Parse JSON tolerantly, handling control characters and trailing commas."""
    import re

    text = _strip_markdown_fences(text.strip())

    # First attempt: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Escape unescaped control chars inside string values (common LLM issue)
    cleaned = re.sub(r'[\x00-\x1f\x7f]', lambda m: ' ' if m.group() in ('\n', '\r', '\t') else '', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
            return json.loads(fixed)

    raise json.JSONDecodeError("Could not extract valid JSON", text, 0)


def _module_to_skeleton(module: Module) -> str:
    """Build a compact JSON skeleton of a module for the LLM prompt."""
    skeleton = {
        "title": module.title,
        "description": module.description,
        "submodules": [],
    }
    for sm in module.submodules:
        sm_data = {"title": sm.title, "sections": []}
        for sec in sm.sections:
            theory_snippet = sec.theory[:300] + "..." if len(sec.theory) > 300 else sec.theory
            sm_data["sections"].append({
                "title": sec.title,
                "theory_snippet": theory_snippet,
                "theory_length": len(sec.theory),
                "has_images": bool(sec.source_images),
            })
        skeleton["submodules"].append(sm_data)
    return json.dumps(skeleton, ensure_ascii=False, indent=2)


def _strip_numbered_prefix(title: str) -> str:
    """Remove leading numbered prefixes like ``1.1``, ``2.3.1`` from a title."""
    return _NUMBERED_PREFIX_RE.sub("", title).strip()


def _normalize_title_for_match(title: str) -> str:
    """Normalize a title for fuzzy matching between original and LLM output."""
    t = _strip_numbered_prefix(title).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _apply_restructure(module: Module, llm_result: dict) -> Module:
    """Apply restructuring decisions from the LLM back to the module."""
    if llm_result.get("title"):
        module.title = _strip_numbered_prefix(llm_result["title"])

    if llm_result.get("description"):
        module.description = llm_result["description"]

    llm_submodules = llm_result.get("submodules", [])
    if not llm_submodules:
        return module

    # Build lookup: normalized original title -> LLM section info
    # Also keep a parallel list for positional fallback
    sm_lookup: list[dict] = []
    for sm in llm_submodules:
        sec_list = []
        for sec in sm.get("sections", []):
            sec_list.append({
                "title": sec.get("title", ""),
                "norm": _normalize_title_for_match(sec.get("title", "")),
                "keep": sec.get("keep", True),
                "summary": sec.get("summary", ""),
            })
        sm_lookup.append({
            "title": _strip_numbered_prefix(sm.get("title", "")),
            "norm": _normalize_title_for_match(sm.get("title", "")),
            "sections": sec_list,
        })

    def _find_llm_section(orig_title: str, llm_sections: list[dict]) -> dict | None:
        norm = _normalize_title_for_match(orig_title)
        for ls in llm_sections:
            if ls["norm"] == norm:
                return ls
        for ls in llm_sections:
            if norm in ls["norm"] or ls["norm"] in norm:
                return ls
        return None

    new_submodules: list[Submodule] = []
    for sm_idx, orig_sm in enumerate(module.submodules):
        # Match LLM submodule by normalized title or position
        llm_sm = None
        orig_norm = _normalize_title_for_match(orig_sm.title)
        for lsm in sm_lookup:
            if lsm["norm"] == orig_norm or orig_norm in lsm["norm"] or lsm["norm"] in orig_norm:
                llm_sm = lsm
                break
        if llm_sm is None and sm_idx < len(sm_lookup):
            llm_sm = sm_lookup[sm_idx]

        if llm_sm is not None:
            orig_sm.title = _strip_numbered_prefix(llm_sm["title"] or orig_sm.title)

        llm_secs = llm_sm["sections"] if llm_sm else []

        kept_sections: list[Section] = []
        for sec in orig_sm.sections:
            matched = _find_llm_section(sec.title, llm_secs)
            keep_val = matched["keep"] if matched else True

            if matched and matched.get("title"):
                sec.title = _strip_numbered_prefix(matched["title"])
            else:
                sec.title = _strip_numbered_prefix(sec.title)

            if keep_val is False:
                logger.info("  Removed junk section: %s", sec.title)
                continue

            if keep_val == "merge_with_previous" and kept_sections:
                prev = kept_sections[-1]
                prev.theory = prev.theory + "\n\n" + sec.theory
                if sec.source_images:
                    prev.source_images = (prev.source_images or []) + sec.source_images
                logger.info("  Merged section into previous: %s -> %s", sec.title, prev.title)
                continue

            if matched and matched.get("summary"):
                sec.summary = matched["summary"]

            kept_sections.append(sec)

        if kept_sections:
            for i, sec in enumerate(kept_sections, 1):
                sec.index = i
            orig_sm.sections = kept_sections
            new_submodules.append(orig_sm)

    if new_submodules:
        for i, sm in enumerate(new_submodules, 1):
            sm.index = i
        module.submodules = new_submodules

    return module


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def restructure_module(
    module: Module,
    course_title: str,
    provider: str = "mistral",
    max_retries: int = 3,
) -> tuple[Module, str | None]:
    """Restructure a single module using an LLM.

    Returns:
        Tuple of (improved Module, detected language or None).
    """
    skeleton = _module_to_skeleton(module)

    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.1}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)

    chain = _restructure_prompt | llm | StrOutputParser()

    for attempt in range(max_retries):
        try:
            raw = chain.invoke({
                "course_title": course_title,
                "module_index": module.index,
                "module_skeleton": skeleton,
            })
            result = _robust_json_loads(raw)

            detected_lang = result.get("language")
            module = _apply_restructure(module, result)
            return module, detected_lang

        except json.JSONDecodeError as e:
            logger.warning(
                "Restructure attempt %d/%d failed (JSON): %s",
                attempt + 1, max_retries, e,
            )
        except Exception as e:
            logger.warning(
                "Restructure attempt %d/%d failed: %s",
                attempt + 1, max_retries, e,
            )

    logger.error("All restructure attempts failed for module %d, keeping original", module.index)
    return module, None


def restructure_course(
    state: CourseState,
    provider: str | None = None,
    max_retries: int = 3,
) -> CourseState:
    """Restructure all modules in a CourseState.

    Returns the updated state.  If a detected language is consistent across
    modules it is also returned via ``state.config.language`` override.
    """
    provider = provider or state.config.text_llm_provider
    course_title = state.title or "Untitled Course"

    print(f"Restructuring {len(state.modules)} modules with LLM ({provider})...")

    detected_languages: list[str] = []

    for module in state.modules:
        print(f"   Module {module.index}: {module.title[:60]}...")
        module, lang = restructure_module(
            module,
            course_title=course_title,
            provider=provider,
            max_retries=max_retries,
        )
        if lang:
            detected_languages.append(lang)
        secs = sum(len(sm.sections) for sm in module.submodules)
        print(f"      → {module.title[:60]}  ({len(module.submodules)} submodules, {secs} sections)")

    if detected_languages:
        from collections import Counter
        dominant = Counter(detected_languages).most_common(1)[0][0]
        print(f"   Detected language: {dominant}")
        state.config.language = dominant

    print("Restructuring complete!")
    return state


def detect_content_language(state: CourseState) -> str | None:
    """Detect the dominant language from theory text without using an LLM.

    Checks for common Spanish, French, and Portuguese stopwords.  Falls back
    to ``None`` (caller keeps existing config) if detection is inconclusive.
    """
    theory_sample = ""
    for m in state.modules:
        for sm in m.submodules:
            for sec in sm.sections:
                theory_sample += " " + sec.theory[:500]
                if len(theory_sample) > 3000:
                    break

    if not theory_sample.strip():
        return None

    words = set(theory_sample.lower().split())

    _ES_STOPWORDS = {"de", "la", "el", "en", "los", "las", "del", "que", "por", "con", "una", "para", "como", "más", "este", "esta", "pero", "también", "puede", "cuando"}
    _EN_STOPWORDS = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "been", "have", "has", "which", "their", "they", "will", "would", "could"}
    _FR_STOPWORDS = {"les", "des", "une", "dans", "pour", "avec", "sur", "par", "qui", "sont", "cette", "mais", "aussi", "peut", "entre", "comme", "fait", "nous", "vous"}
    _PT_STOPWORDS = {"dos", "das", "uma", "nos", "nas", "para", "com", "por", "que", "como", "mais", "esta", "pode", "quando", "sobre", "foi", "são", "tem", "seu"}

    scores = {
        "Español": len(words & _ES_STOPWORDS),
        "English": len(words & _EN_STOPWORDS),
        "Français": len(words & _FR_STOPWORDS),
        "Português": len(words & _PT_STOPWORDS),
    }

    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]

    if best_score < 3:
        return None

    return best_lang
