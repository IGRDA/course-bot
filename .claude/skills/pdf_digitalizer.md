---
name: pdf-to-course-markdown
description: Convert any PDF (textbook, course, slides) into well-structured markdown chapter files ready for the course-generator workflow. Use when the user wants to digitalize a PDF, convert a PDF to course format, or prepare markdown from a PDF for workflow_digitalize.py.
---

# PDF to Course Markdown

Convert a PDF into a folder of clean, well-structured markdown files (one per module/chapter) that feed directly into `workflow_digitalize.py`.

## Input / Output Contract

- **Input**: Any PDF -- textbook, course material, slide deck, scanned document
- **Output**: A folder of `.md` files with this heading structure:

```markdown
# Module Title (descriptive, no numbering)

Descriptive paragraph about the module content.

## Submodule Title (descriptive, no numbering)

### Section Title (descriptive, no numbering)
Theory body text...

### Another Section
More theory...
```

## Step 1: Run Docling Conversion

```bash
cd /Users/inaki/Documents/Personal/course-generator-agent
python -m tools.pdf2md.docling.converter <PDF_PATH> --split -o <OUTPUT_DIR>
```

Options to consider:
- `--language es` (default) or `--language en` for English PDFs
- `--no-ocr` only if the PDF has a trusted embedded text layer

This produces: a single `.md` file + `*_artifacts/` folder with images, then splits into `chapters/` with one `.md` per module.

## Step 2: Post-Process Each Chapter File

Read every `.md` file in `chapters/` and apply these fixes **in order**:

### 2a. Strip Numbers, Letters, and Prefixes from Headings

Remove **all** leading numbering and lettering from headings. This includes numeric patterns (`1.`, `1.1`, `2.1.3`), letter prefixes (`A.`, `B.`, `C.`), and Roman numerals (`I.`, `II.`, `III.`). The hierarchy is encoded in the heading level itself (`#`, `##`, `###`), so prefixes are redundant and produce ugly titles in the final course.

```
BEFORE: ## 1.1 Limitaciones de la fisica clasica
AFTER:  ## Limitaciones de la fisica clasica

BEFORE: ### 2.1.1 El experimento de la doble rendija
AFTER:  ### El experimento de la doble rendija

BEFORE: ## A. Servicios e instituciones
AFTER:  ## Servicios e instituciones

BEFORE: ### III. Parámetros dietéticos
AFTER:  ### Parámetros dietéticos
```

Regex: strip `^(\d+(\.\d+)*\.?\s*|[A-Z]\.\s*|[IVXLC]+\.\s*)` from the title text after the `#` markers.

### 2b. Clean OCR Artifacts

Remove these common artifacts from body text:
- Copyright lines: `© 20XX ... Todos los derechos reservados`
- Formula placeholders: `<!-- formula-not-decoded -->`
- Stray single uppercase letters on their own line
- Unrelated English fragments in Spanish text (e.g. `ATHLETIC`, `Infection`, `HIGH FATALITY RATE`, `EXIT OF WORMHOLE`) -- these come from OCR picking up unrelated page elements
- Repeated/duplicated paragraphs (Docling sometimes duplicates content)

### 2c. Fix Spacing Issues

OCR often merges words. Scan for camelCase-like patterns where a lowercase letter is followed by an uppercase letter with no space: `ThomasYoung` -> `Thomas Young`, `MaxPlanck` -> `Max Planck`.

### 2d. Fix Malformed Headings

If a heading looks like a formula, equation, or garbled text (no recognizable words, mostly symbols), either:
- Merge its content into the previous section, or
- Replace it with a descriptive title derived from the first sentence of its body

### 2e. Remove Duplicate and Redundant Headings

The `##` (submodule) title and the first `###` (section) title underneath it must **not** be identical or near-identical. When a submodule contains a single child section with the same name, merge them into one `##` heading and place the body text directly under it.

```
BAD:
## Instituciones residenciales        ← submodule
### Instituciones residenciales       ← section (same title)

GOOD:
## Instituciones residenciales        ← single heading, body text follows
```

Also ensure the description paragraph under each `##` is not just a copy of the title.

### 2f. Ensure Minimum Section Content

Each `###` section should have at least ~150 words of theory. If a section has fewer:
- Merge it with the previous or next section on the same topic
- If it's a wrapper/intro section with a short paragraph, move its text into the first child section and remove the heading

### 2g. Validate Image References

Check that `![Image](path)` references point to existing files. Remove broken references. Ensure paths are relative to the chapter file.

## Step 3: Validate the Result

Before passing to `workflow_digitalize.py`, verify:

- [ ] Each `.md` file starts with a `#` heading (module title)
- [ ] Module titles are descriptive (not just "Modulo 1")
- [ ] No numbers or letters prefix any heading (no `1.`, `A.`, `III.`)
- [ ] No submodule (`##`) has a child section (`###`) with the same title
- [ ] Heading levels follow `#` > `##` > `###` hierarchy (no skips)
- [ ] No section has fewer than 100 words of theory
- [ ] No OCR garbage remains in the text
- [ ] Image paths resolve to existing files
- [ ] `00_frontmatter.md` exists but can be ignored (it's skipped by the parser)

## Step 4: Run the Digitalization Workflow

Always load API keys first with `source env.secrets`.

The default LLM provider is **Mistral**. Do not change the provider unless the user explicitly requests it.

```bash
source env.secrets
python -m main.workflow_digitalize \
  --source <CHAPTERS_FOLDER> \
  --html \
  --title "Course Title" \
  --language auto
```

Add `--pdf` or `--podcast` as needed. By default all enrichments are enabled (videos, bibliography, people, mindmap, activities, images).

### Rate Limits and Retries

**Do NOT kill the workflow process when you see rate-limit errors (HTTP 429).** All LLM clients (Mistral, Gemini, etc.) have built-in exponential-backoff retry logic (up to 10 attempts, waits up to 500 seconds between retries). Rate-limit messages in the output are normal and expected -- the process will recover on its own. Be patient and let it run.

Only intervene if the process has been completely idle (no new output and no new step files in the `steps/` folder) for more than **15 minutes**.

## Iterative Correction Pattern

After running the workflow, inspect the output JSON modules:

1. **Check enrichment fields**: Every module should have non-null `video`, `bibliography`, `relevant_people`, and `mindmap`
2. **Check section quality**: Theory text should be clean, summaries should be populated, image queries should be descriptive (not just "Image")
3. **If issues found**: Go back to the markdown files, fix the root cause, and re-run

## Robustness Notes for Different PDF Types

| PDF Type | Likely Issues | Mitigation |
|----------|--------------|------------|
| Textbook | Deep heading hierarchy, many equations | Use `--language` flag, watch for formula-as-heading |
| Slide deck | Very short sections, bullet-heavy | Merge slides into coherent sections of 150+ words |
| Scanned document | OCR noise, spacing issues | Use `force_ocr=True` (default), aggressive artifact cleaning |
| Multi-column | Text interleaving between columns | Docling handles this; verify paragraph continuity |
| Generated PDF (LaTeX) | Clean text but may need repair | Converter auto-repairs via PyMuPDF if needed |

## Key Files Reference

- Converter: `tools/pdf2md/docling/converter.py`
- Parser: `agents/md_digitalizer/parser.py`
- Restructurer: `agents/md_digitalizer/restructurer.py`
- Workflow: `main/workflow_digitalize.py`
- Image injection: `main/nodes/digitalize.py`
