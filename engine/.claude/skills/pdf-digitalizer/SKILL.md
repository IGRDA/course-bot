---
name: pdf-digitalizer
description: Converts a PDF into structured markdown chapters and runs the digitalization workflow to produce a complete course. Use when the user wants to digitalize a PDF, convert a PDF to a course, or run workflow_digitalize.py from PDF source material.
---

# PDF to Course Digitalizer

Convert a PDF into clean markdown chapters, then run the digitalization workflow.

## Step 1: Convert PDF to markdown

```bash
python -m tools.pdf2md.docling.converter <PDF_PATH> --no-ocr --split -o <OUTPUT_DIR>
```

Use `--language en` for English PDFs (default: `es`). Output lands in `<OUTPUT_DIR>/chapters/` with one `.md` per module.

## Step 2: Post-process chapters

Read every `.md` file in `chapters/` and apply the fixes in [POST_PROCESSING.md](POST_PROCESSING.md).

### CRITICAL -- Structure constraints

Validate these limits before proceeding to Step 3. Split or merge files and headings as needed:

- **Modules**: minimum 3, maximum 15 (one `#` heading per `.md` file)
- **Submodules per module**: minimum 3, maximum 15 (`##` headings)
- **Sections** (`###`): each must have at least 150 words of flowing prose

### Required heading structure

Each `.md` file must follow this hierarchy exactly:

```markdown
# Module Title

Brief module description paragraph.

## Submodule Title

### Section Title
Theory text (150+ words, prose paragraphs, not bullet lists)...

### Another Section
More theory...

## Another Submodule

### Section Title
...
```

Rules: no numbered prefixes on headings, no duplicate `##`/`###` titles within the same module, no level skips (`#` to `###`).

## Step 3: Run the digitalization workflow

```bash
python -m workflows.workflow_digitalize \
  --source <CHAPTERS_FOLDER> --html \
  --title "Course Title" --language auto
```

Add `--no-images` ONLY when the user explicitly does not want images. Add `--pdf` or `--podcast` as needed. Default LLM provider is Mistral.

### CRITICAL -- Never kill the workflow

NEVER kill, restart, or interrupt `workflow_digitalize`. The process handles rate-limit errors (HTTP 429) automatically with exponential backoff. Monitor by checking terminal output every 3 minutes. Only intervene if the process has produced zero new output for more than 15 minutes.

## Step 4: Validate output

Check the output JSON modules:

- Every module has non-null `video`, `bibliography`, `relevant_people`, `mindmap`
- Section theory is clean, summaries are populated
- Image queries are descriptive (not just "Image")
- If issues found, fix the markdown source and re-run Step 3
