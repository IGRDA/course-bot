---
name: pdf-digitalizer
description: Converts a PDF into structured markdown chapters and runs the digitalization workflow to produce a complete course. Use when the user wants to digitalize a PDF, convert a PDF to a course, or run workflow_digitalize.py from PDF source material.
---

# PDF to Course Digitalizer

Convert the PDF to markdown chapters, fix up and embed images, validate, then run the workflow.

## Step 1: Convert PDF to markdown

```bash
python -m tools.pdf2md.docling.converter <PDF_PATH> --no-ocr --split -o <OUTPUT_DIR>
```

Use `--language en` for English PDFs (default: `es`). Output lands in `<OUTPUT_DIR>/chapters/` with one `.md` per module plus an artifacts directory with extracted images.

The converter uses Docling with PyMuPDF fallback — always use this command, even if conversion has issues. If it fails, report the error.

## Step 2: Fix up chapters and embed images

After conversion, you may need to clean up headings, merge short sections, or restructure content. When doing so, **preserve image references exactly as the converter produced them.**

**CRITICAL — use exact filenames from the artifacts directory, never invent your own:**

1. List the actual image files extracted by the converter:
   ```bash
   ls <OUTPUT_DIR>/chapters/artifacts/
   ```
2. When embedding images in markdown, copy the **exact filename** from the directory listing. The converter names files based on internal PDF identifiers (e.g. `artifacts/image_page5_1.png`) — you MUST use these exact names. **NEVER rename, simplify, or generate sequential names** like `img_001.jpg`, `img_1.png`, etc. Those files do not exist on disk and will break the pipeline.
3. Embed as: `![descriptive alt text](artifacts/image_page5_1.png)` — the path inside `()` must match a real file from step 1.
4. Match images to nearby text content. Max 4 images per `###` section. Distribute across chapters.
5. Replace generic alt text with 5-10 word descriptions derived from surrounding content.
6. **Validation (required before proceeding):**
   ```bash
   # Extract all image filenames referenced in markdown
   grep -ohP '(?<=\(artifacts/)[^)]+' <OUTPUT_DIR>/chapters/*.md | sort > /tmp/md_images.txt
   # List actual files on disk
   ls <OUTPUT_DIR>/chapters/artifacts/ | sort > /tmp/disk_images.txt
   # Show any referenced files that don't exist on disk (MUST be empty)
   comm -23 /tmp/md_images.txt /tmp/disk_images.txt
   ```
   If `comm` outputs any filenames, those images will be missing from the final course. Fix them by replacing with actual filenames from `/tmp/disk_images.txt`.

## Step 3: Validate chapters

Before running the workflow, verify all checks in [validation-checklist.md](validation-checklist.md).

## Step 4: Run the digitalization workflow.

```bash
python -m workflows.workflow_digitalize \
  --source <CHAPTERS_FOLDER> --html \
  --title "Course Title" --language auto
```

Add `--pdf` for PDF book, `--no-images` to skip internet image search, `--podcast` for audio.

Let it run to completion — it handles rate-limit retries automatically.

Monitor by checking the `steps/` directory: `12_parse_markdown.json` → `16_validate_structure.json` → `14_restructure.json` → activities, HTML, etc.

The **validate_structure** step (step 16) runs automatically after parsing. If the converter produced too few modules (e.g. a single module with many submodules because the chapter-splitting regex didn't match), this step uses an LLM to analyze the heading structure and re-split into proper modules. It only makes an LLM call when an anomaly is detected; well-structured inputs pass through instantly.

**IMPORTANT: Do NOT kill the workflow.** The restructure and validation steps make multiple LLM calls with automatic retries and fallback providers. Log lines like `[Mistral] Retrying call` mean the process IS working — do not interrupt it. Only intervene if you see a fatal error (unhandled exception / stack trace).

## Step 5: Check output

- Module JSON files should be > 5 KB with non-empty `description` fields and `html` content
- If `--pdf` was used, check `book/book.pdf` exists

For PDF-specific troubleshooting, see [validation-checklist.md](validation-checklist.md).
