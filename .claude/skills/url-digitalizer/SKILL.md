---
name: url-digitalizer
description: Fetches a website and its child pages, extracts images, and converts content into structured markdown chapters for workflow_digitalize.py. Use when the user wants to create a course from a URL, digitalize a website, or scrape web content into course format.
---

# URL to Course Markdown

Fetch a website (root + child pages), extract images, produce markdown chapter files, then run `workflow_digitalize.py`.

## Step 1: Fetch & Extract

1. **WebFetch** the root URL. Extract all same-domain internal links.
2. Filter out anchors, static assets, auth pages, social links, duplicates, external domains.
3. Fetch child links (max 20 pages) via WebFetch. Prioritize navigation links and content-heavy pages. Skip 404s, auth-gated, or <50-word pages.
4. **Extract images** from all fetched URLs:

```bash
python -m tools.web_image_extractor "URL1" "URL2" ...
```

Returns JSON with `{src, alt, context_heading, context_text}` per image. WebFetch strips images, so this step is required.

## Step 2: Build Markdown Chapters

Strip web boilerplate (nav, footer, cookies, sidebars, breadcrumbs) from all pages, then organize into markdown files.

### Structure Constraints (CRITICAL)

- **Modules** (files): min 3, max 15
- **Submodules per module** (`##`): min 3, max 15
- **Sections** (`###`): each must have **150+ words** of prose (not bullets)
- Heading hierarchy: exactly `#` > `##` > `###` — no level skips
- **No numbering** in headings — no `1.`, `A.`, `III.` prefixes
- A `##` and its first `###` must NOT share the same title — merge if they would
- Convert bullet lists into flowing paragraphs

### File Format

```markdown
# Descriptive Module Title

Paragraph describing what this module covers.

## Submodule Title

### Section Title
At least 150 words of substantive theory content as flowing prose.
Preserve factual accuracy — do not invent information.

![Descriptive alt text](https://example.com/image.jpg)

### Another Section
More theory...
```

### Save Files

```
url_source/<slugified-title>/
  01_introduction.md
  02_core_features.md
  ...
```

### Insert Images

Place `![alt](url)` from the extractor output near the paragraph matching `context_text`. Max 3 images per `###` section. Replace generic alt text (`""`, `"image"`, filenames) with 5-10 word descriptions from surrounding text.

## Step 3: Validate

Before running the workflow, verify all checks in [reference.md](reference.md).

## Step 4: Run Workflow

```bash
python -m main.workflow_digitalize \
  --source <CHAPTERS_FOLDER> \
  --html \
  --title "Course Title" \
  --language auto
```

- If user does NOT want images: add `--no-images`
- If user wants images (default): do NOT add `--no-images`
- Add `--pdf` or `--podcast` as needed
- Default provider is **Mistral** — do not change unless user requests it

### CRITICAL: Process Management

- **NEVER kill, restart, or interrupt the workflow process**
- Rate-limit errors (HTTP 429) are normal — built-in retry handles them automatically
- **Check process status every 3 minutes** by reading the terminal output
- Only intervene if completely idle (no new output, no new step files) for **15+ minutes**
- Wait for the complete process to finish before reporting results

## Step 5: Inspect Output

Check output JSON modules for non-null `video`, `bibliography`, `relevant_people`, `mindmap`. If issues found, fix markdown source and re-run.

For site-type specific guidance, see [reference.md](reference.md).
