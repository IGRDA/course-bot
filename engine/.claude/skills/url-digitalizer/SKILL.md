---
name: url-digitalizer
description: Scrapes a website (root + child pages) using web_scraper, extracts text and images, and converts content into structured markdown chapters for workflow_digitalize.py. Use when the user wants to create a course from a URL, digitalize a website, or scrape web content into course format.
---

# URL to Course Markdown

Scrape a website (root + child pages) with `web_scraper`, produce markdown chapter files with embedded images, then run `workflows.workflow_digitalize`.

## Step 1: Scrape Website (single command)

Run the `web_scraper` tool to crawl the root URL and all same-domain child pages, extracting text content and images from every page in one pass:

```bash
python -m tools.web_scraper --output-dir url_source/<slug>/images --max-pages 20 --max-depth 1 <ROOT_URL> > /tmp/scraped_site.json
```

This single command:
- Fetches the root page and discovers all same-domain internal links
- Crawls up to 20 child pages (configurable via `--max-pages`)
- Extracts cleaned text content (title, headings, body text) from every page
- Extracts images from all 4 HTML delivery methods (`<img>`, `data-src`, `<picture><source srcset>`, CSS `background-image`)
- Downloads all images to the `--output-dir` directory
- Outputs structured JSON to stdout with everything needed for chapter generation
- Uses Playwright stealth browser as fallback (handles CAPTCHA, SiteGround, Cloudflare)

Read the JSON from `/tmp/scraped_site.json`. It contains:
```json
{
  "root_url": "https://example.com/",
  "pages_fetched": 14,
  "total_images": 112,
  "pages": [
    {
      "url": "https://example.com/about/",
      "title": "About - Example",
      "content": "Full cleaned text...",
      "headings": ["Heading 1", "Heading 2"],
      "content_length": 3855,
      "images": [
        {
          "src": "https://example.com/wp-content/uploads/photo.jpg",
          "alt": "Description",
          "context_heading": "Nearest heading",
          "context_text": "Surrounding text...",
          "local_path": "images/img_abc123.jpg"
        }
      ],
      "internal_links": ["https://example.com/other-page/"]
    }
  ]
}
```

If the scraper fails entirely (e.g. site requires authentication), proceed without images — the workflow can still generate a course from text alone.

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

![Descriptive alt text](images/img_abc123def456.jpg)

### Another Section
More theory...
```

### Create Files with Embedded Images

Your Python script that generates each markdown file **MUST also embed images** from the scraped JSON inline. Image embedding is NOT a separate step — it happens during file creation.

Save files to:
```
url_source/<slugified-title>/
  01_introduction.md
  02_core_features.md
  ...
```

**Image embedding pattern** — include this logic in your file-generation script:

```python
# For each page that contributes content to a section:
for img in page["images"]:
    path = img.get("local_path") or img.get("src", "")
    if not path:
        continue
    alt = img.get("alt", "").strip()
    if not alt or alt.lower() in ("image", "img", "figure", ""):
        alt = img.get("context_text", "Image")[:60].strip()
    heading = img.get("context_heading", "")
    # Place image after the section whose heading best matches context_heading
    section_text += f'\n\n![{alt}]({path})\n'
```

**Image rules (CRITICAL):**
- Use the **exact `local_path`** from JSON (e.g. `images/img_a1b2c3d4e5f6.jpg`). **NEVER** invent filenames like `img_001.jpg`.
- If `local_path` is null, use the `src` URL as fallback.
- Match each image to the `###` section nearest to its `context_heading`. Max 4 images per section; distribute across chapters.
- Replace generic alt text with 5-10 word descriptions from `context_text`.

## Step 3: Validate (HARD GATE)

Before running the workflow, verify all checks in [validation-checklist.md](validation-checklist.md).

Then run this image gate — **DO NOT proceed to Step 4 until it passes**:

```bash
IMG_COUNT=$(grep -rc '!\[' url_source/<slug>/*.md | awk -F: '{s+=$2}END{print s}')
DISK_COUNT=$(ls url_source/<slug>/images/ 2>/dev/null | wc -l)
if [ "$DISK_COUNT" -gt 0 ] && [ "$IMG_COUNT" -eq 0 ]; then
  echo "FATAL: $DISK_COUNT images on disk but 0 referenced in markdown. Go back to Step 2 and embed images."
  false
fi
echo "Images OK: $IMG_COUNT references, $DISK_COUNT files on disk"
# Verify referenced files exist on disk
grep -oE 'images/[^)]+' url_source/<slug>/*.md | sed 's|.*images/||' | sort -u > /tmp/md_imgs.txt
ls url_source/<slug>/images/ | sort > /tmp/disk_imgs.txt
MISSING=$(comm -23 /tmp/md_imgs.txt /tmp/disk_imgs.txt | wc -l)
if [ "$MISSING" -gt 0 ]; then
  echo "FATAL: $MISSING image(s) referenced in markdown but missing from disk:"
  comm -23 /tmp/md_imgs.txt /tmp/disk_imgs.txt
  false
fi
echo "All referenced images exist on disk."
```

If either check fails, go back to Step 2 and fix the markdown files before proceeding.

## Step 4: Run Workflow

Run this command **exactly** — copy-paste and only replace `<CHAPTERS_FOLDER>` and `"Course Title"`:

```bash
python -m workflows.workflow_digitalize \
  --source <CHAPTERS_FOLDER> \
  --html \
  --title "Course Title" \
  --language auto
```

Optional flags (append to the command above):
- `--no-images` — skip internet stock image search (e.g. Freepik). Website images from `web_scraper` are always preserved regardless.
- `--pdf` — generate a PDF book
- `--podcast` — generate podcast audio

Do NOT change the provider (default Mistral) unless the user requests it.

### FORBIDDEN — do NOT use any of these

- `unbuffer`, `stdbuf`, `script`, or any output-buffering wrapper
- `nohup`, `setsid`, or any backgrounding wrapper
- `timeout` command prefix
- Piping the workflow through `tee`, `grep`, `sed`, or any filter
- Adding arguments not listed above (e.g. `--verbose`, `--debug`)

The command MUST be exactly `python -m workflows.workflow_digitalize ...` with no prefix and no pipe.

### CRITICAL: Process Management

- **NEVER kill, restart, or interrupt the workflow process**
- **NEVER run the workflow a second time** — if it fails, report the error from `WORKFLOW_ERROR` in the output folder. Do NOT retry.
- Rate-limit errors (HTTP 429) are normal — built-in retry handles them automatically
- The restructure step makes multiple LLM calls with automatic retries and fallback providers (Mistral → Groq → OpenAI). Log lines like `[Mistral] Retrying call` or `All attempts with mistral failed, trying next provider...` are normal — do NOT intervene
- **Let the Bash command run to completion** — do NOT set a short timeout on the Bash tool. The workflow can take 30-60 minutes for large courses. This is expected.
- Only intervene if the process has already exited with a non-zero exit code

## Step 5: Inspect Output

Check output JSON modules for non-null `video`, `bibliography`, `relevant_people`, `mindmap`. If issues found, fix markdown source and re-run.

For site-type specific guidance, see [validation-checklist.md](validation-checklist.md).
