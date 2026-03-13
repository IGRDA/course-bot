---
name: url-to-course-markdown
description: Fetch a website (main URL + child links) and convert the content into well-structured markdown chapter files ready for the course-generator workflow. Use when the user wants to create a course from a website, digitalize a URL, or scrape web content for workflow_digitalize.py.
---

# URL to Course Markdown

Fetch a website and its child pages, then produce a folder of clean, well-structured markdown files (one per module/chapter) that feed directly into `workflow_digitalize.py`.

## Input / Output Contract

- **Input**: A root URL -- company site, documentation portal, wiki, blog, knowledge base
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

## Step 1: Fetch the Main URL

Use the **WebFetch** tool to retrieve the root URL the user provided.

```
WebFetch(url="https://example.com")
```

From the returned markdown content:

1. Read and understand the page structure, navigation, and main content.
2. **Extract all internal links** -- links sharing the same domain or relative paths. Collect them into a list.
3. **Filter out** links that are:
   - Anchor-only (`#section`)
   - Static assets (`.css`, `.js`, `.png`, `.jpg`, `.svg`, `.pdf`, `.zip`)
   - Login, auth, or account pages (`/login`, `/signup`, `/account`, `/oauth`)
   - Social media share links
   - Duplicate URLs (normalize trailing slashes and query parameters)
   - Links to external domains (keep only same-domain or explicitly related subdomains)

**Also extract images** from the main URL using the image extractor tool:

```bash
python -m tools.web_image_extractor "https://example.com"
```

This returns a JSON array with `{src, alt, context_heading, context_text}` for each content image. WebFetch strips images from its markdown output, so this separate extraction step is required to capture them. Save the image list alongside the page content for use in Step 3.

## Step 2: Fetch Child Links (Depth 1)

Fetch each filtered child link using **WebFetch**. Batch calls in parallel where possible.

**Hard cap**: Fetch at most **20 child pages**. If more links are available, prioritize:
1. Main navigation / menu links (products, features, about, docs)
2. Content-heavy pages over listing/index pages
3. Pages linked prominently (header/footer nav) over deep inline links

For each fetched page, record:
- Source URL
- Page title (from the first `#` heading or `<title>`)
- The full markdown content returned by WebFetch

**Skip gracefully** any page that:
- Returns an error (404, 403, timeout)
- Requires authentication
- Returns very little content (fewer than 50 words after stripping boilerplate)

**Extract images from all fetched URLs** in a single batch call:

```bash
python -m tools.web_image_extractor "URL1" "URL2" "URL3" ...
```

Pass every successfully fetched URL (main + children) as arguments. The tool outputs a JSON array where each entry has `page_url` and an `images` list. Keep this output for Step 3e.

## Step 3: Clean and Organize Content into Markdown Chapters

### 3a. Strip Web Boilerplate

For every fetched page, remove all non-content elements before organizing:

- Navigation bars and menus
- Cookie consent banners
- Footer text (copyright, legal links, social links)
- Sidebar widgets (related posts, ads, newsletter signup)
- Breadcrumb trails
- "Share this page" blocks
- Repeated header/footer content that appears across multiple pages
- HTML comments and tracking snippets

### 3b. Group Content Thematically into Modules

Analyze all cleaned content and group it into **logical modules** (chapters). Each module becomes one `.md` file. Grouping strategies:

- **By site section**: If the site has clear sections (Products, Features, About, Docs), each becomes a module.
- **By topic**: If pages cover overlapping topics, merge related pages into coherent modules.
- **By hierarchy**: If the site has a parent/child page structure, respect it.

Aim for **3-8 modules** depending on content volume. Each module should have substantial content (multiple submodules with 150+ words per section).

### 3c. Structure Each Module File

For each module, create a `.md` file with the required heading hierarchy:

```markdown
# Descriptive Module Title

Brief paragraph describing what this module covers and why it matters.

## Submodule Title

### Section Title
Body text with at least 150 words of substantive theory content.
Rewrite bullet lists into flowing paragraphs where appropriate.
Preserve factual accuracy -- do not invent information not present in the source.

### Another Section
More theory content...

## Another Submodule

### Section Title
...
```

Rules for structuring:

- **`#` (Module)**: One per file. Descriptive title, no numbering.
- **`##` (Submodule)**: Logical grouping within the module. 2-5 per module.
- **`###` (Section)**: Individual topics. Each must have at least **150 words** of theory.
- **No heading prefixes**: No numbers (`1.`, `1.1`), letters (`A.`), or Roman numerals (`III.`).
- **No identical headings**: A `##` and its first `###` child must not share the same title. If they would, merge them into a single `##` with body text directly underneath.
- **Expand bullet points**: Convert short bullet lists into descriptive paragraphs. The course renderer works best with flowing prose, not bullet-heavy content.
- **Preserve accuracy**: Only use information that was present in the fetched web content. Do not hallucinate facts, features, or details.

### 3d. Write the Files

Save all module files to a folder:

```
url_source/<slugified_course_title>/
  01_introduction.md
  02_core_features.md
  03_use_cases.md
  ...
```

Use the **Write** tool to create each file. File names should be lowercase, snake_case, with a two-digit numeric prefix for ordering.

### 3e. Insert Content Images from the Extractor

WebFetch does **not** include images in its markdown output. Use the image data collected in Steps 1-2 via `python -m tools.web_image_extractor` to place images into the markdown files.

**Match images to sections** -- for each image from the extractor output, use its `context_heading` and `context_text` to find the module section where it belongs. Insert `![alt](src)` on its own line, near the paragraph that best matches `context_text`.

**Limit images per section** -- include at most **2-3 images per `###` section**. Pick the most relevant or visually distinctive images when a page has many. Prefer images that illustrate key concepts over decorative or repetitive ones.

**Improve generic alt text** -- if the extractor returned an empty or generic alt (`""`, `"image"`, `"img"`, `"photo"`, or a filename like `"IMG_1_SOBRE_SAPA"`), replace it with a short description (5-10 words) derived from the surrounding paragraph. Descriptive alt text becomes the image `query` in the course JSON.

**Keep images in position** -- place each `![alt](url)` immediately after the paragraph it illustrates. The parser uses the ~200 characters before each image to match it to the right HTML block in the final course. Moving images away from their context breaks this matching.

**Example** of a section with an image inserted:

```markdown
### Transmisiones sin convertidor de par

Los sistemas de movilidad de SAPA se caracterizan por maximizar las prestaciones
del vehículo logrando una eficiencia significativamente superior gracias a su
concepto de transmisión sin convertidor de par...

![Transmisión de alta eficiencia SAPA para vehículos de cadenas](https://sapa.es/wp-content/uploads/2024/11/IMG_1_SOBRE_SAPA.jpg)

La eliminación del convertidor de par reduce significativamente la emisión de calor...
```

**How this works in the pipeline**: The parser (`agents/md_digitalizer/parser.py`) extracts `![alt](url)` references along with the ~200 characters of surrounding text and stores them in `Section.source_images`. The `inject_local_images_node` recognizes HTTP URLs and uses them directly as the image `content` field -- the same `{"type": "img", "query": "...", "content": "https://..."}` format used by internet image search. Any sections without source images still get images from the workflow's built-in internet image search step (if enabled).

## Step 4: Validate the Result

Before passing to `workflow_digitalize.py`, verify:

- [ ] Each `.md` file starts with a single `#` heading (module title)
- [ ] Module titles are descriptive (not just "Page 1" or the raw URL)
- [ ] No numbers or letters prefix any heading (no `1.`, `A.`, `III.`)
- [ ] No submodule (`##`) has a child section (`###`) with the same title
- [ ] Heading levels follow `#` > `##` > `###` hierarchy (no level skips)
- [ ] No section has fewer than 100 words of theory
- [ ] No web boilerplate remains (nav links, cookie text, footer content)
- [ ] Content is factually faithful to the source website
- [ ] At least 3 module files exist with substantive content
- [ ] `![alt](url)` images are present in the markdown files (check the extractor found images)
- [ ] Any `![alt](url)` images are content-relevant (no icons, logos, tracking pixels, SVGs)
- [ ] Image alt text is descriptive (not empty, not just "image", not raw filenames)
- [ ] Images are positioned near the text they illustrate (not grouped at top/bottom)
- [ ] No more than 3 images per `###` section

If any check fails, fix the markdown files before proceeding.

## Step 5: Run the Digitalization Workflow

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

## Robustness Notes for Different Website Types

| Site Type | Likely Issues | Mitigation |
|-----------|--------------|------------|
| Corporate / marketing site | Thin content, heavy on slogans and CTAs | Merge short pages; expand bullet features into paragraphs |
| Documentation portal | Very structured, may have deep nesting | Respect existing hierarchy; flatten to 3 heading levels max |
| Blog / knowledge base | Articles may be standalone and unrelated | Group by topic/tag; create a coherent module narrative |
| Single-page app (SPA) | WebFetch may return minimal content | Note limitation to user; try fetching known subpages directly |
| Wiki | Dense cross-linking, inconsistent formatting | Focus on main content area; ignore sidebar/meta links |
| Paywall / auth-gated site | Pages return login prompts instead of content | Skip with a warning; report which URLs were inaccessible |

## Key Files Reference

- Web image extractor: `tools/web_image_extractor.py`
- Parser: `agents/md_digitalizer/parser.py`
- Restructurer: `agents/md_digitalizer/restructurer.py`
- Workflow: `main/workflow_digitalize.py`
- Image injection: `main/nodes/digitalize.py`
