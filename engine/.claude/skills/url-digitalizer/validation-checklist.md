# URL Digitalizer Reference

## Validation Checklist

Before running `workflow_digitalize.py`, verify:

- [ ] **Image gate passed**: ran the bash gate from Step 3 and it printed "Images OK" (if images were downloaded)
- [ ] Each `.md` file starts with a single `#` heading
- [ ] Module titles are descriptive (not "Page 1" or raw URLs)
- [ ] No numbered/lettered prefixes on any heading
- [ ] No `##` has a `###` child with the same title
- [ ] Heading hierarchy: `#` > `##` > `###` (no skips)
- [ ] No section has fewer than 150 words of theory
- [ ] No web boilerplate remains (nav, cookies, footer)
- [ ] Content is factually faithful to the source website
- [ ] 3-15 module files with substantive content
- [ ] 3-15 submodules per module
- [ ] `![alt](images/img_xxx.jpg)` images present using local paths (not remote URLs) and content-relevant (no icons, logos, SVGs)
- [ ] Image filenames in markdown match actual files on disk (`ls url_source/<slug>/images/` to verify) — never use invented sequential names like `img_001.jpg`
- [ ] Image alt text is descriptive (not empty, not filenames)
- [ ] Images positioned near illustrating text, max 5 per section

## Site-Type Guidance

| Site Type | Issues | Mitigation |
|-----------|--------|------------|
| Corporate/marketing | Thin content, slogans | Merge short pages; expand bullets into paragraphs |
| Documentation portal | Deep nesting | Flatten to 3 heading levels max |
| Blog/knowledge base | Standalone articles | Group by topic; create coherent module narrative |
| SPA | Scraper may get minimal content | Try increasing `--max-depth` or fetching known subpages directly |
| Wiki | Dense cross-linking | Focus on main content area; ignore sidebar links |
| Paywall/auth-gated | Login prompts instead of content | Skip with warning; report inaccessible URLs |

## Image Pipeline Details

`web_scraper --output-dir` crawls the site, discovers image URLs from all pages' HTML, and downloads them locally to `url_source/<slug>/images/`. The markdown chapters reference these local files via relative paths (e.g. `![alt](images/img_abc123.jpg)`).

The parser (`agents/md_digitalizer/parser.py`) extracts `![alt](path)` references with ~200 chars of surrounding text into `Section.source_images`, resolving relative paths against the markdown file's directory. The `inject_local_images_node` copies local image files to the output `images/` directory and sets `{"type": "img", "query": "...", "content": "/path/to/output/images/file.jpg"}` on ParagraphBlocks. The PDF generator finds these local files and includes them via `\includegraphics`.

Sections without source images get stock images from internet search (if enabled via `--images` / no `--no-images` flag). When both are present, scraped website images take priority over stock images for blocks where there is a text match.

## Key Files

- Website scraper: `tools/web_scraper/scraper.py`
- Image extractor (internals reused by scraper): `tools/web_image_extractor/extractor.py`
- Parser: `agents/md_digitalizer/parser.py`
- Restructurer: `agents/md_digitalizer/restructurer.py`
- Workflow: `workflows/workflow_digitalize.py`
- Image injection: `workflows/nodes/digitalize.py`
- PDF generator: `tools/json2book/generator.py`
