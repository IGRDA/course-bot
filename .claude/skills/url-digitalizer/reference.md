# URL Digitalizer Reference

## Validation Checklist

Before running `workflow_digitalize.py`, verify:

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
- [ ] `![alt](url)` images present and content-relevant (no icons, logos, SVGs)
- [ ] Image alt text is descriptive (not empty, not filenames)
- [ ] Images positioned near illustrating text, max 3 per section

## Site-Type Guidance

| Site Type | Issues | Mitigation |
|-----------|--------|------------|
| Corporate/marketing | Thin content, slogans | Merge short pages; expand bullets into paragraphs |
| Documentation portal | Deep nesting | Flatten to 3 heading levels max |
| Blog/knowledge base | Standalone articles | Group by topic; create coherent module narrative |
| SPA | WebFetch returns minimal content | Try fetching known subpages directly |
| Wiki | Dense cross-linking | Focus on main content area; ignore sidebar links |
| Paywall/auth-gated | Login prompts instead of content | Skip with warning; report inaccessible URLs |

## Image Pipeline Details

The parser (`agents/md_digitalizer/parser.py`) extracts `![alt](url)` references with ~200 chars of surrounding text into `Section.source_images`. The `inject_local_images_node` uses HTTP URLs directly as `{"type": "img", "query": "...", "content": "https://..."}`. Sections without source images get images from internet search (if enabled).

## Key Files

- Parser: `agents/md_digitalizer/parser.py`
- Restructurer: `agents/md_digitalizer/restructurer.py`
- Workflow: `main/workflow_digitalize.py`
- Image injection: `main/nodes/digitalize.py`
