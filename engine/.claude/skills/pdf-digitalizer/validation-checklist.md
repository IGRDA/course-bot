# PDF Digitalizer Reference

## Validation Checklist

Before running `workflow_digitalize.py`, verify the chapters produced by `pdf2md`:

- [ ] Each `.md` file starts with a single `#` heading
- [ ] Module titles are descriptive (not "Page 1", "Chapter 1", or raw filenames)
- [ ] No numbered/lettered prefixes on any heading
- [ ] No `##` has a `###` child with the same title
- [ ] Heading hierarchy: `#` > `##` > `###` (no skips)
- [ ] No section has fewer than 150 words of theory
- [ ] No PDF artifacts remain (headers, footers, page numbers, watermarks)
- [ ] Content is factually faithful to the source PDF
- [ ] 3-15 module files with substantive content
- [ ] 3-15 submodules per module
- [ ] Extracted images referenced via local paths (`![alt](artifacts/img_xxx.jpg)`) — not remote URLs
- [ ] Image filenames in markdown match actual files on disk (`ls <chapters>/artifacts/` to verify) — never use invented sequential names like `img_001.jpg`
- [ ] Image alt text is descriptive (not empty, not filenames)
- [ ] Images positioned near illustrating text, max 5 per section

## PDF-Specific Issues

| Issue | Mitigation |
|-------|------------|
| Merged/missing headings from PDF layout | Manually split or add `##`/`###` headings based on content topics |
| Tables converted to garbled text | Re-format as markdown tables or convert to prose |
| Multi-column layout scrambled | Reorder paragraphs into logical reading flow |
| Figures extracted without context | Match figures to nearby text and add descriptive alt text |
| Repeated headers/footers in body text | Strip repeated lines that appear across pages |
| OCR artifacts (if scanned PDF) | Clean up misspellings and broken words |

## Image Pipeline Details

`pdf2md` extracts images from the PDF into an `artifacts/` directory alongside the chapter markdown files. The markdown chapters reference these local files via relative paths (e.g. `![alt](artifacts/img_page3.jpg)`).

The parser (`agents/md_digitalizer/parser.py`) extracts `![alt](path)` references with ~200 chars of surrounding text into `Section.source_images`, resolving relative paths against the markdown file's directory. The `inject_local_images_node` copies local image files to the output `images/` directory and sets `{"type": "img", "query": "...", "content": "/path/to/output/images/file.jpg"}` on ParagraphBlocks. The PDF generator finds these local files and includes them via `\includegraphics`.

Sections without source images get stock images from internet search (if enabled via `--images` / no `--no-images` flag). When both are present, extracted PDF images take priority over stock images for blocks where there is a text match.

## Key Files

- PDF converter: `tools/pdf2md/docling/converter.py`
- Parser: `agents/md_digitalizer/parser.py`
- Restructurer: `agents/md_digitalizer/restructurer.py`
- Workflow: `workflows/workflow_digitalize.py`
- Image injection: `workflows/nodes/digitalize.py`
- PDF generator: `tools/json2book/generator.py`
