# Chapter Post-Processing

The parser and restructurer handle OCR cleanup, heading prefix stripping, malformed heading merging, and image extraction automatically. This file covers only the fixes that require manual intervention.

## 1. Enforce structure constraints

Count modules (files) and submodules (`##` per file). Both must be in the **3-15** range:

- **Too few modules (<3)**: Split large files by topic into separate `.md` files.
- **Too many modules (>15)**: Merge closely related modules into single files.
- **Too few submodules (<3 per module)**: Break large sections into logical subgroups.
- **Too many submodules (>15 per module)**: Merge related submodules or split the module into two files.

## 2. Fix heading hierarchy

Heading levels must follow `#` > `##` > `###` with no skips. If a `#` jumps directly to `###`, insert an appropriate `##` in between or promote the `###` to `##`.

## 3. Deduplicate headings

A `##` and its first `###` child must not share the same or near-identical title. When they do, merge them into a single `##` with body text directly underneath.

## 4. Expand bullet-heavy content

The course renderer works best with flowing prose. Convert short bullet lists into descriptive paragraphs. Each `###` section must have at least 150 words of theory text.

## 5. Merge short sections

Sections with fewer than ~100 words should be merged with the previous or next section on the same topic. If a wrapper section has only a brief intro paragraph, move its text into the first child section and remove the heading.

## 6. Remove remaining boilerplate

Strip any content the parser missed: navigation text, cookie banners, footer links, table-of-contents lists, or repeated headers/footers that appear across multiple files.

## 7. Validate image references

Check that `![alt](path)` references point to existing files. Remove broken references. Ensure paths are relative to the chapter file location.

## Pre-workflow checklist

- [ ] 3-15 `.md` files (modules), each starting with a single `#`
- [ ] 3-15 `##` submodules per module
- [ ] No `###` section under 100 words
- [ ] No level skips in heading hierarchy
- [ ] No duplicate `##`/`###` titles within the same module
- [ ] No boilerplate or nav text remaining
- [ ] Image paths resolve to existing files
