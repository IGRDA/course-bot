"""
PDF Book Generator.

Converts course.json to professional academic PDF books via LaTeX.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .factory import get_template_path
from .utils import (
    escape_latex,
    escape_latex_simple,
    markdown_to_latex,
    download_image,
    extract_image_urls,
    generate_bibtex,
    get_babel_language,
)


def generate_pdf_book(
    course_json_path: str | Path,
    output_dir: str | Path | None = None,
    template: str = "academic",
    download_images: bool = True,
    cleanup: bool = True,
) -> Path:
    """Generate a PDF book from course.json.
    
    Args:
        course_json_path: Path to course.json file
        output_dir: Directory for output files (defaults to same as course.json)
        template: Template name (default: "academic")
        download_images: Whether to download images from URLs
        cleanup: Whether to remove intermediate LaTeX files after compilation
        
    Returns:
        Path to generated PDF file
        
    Raises:
        FileNotFoundError: If course.json or template not found
        RuntimeError: If PDF compilation fails
    """
    course_path = Path(course_json_path)
    if not course_path.exists():
        raise FileNotFoundError(f"Course file not found: {course_path}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = course_path.parent / "book"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load course data
    with open(course_path, 'r', encoding='utf-8') as f:
        course_data = json.load(f)
    
    # Load bibliography if available
    bibliography_path = course_path.parent / "bibliography.json"
    bibliography_data = None
    if bibliography_path.exists():
        with open(bibliography_path, 'r', encoding='utf-8') as f:
            bibliography_data = json.load(f)
    
    print(f"📚 Generating PDF book from: {course_path}")
    print(f"   Output directory: {output_dir}")
    
    # Download images if requested
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    if download_images:
        print("📷 Downloading images...")
        image_urls = extract_image_urls(course_data)
        downloaded = 0
        for url in image_urls:
            result = download_image(url, images_dir)
            if result:
                downloaded += 1
        print(f"   Downloaded {downloaded}/{len(image_urls)} images")
    
    # Generate BibTeX file
    bibtex_path = output_dir / "references.bib"
    if bibliography_data:
        all_books = bibliography_data.get('all_books', [])
        if all_books:
            print(f"📖 Generating bibliography ({len(all_books)} references)...")
            bibtex_content = generate_bibtex(all_books)
            with open(bibtex_path, 'w', encoding='utf-8') as f:
                f.write(bibtex_content)
            # Clean up cached biber files to ensure fresh processing
            for ext in ['.bbl', '.bcf', '.run.xml']:
                cached = output_dir / f"book{ext}"
                if cached.exists():
                    cached.unlink()
    
    # Generate LaTeX content
    print("📝 Generating LaTeX document...")
    latex_content = _generate_latex_document(
        course_data,
        template,
        has_bibliography=bibtex_path.exists(),
        images_dir=images_dir,
    )
    
    # Write LaTeX file
    tex_path = output_dir / "book.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    # Compile PDF
    print("🔨 Compiling PDF (this may take a moment)...")
    pdf_path = _compile_pdf(tex_path, bibtex_path.exists())
    
    if pdf_path and pdf_path.exists():
        print(f"✅ PDF generated successfully: {pdf_path}")
        
        # Cleanup intermediate files if requested
        if cleanup:
            _cleanup_latex_files(output_dir)
        
        return pdf_path
    else:
        raise RuntimeError("PDF compilation failed. Check LaTeX output for errors.")


def _generate_latex_document(
    course_data: dict,
    template: str,
    has_bibliography: bool,
    images_dir: Path = None,
) -> str:
    """Generate complete LaTeX document from course data.
    
    Args:
        course_data: Parsed course.json
        template: Template name
        has_bibliography: Whether bibliography file exists
        images_dir: Directory where images are stored
        
    Returns:
        Complete LaTeX document as string
    """
    # Load template
    template_path = get_template_path(template)
    with open(template_path, 'r', encoding='utf-8') as f:
        latex = f.read()
    
    # Extract metadata
    config = course_data.get('config', {})
    title = config.get('title', course_data.get('title', 'Untitled Course'))
    description = config.get('description', '')
    language = config.get('language', 'English')
    
    # Replace metadata placeholders
    latex = latex.replace('COURSE_TITLE', escape_latex_simple(title))
    latex = latex.replace('COURSE_DESCRIPTION', escape_latex_simple(description[:500]))  # Limit description
    babel_lang = get_babel_language(language)
    latex = latex.replace('COURSE_LANGUAGE', babel_lang)
    
    # Build babel language list: always include english + the course language
    babel_parts = ['english']
    if babel_lang != 'english':
        extra = babel_lang
        if babel_lang == 'spanish':
            extra = 'spanish,es-noquoting'
        babel_parts.append(extra)
    latex = latex.replace('BABEL_LANGUAGES', ','.join(babel_parts))
    
    # Handle cover image
    cover_image = _find_cover_image(images_dir)
    if cover_image:
        latex = latex.replace('COVER_IMAGE', cover_image)
    else:
        # Use a TikZ-generated gradient if no cover image available
        # Remove the entire background image node that would fail without an image
        latex = latex.replace(
            r'''        % Background image
        \node[anchor=center, inner sep=0] at (current page.center) {%
            \includegraphics[width=\paperwidth, height=\paperheight, keepaspectratio=false]{COVER_IMAGE}%
        };''',
            r'        % No cover image - using gradient background'
        )
        # Replace the overlay with a solid gradient (no image behind it)
        latex = latex.replace(
            r'\fill[coverblue, opacity=0.88]',
            r'\shade[top color=coverblue, bottom color=adinhubblue]'
        )
    
    # Add bibliography resource if available
    if has_bibliography:
        # Add after biblatex package
        latex = latex.replace(
            r']{biblatex}',
            r']{biblatex}' + '\n' + r'\addbibresource{references.bib}'
        )
    
    # Generate main content
    content = _generate_content(course_data, images_dir)
    
    # Replace content placeholder
    latex = latex.replace('% CONTENT_PLACEHOLDER - Will be replaced by generator', content)
    
    return latex


def _generate_content(course_data: dict, images_dir: Path = None) -> str:
    """Generate LaTeX content for all modules.
    
    Args:
        course_data: Parsed course.json
        images_dir: Directory where images are stored (to check existence)
        
    Returns:
        LaTeX content string for main body
    """
    lines = []
    
    modules = course_data.get('modules', [])
    
    # Get research data for learning objectives (if available)
    research = course_data.get('research') or {}
    learning_objectives = research.get('learning_objectives', [])
    
    for idx, module in enumerate(modules):
        # Chapter (module)
        module_title = module.get('title', 'Untitled Module')
        lines.append(f'\n\\chapter{{{escape_latex_simple(module_title)}}}')
        
        # Module intro box with description
        module_desc = module.get('description', '')
        if module_desc:
            lines.append('')
            lines.append(r'\begin{quote}')
            lines.append(r'\textit{' + escape_latex_simple(module_desc) + '}')
            lines.append(r'\end{quote}')
            lines.append('')
        
        # Add learning objectives box for first chapter (course-level objectives)
        # or extract section titles as module objectives
        if idx == 0 and learning_objectives:
            # Course-level objectives in first chapter
            lines.append('')
            lines.append(r'\begin{objectivesbox}[Course Learning Objectives]')
            lines.append(r'After completing this course, you will be able to:')
            lines.append(r'\begin{itemize}')
            for obj in learning_objectives[:6]:  # Limit to 6 objectives
                clean_obj = escape_latex_simple(obj)
                # Remove markdown bold markers
                clean_obj = clean_obj.replace('**', '')
                lines.append(f'\\item {clean_obj}')
            lines.append(r'\end{itemize}')
            lines.append(r'\end{objectivesbox}')
            lines.append('')
        else:
            # Generate module-specific objectives from submodule titles
            submodules = module.get('submodules', [])
            if submodules:
                lines.append('')
                lines.append(r'\begin{objectivesbox}[Module Objectives]')
                lines.append(r'In this module, you will learn about:')
                lines.append(r'\begin{itemize}')
                for submod in submodules[:5]:  # Limit to 5 items
                    sub_title = submod.get('title', '')
                    if sub_title:
                        lines.append(f'\\item {escape_latex_simple(sub_title)}')
                lines.append(r'\end{itemize}')
                lines.append(r'\end{objectivesbox}')
                lines.append('')
        
        # Submodules become sections
        for submodule in module.get('submodules', []):
            submodule_title = submodule.get('title', 'Untitled Section')
            lines.append(f'\n\\section{{{escape_latex_simple(submodule_title)}}}')
            
            submodule_desc = submodule.get('description', '')
            if submodule_desc and submodule_desc != submodule_title:
                lines.append('')
                lines.append(r'\textit{' + escape_latex_simple(submodule_desc) + '}')
                lines.append('')
            
            # Sections become subsections
            for section in submodule.get('sections', []):
                section_title = section.get('title', 'Untitled Subsection')
                lines.append(f'\n\\subsection{{{escape_latex_simple(section_title)}}}')
                
                # Theory content
                theory = section.get('theory', '')
                if theory:
                    # Process theory: escape LaTeX chars, convert markdown formatting
                    processed_theory = escape_latex(theory, preserve_math=True)
                    processed_theory = markdown_to_latex(processed_theory)
                    
                    # Split into paragraphs
                    paragraphs = processed_theory.split('\n\n')
                    for para in paragraphs:
                        para = para.strip()
                        if para:
                            lines.append('')
                            lines.append(para)
                    lines.append('')
                
                # Process HTML elements for images and other content
                html_elements = section.get('html', []) or []
                _process_html_elements(html_elements, lines, images_dir)

                # Fallback: if no HTML elements, inject source_images directly
                if not html_elements:
                    for img_info in (section.get('source_images') or []):
                        img_path = Path(img_info.get('path', ''))
                        if img_path.exists() and _is_valid_latex_image(img_path):
                            if images_dir:
                                dest = images_dir / img_path.name
                                if not dest.exists():
                                    shutil.copy2(img_path, dest)
                                filename = img_path.name
                            else:
                                filename = str(img_path)
                            alt = img_info.get('alt', '')
                            lines.append('')
                            lines.append(r'\begin{figure}[H]')
                            lines.append(r'\centering')
                            lines.append(
                                f'\\includegraphics[width=0.8\\textwidth,keepaspectratio]{{{filename}}}'
                            )
                            if alt and alt.lower() not in ('image', 'img', 'figure'):
                                lines.append(f'\\caption{{{escape_latex_simple(alt)}}}')
                            lines.append(r'\end{figure}')
                            lines.append('')
    
    return '\n'.join(lines)


def _process_html_elements(elements: list, lines: list, images_dir: Path = None) -> None:
    """Process HTML elements and add corresponding LaTeX content.
    
    Args:
        elements: List of HTML element dictionaries
        lines: List to append LaTeX lines to
        images_dir: Directory where images are downloaded (to check existence)
    """
    import hashlib
    from urllib.parse import urlparse
    
    for element in elements:
        if not isinstance(element, dict):
            continue
        
        element_type = element.get('type', '')
        
        # Handle image elements
        if element_type == 'image' or 'image' in element:
            image_data = element.get('image', {})
            if isinstance(image_data, dict):
                url = image_data.get('content', '')
                caption = image_data.get('caption', '') or element.get('caption', '')
                
                if url and url.startswith(('http://', 'https://')):
                    # Generate filename from URL hash (same as download_image)
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                    parsed = urlparse(url)
                    path_ext = Path(parsed.path).suffix.lower()
                    if path_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.pdf']:
                        path_ext = '.jpg'  # Default extension
                    filename = f"img_{url_hash}{path_ext}"
                    
                    # Only include image if it was actually downloaded and is a valid format
                    if images_dir:
                        image_path = images_dir / filename
                        if image_path.exists() and _is_valid_latex_image(image_path):
                            lines.append('')
                            lines.append(r'\begin{figure}[H]')
                            lines.append(r'\centering')
                            lines.append(f'\\includegraphics[width=0.8\\textwidth,keepaspectratio]{{{filename}}}')
                            if caption:
                                lines.append(f'\\caption{{{escape_latex_simple(caption)}}}')
                            lines.append(r'\end{figure}')
                            lines.append('')
                elif url and not url.startswith(('http://', 'https://')):
                    # Local file path (from content digitalizer)
                    local_path = Path(url)
                    if local_path.exists() and _is_valid_latex_image(local_path):
                        # Copy to images_dir so LaTeX can find it
                        if images_dir:
                            import shutil
                            dest = images_dir / local_path.name
                            if not dest.exists():
                                shutil.copy2(local_path, dest)
                            filename = local_path.name
                        else:
                            filename = str(local_path)
                        lines.append('')
                        lines.append(r'\begin{figure}[H]')
                        lines.append(r'\centering')
                        lines.append(f'\\includegraphics[width=0.8\\textwidth,keepaspectratio]{{{filename}}}')
                        if caption:
                            lines.append(f'\\caption{{{escape_latex_simple(caption)}}}')
                        lines.append(r'\end{figure}')
                        lines.append('')
        
        # Handle paragraphs with text content
        elif element_type == 'p' and 'content' in element:
            content = element.get('content', '')
            if isinstance(content, str) and content.strip():
                processed = escape_latex(content, preserve_math=True)
                processed = markdown_to_latex(processed)
                lines.append('')
                lines.append(processed)
                lines.append('')
        
        # Recursively process nested content
        content = element.get('content')
        if isinstance(content, list):
            _process_html_elements(content, lines, images_dir)
        
        elements_list = element.get('elements')
        if isinstance(elements_list, list):
            _process_html_elements(elements_list, lines, images_dir)


def _is_valid_latex_image(image_path: Path) -> bool:
    """Check if image file is in a format LaTeX can handle.
    
    Checks the actual file content, not just the extension,
    since some files may have wrong extensions (e.g., HTML error pages).
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image is valid for LaTeX (PNG, JPEG, PDF)
    """
    if not image_path.exists():
        return False
    
    try:
        # Read first bytes to detect actual format
        with open(image_path, 'rb') as f:
            header = f.read(32)
        
        # Reject HTML files (common when image download fails)
        if b'<!DOCTYPE' in header or b'<html' in header.lower() or b'<HTML' in header:
            return False
        
        # PNG: starts with 0x89 PNG
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return True
        
        # JPEG: starts with 0xFF 0xD8 0xFF
        if header[:3] == b'\xff\xd8\xff':
            return True
        
        # PDF: starts with %PDF
        if header[:4] == b'%PDF':
            return True
        
        # GIF: starts with GIF87a or GIF89a - NOT well supported by XeLaTeX
        if header[:6] in (b'GIF87a', b'GIF89a'):
            return False
        
        # WebP: starts with RIFF....WEBP (not supported by standard LaTeX)
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return False
        
        # SVG files (XML-based, not directly supported)
        if b'<?xml' in header or b'<svg' in header.lower():
            return False
        
        # Unknown format - reject to be safe
        return False
        
    except Exception:
        return False


def _find_cover_image(images_dir: Path) -> Optional[str]:
    """Find a suitable cover image from the images directory.
    
    Looks for images named 'cover.*' or uses the first large image found.
    
    Args:
        images_dir: Directory containing downloaded images
        
    Returns:
        Filename of cover image, or None if not found
    """
    if not images_dir or not images_dir.exists():
        return None
    
    # First, look for explicitly named cover images
    cover_names = ['cover', 'banner', 'header', 'hero']
    for name in cover_names:
        for ext in ['.jpg', '.jpeg', '.png', '.pdf']:
            cover_path = images_dir / f"{name}{ext}"
            if cover_path.exists() and _is_valid_latex_image(cover_path):
                return f"{name}{ext}"
    
    # Otherwise, find the first valid image (prefer larger files as they're likely better quality)
    valid_images = []
    for img_file in images_dir.iterdir():
        if img_file.is_file() and _is_valid_latex_image(img_file):
            valid_images.append((img_file, img_file.stat().st_size))
    
    if valid_images:
        # Sort by size (largest first) and return the first one
        valid_images.sort(key=lambda x: x[1], reverse=True)
        return valid_images[0][0].name
    
    return None


def _compile_pdf(tex_path: Path, has_bibliography: bool) -> Optional[Path]:
    """Compile LaTeX document to PDF.
    
    Args:
        tex_path: Path to .tex file
        has_bibliography: Whether to run biber for bibliography
        
    Returns:
        Path to generated PDF, or None if compilation failed
    """
    work_dir = tex_path.parent
    tex_name = tex_path.stem
    
    # Check for pdflatex or xelatex
    latex_cmd = _find_latex_command()
    if not latex_cmd:
        print("❌ Error: pdflatex or xelatex not found. Please install TeX Live or MiKTeX.")
        return None
    
    # Common pdflatex options (no output-directory since we cwd to work_dir)
    latex_options = [
        '-interaction=nonstopmode',
        '-halt-on-error',
        '-file-line-error',
    ]
    
    # Use just the filename since we're running from work_dir
    tex_filename = tex_path.name
    
    try:
        # First pass
        print("   Pass 1/3: Initial compilation...")
        result = subprocess.run(
            [latex_cmd] + latex_options + [tex_filename],
            cwd=work_dir,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=300,
        )
        
        if result.returncode != 0:
            print(f"❌ LaTeX compilation error:\n{result.stdout[-2000:]}")
            _save_log(work_dir / f"{tex_name}.log", "latex_error.log")
            return None
        
        # Run biber if needed
        if has_bibliography:
            print("   Pass 2/3: Processing bibliography...")
            biber_result = subprocess.run(
                ['biber', tex_name],
                cwd=work_dir,
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                timeout=60,
            )
            if biber_result.returncode != 0:
                print(f"⚠️  Biber warning (continuing): {biber_result.stderr[:500]}")
        
        # Second pass (for references and TOC)
        print("   Pass 3/4: Second compilation...")
        result = subprocess.run(
            [latex_cmd] + latex_options + [tex_filename],
            cwd=work_dir,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=300,
        )
        if result.returncode != 0:
            print(f"⚠️  Pass 2 warnings (continuing): {result.stdout[-500:]}")
        
        # Third pass (for bookmarks and final references)
        print("   Pass 4/4: Final compilation...")
        result = subprocess.run(
            [latex_cmd] + latex_options + [tex_filename],
            cwd=work_dir,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=300,
        )
        if result.returncode != 0:
            print(f"⚠️  Pass 3 warnings (continuing): {result.stdout[-500:]}")
        
        pdf_path = work_dir / f"{tex_name}.pdf"
        if pdf_path.exists():
            return pdf_path
        else:
            print("❌ PDF file not created")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ LaTeX compilation timed out")
        return None
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        return None


def _find_latex_command() -> Optional[str]:
    """Find available LaTeX command (prefer xelatex for Unicode support).
    
    Checks both PATH and common TeX installation directories.
    Prefers xelatex over pdflatex for better Unicode handling.
    """
    # Common TeX installation paths (macOS, Linux)
    tex_paths = [
        '/Library/TeX/texbin',  # macOS TeX Live / BasicTeX
        '/usr/local/texlive/2025/bin/universal-darwin',  # macOS full TeX Live
        '/usr/local/texlive/2024/bin/universal-darwin',
        '/usr/local/texlive/2025/bin/x86_64-linux',  # Linux TeX Live
        '/usr/local/texlive/2024/bin/x86_64-linux',
        '/usr/bin',  # System default
    ]
    
    # Prefer xelatex for better Unicode support (Greek letters, etc.)
    for cmd in ['xelatex', 'pdflatex']:
        # First try PATH
        if shutil.which(cmd):
            return cmd
        
        # Then check common installation directories
        for tex_path in tex_paths:
            full_path = Path(tex_path) / cmd
            if full_path.exists() and os.access(full_path, os.X_OK):
                return str(full_path)
    
    return None


def _save_log(log_path: Path, dest_name: str):
    """Save LaTeX log file for debugging."""
    if log_path.exists():
        dest = log_path.parent / dest_name
        shutil.copy(log_path, dest)
        print(f"   Log saved to: {dest}")


def _cleanup_latex_files(output_dir: Path):
    """Remove intermediate LaTeX files."""
    extensions = ['.aux', '.log', '.out', '.toc', '.bbl', '.blg', '.bcf', '.run.xml']
    for ext in extensions:
        for file in output_dir.glob(f'*{ext}'):
            try:
                file.unlink()
            except Exception:
                pass

