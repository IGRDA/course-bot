"""
Utility functions for PDF book generation.

Includes LaTeX escaping, image downloading, and BibTeX generation.
"""

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests


# ============================================
# LATEX ESCAPING
# ============================================

# LaTeX special characters that need escaping (outside math mode)
LATEX_SPECIAL_CHARS = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}

# Unicode characters to LaTeX replacements (for symbols not in standard fonts)
UNICODE_TO_LATEX = {
    # Math symbols
    '×': r'$\times$',
    '÷': r'$\div$',
    '±': r'$\pm$',
    '≈': r'$\approx$',
    '≠': r'$\neq$',
    '≤': r'$\leq$',
    '≥': r'$\geq$',
    '∞': r'$\infty$',
    '∑': r'$\sum$',
    '∏': r'$\prod$',
    '√': r'$\sqrt{}$',
    '∈': r'$\in$',
    '∉': r'$\notin$',
    '⊂': r'$\subset$',
    '⊃': r'$\supset$',
    '∪': r'$\cup$',
    '∩': r'$\cap$',
    '→': r'$\rightarrow$',
    '←': r'$\leftarrow$',
    '↔': r'$\leftrightarrow$',
    '⇒': r'$\Rightarrow$',
    '⇐': r'$\Leftarrow$',
    '⇔': r'$\Leftrightarrow$',
    # Greek letters (lowercase)
    'α': r'$\alpha$',
    'β': r'$\beta$',
    'γ': r'$\gamma$',
    'δ': r'$\delta$',
    'ε': r'$\varepsilon$',
    'ζ': r'$\zeta$',
    'η': r'$\eta$',
    'θ': r'$\theta$',
    'ι': r'$\iota$',
    'κ': r'$\kappa$',
    'λ': r'$\lambda$',
    'μ': r'$\mu$',
    'ν': r'$\nu$',
    'ξ': r'$\xi$',
    'π': r'$\pi$',
    'ρ': r'$\rho$',
    'σ': r'$\sigma$',
    'τ': r'$\tau$',
    'υ': r'$\upsilon$',
    'φ': r'$\varphi$',
    'χ': r'$\chi$',
    'ψ': r'$\psi$',
    'ω': r'$\omega$',
    # Greek letters (uppercase)
    'Γ': r'$\Gamma$',
    'Δ': r'$\Delta$',
    'Θ': r'$\Theta$',
    'Λ': r'$\Lambda$',
    'Ξ': r'$\Xi$',
    'Π': r'$\Pi$',
    'Σ': r'$\Sigma$',
    'Φ': r'$\Phi$',
    'Ψ': r'$\Psi$',
    'Ω': r'$\Omega$',
    # Subscript numbers (convert to regular with subscript notation)
    '₀': r'$_0$',
    '₁': r'$_1$',
    '₂': r'$_2$',
    '₃': r'$_3$',
    '₄': r'$_4$',
    '₅': r'$_5$',
    '₆': r'$_6$',
    '₇': r'$_7$',
    '₈': r'$_8$',
    '₉': r'$_9$',
    '₊': r'$_+$',
    '₋': r'$_-$',
    '₌': r'$_=$',
    '₍': r'$_($',
    '₎': r'$_)$',
    'ₐ': r'$_a$',
    'ₑ': r'$_e$',
    'ₒ': r'$_o$',
    'ₓ': r'$_x$',
    'ₜ': r'$_t$',
    'ₙ': r'$_n$',
    'ₘ': r'$_m$',
    'ₖ': r'$_k$',
    'ₛ': r'$_s$',
    'ₚ': r'$_p$',
    # Superscript numbers and letters
    '⁰': r'$^0$',
    '¹': r'$^1$',
    '²': r'$^2$',
    '³': r'$^3$',
    '⁴': r'$^4$',
    '⁵': r'$^5$',
    '⁶': r'$^6$',
    '⁷': r'$^7$',
    '⁸': r'$^8$',
    '⁹': r'$^9$',
    'ⁿ': r'$^n$',
    'ⁱ': r'$^i$',
    'ᵃ': r'$^a$',
    'ᵇ': r'$^b$',
    'ᶜ': r'$^c$',
    'ᵈ': r'$^d$',
    'ᵉ': r'$^e$',
    'ᶠ': r'$^f$',
    'ᵍ': r'$^g$',
    'ʰ': r'$^h$',
    'ʲ': r'$^j$',
    'ᵏ': r'$^k$',
    'ˡ': r'$^l$',
    'ᵐ': r'$^m$',
    'ᵒ': r'$^o$',
    'ᵖ': r'$^p$',
    'ʳ': r'$^r$',
    'ˢ': r'$^s$',
    'ᵗ': r'$^t$',
    'ᵘ': r'$^u$',
    'ᵛ': r'$^v$',
    'ʷ': r'$^w$',
    'ˣ': r'$^x$',
    'ʸ': r'$^y$',
    'ᶻ': r'$^z$',
    # Blackboard bold
    'ℝ': r'$\mathbb{R}$',
    'ℂ': r'$\mathbb{C}$',
    'ℕ': r'$\mathbb{N}$',
    'ℤ': r'$\mathbb{Z}$',
    'ℚ': r'$\mathbb{Q}$',
    # Other common symbols
    '°': r'$^\circ$',
    '′': r"$'$",
    '″': r"$''$",
    '…': r'\ldots{}',
    '—': r'---',
    '–': r'--',
    '"': "``",
    '"': "''",
    ''': "`",
    ''': "'",
}


def escape_latex(text: str, preserve_math: bool = True) -> str:
    """Escape special LaTeX characters while preserving math environments.
    
    Args:
        text: Raw text that may contain LaTeX math
        preserve_math: If True, preserve \\(...\\) and \\[...\\] math environments
        
    Returns:
        Text with special characters escaped, math environments preserved
    """
    if not text:
        return ""
    
    if preserve_math:
        # Split text by math environments and only escape non-math parts
        # Pattern matches \(...\) and \[...\] including their content
        math_pattern = r'(\\\(.*?\\\)|\\\[.*?\\\])'
        parts = re.split(math_pattern, text, flags=re.DOTALL)
        
        result = []
        for part in parts:
            if part.startswith(r'\(') or part.startswith(r'\['):
                # This is a math environment - preserve mostly as-is
                # BUT we must escape % inside \text{} commands because % is a comment
                part = _fix_percent_in_math_text(part)
                result.append(part)
            else:
                # This is regular text - escape special chars
                result.append(_escape_special_chars(part))
        
        return ''.join(result)
    else:
        return _escape_special_chars(text)


def _fix_percent_in_math_text(math_content: str) -> str:
    """Fix unescaped % inside \\text{} commands in math environments.
    
    The % character is a comment in LaTeX, even inside math mode's \\text{}.
    This causes parsing errors when % appears in \\text{...}.
    """
    def escape_percent_in_text(match):
        text_content = match.group(1)
        # Escape any unescaped % (not already \%)
        text_content = re.sub(r'(?<!\\)%', r'\\%', text_content)
        return r'\text{' + text_content + '}'
    
    # Fix \text{...} with unescaped %
    result = re.sub(r'\\text\{([^}]*)\}', escape_percent_in_text, math_content)
    return result


def _escape_special_chars(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    # First handle backslash (must be done first)
    # But preserve \\ for line breaks
    text = re.sub(r'\\(?![\\()\[\]])', r'\\textbackslash{}', text)
    
    # Escape other special characters
    for char, replacement in LATEX_SPECIAL_CHARS.items():
        text = text.replace(char, replacement)
    
    # Convert Unicode characters to LaTeX commands
    for char, replacement in UNICODE_TO_LATEX.items():
        text = text.replace(char, replacement)
    
    # Break up >> and << which are active chars in babel-spanish (quoting shorthands)
    text = text.replace('>>', '>{}>') 
    text = text.replace('<<', '<{}<')
    
    return text


def escape_latex_simple(text: str) -> str:
    """Simple escape for titles and metadata (no math preservation)."""
    if not text:
        return ""
    
    result = text
    # Handle backslash first
    result = result.replace('\\', r'\textbackslash{}')
    
    for char, replacement in LATEX_SPECIAL_CHARS.items():
        result = result.replace(char, replacement)
    
    return result


# ============================================
# MARKDOWN TO LATEX CONVERSION
# ============================================

def markdown_to_latex(text: str) -> str:
    """Convert basic Markdown and HTML formatting to LaTeX.
    
    Handles:
    - ```code blocks``` -> verbatim environment
    - **bold** -> \textbf{bold}
    - *italic* -> \textit{italic}
    - <b>, <i>, <u> HTML tags
    - Tables (basic pipe tables)
    - Lists (- items)
    
    Args:
        text: Markdown-formatted text (already LaTeX-escaped for special chars)
        
    Returns:
        LaTeX-formatted text
    """
    if not text:
        return ""
    
    # First convert HTML tags to LaTeX
    result = _convert_html_tags(text)
    
    # Convert code blocks (before other processing)
    result = _convert_code_blocks(result)
    
    # Protect math environments from markdown conversion
    # Replace math with placeholders, process markdown, then restore
    math_pattern = r'(\\\(.*?\\\)|\\\[.*?\\\])'
    math_parts = re.findall(math_pattern, result, flags=re.DOTALL)
    
    # Replace math with unique placeholders
    placeholder_map = {}
    for i, math in enumerate(math_parts):
        placeholder = f"__MATH_PLACEHOLDER_{i}__"
        placeholder_map[placeholder] = math
        result = result.replace(math, placeholder, 1)
    
    # Convert **bold** to \textbf{} (must be done before italic)
    result = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', result)
    
    # Convert *italic* to \textit{} (single asterisks, but not if preceded/followed by word char)
    # This avoids converting standalone asterisks like "regret*"
    result = re.sub(r'(?<![*\w])\*([^*\n]+?)\*(?![*\w])', r'\\textit{\1}', result)
    
    # Convert Markdown tables to LaTeX tables
    result = _convert_markdown_tables(result)
    
    # Convert bullet lists
    result = _convert_bullet_lists(result)
    
    # Convert numbered lists  
    result = _convert_numbered_lists(result)
    
    # Clean up any itemize/enumerate environments that ended up inside tables
    result = _cleanup_environments_in_tables(result)
    
    # Restore math environments
    for placeholder, math in placeholder_map.items():
        result = result.replace(placeholder, math)
    
    return result


def _cleanup_environments_in_tables(text: str) -> str:
    """Remove itemize/enumerate environments from inside tabularx tables.
    
    This is a safety net for cases where list environments end up inside
    table cells, which breaks LaTeX compilation.
    """
    # Find all tabularx environments and clean them
    def clean_table(match):
        table_content = match.group(0)
        # Remove itemize environments
        table_content = re.sub(r'\\begin\{itemize\}', '', table_content)
        table_content = re.sub(r'\\end\{itemize\}', '', table_content)
        # Remove enumerate environments
        table_content = re.sub(r'\\begin\{enumerate\}', '', table_content)
        table_content = re.sub(r'\\end\{enumerate\}', '', table_content)
        # Convert \item to bullet character
        table_content = re.sub(r'\\item\s*', '• ', table_content)
        return table_content
    
    # Match tabularx environments (including nested content)
    pattern = r'\\begin\{tabularx\}.*?\\end\{tabularx\}'
    result = re.sub(pattern, clean_table, text, flags=re.DOTALL)
    
    # Also clean up any broken math environments containing itemize
    # This handles cases like \frac{1}{... \begin{itemize} ...}
    def clean_math(match):
        math_content = match.group(0)
        # Remove any list environments inside math
        math_content = re.sub(r'\\begin\{itemize\}.*?\\end\{itemize\}', '', math_content, flags=re.DOTALL)
        math_content = re.sub(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', '', math_content, flags=re.DOTALL)
        math_content = re.sub(r'\\begin\{itemize\}', '', math_content)
        math_content = re.sub(r'\\end\{itemize\}', '', math_content)
        math_content = re.sub(r'\\item\s*', '', math_content)
        return math_content
    
    # Clean up \frac{}{} and other math commands
    result = re.sub(r'\\frac\{[^}]*\}\{[^}]*\}', clean_math, result, flags=re.DOTALL)
    
    # Clean up display math environments
    result = re.sub(r'\\\[.*?\\\]', clean_math, result, flags=re.DOTALL)
    
    # Fix % inside \text{} which breaks math (% is a comment in LaTeX)
    # Convert \text{...%...} to \text{...\%...}
    def fix_percent_in_text(match):
        content = match.group(1)
        # Escape any unescaped % inside \text{}
        content = re.sub(r'(?<!\\)%', r'\\%', content)
        return r'\text{' + content + '}'
    
    result = re.sub(r'\\text\{([^}]*)\}', fix_percent_in_text, result)
    
    # Final cleanup: remove any stray unmatched \begin{itemize} or \end{itemize}
    # that might cause issues (orphaned from their pair)
    lines = result.split('\n')
    cleaned_lines = []
    itemize_depth = 0
    enumerate_depth = 0
    
    for line in lines:
        # Count opens and closes
        itemize_opens = len(re.findall(r'\\begin\{itemize\}', line))
        itemize_closes = len(re.findall(r'\\end\{itemize\}', line))
        enumerate_opens = len(re.findall(r'\\begin\{enumerate\}', line))
        enumerate_closes = len(re.findall(r'\\end\{enumerate\}', line))
        
        itemize_depth += itemize_opens - itemize_closes
        enumerate_depth += enumerate_opens - enumerate_closes
        
        cleaned_lines.append(line)
    
    # If we end with unbalanced environments, try to fix
    result = '\n'.join(cleaned_lines)
    
    # Add missing \end{itemize} if needed
    while itemize_depth > 0:
        result += '\n\\end{itemize}'
        itemize_depth -= 1
    
    while enumerate_depth > 0:
        result += '\n\\end{enumerate}'
        enumerate_depth -= 1
    
    return result


def _convert_html_tags(text: str) -> str:
    """Convert HTML formatting tags to LaTeX.
    
    Handles: <b>, <strong>, <i>, <em>, <u>, <code>, <br>
    """
    result = text
    
    # Bold tags
    result = re.sub(r'<b>(.*?)</b>', r'\\textbf{\1}', result, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(r'<strong>(.*?)</strong>', r'\\textbf{\1}', result, flags=re.DOTALL | re.IGNORECASE)
    
    # Italic tags
    result = re.sub(r'<i>(.*?)</i>', r'\\textit{\1}', result, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(r'<em>(.*?)</em>', r'\\textit{\1}', result, flags=re.DOTALL | re.IGNORECASE)
    
    # Underline tags
    result = re.sub(r'<u>(.*?)</u>', r'\\underline{\1}', result, flags=re.DOTALL | re.IGNORECASE)
    
    # Code tags
    result = re.sub(r'<code>(.*?)</code>', r'\\texttt{\1}', result, flags=re.DOTALL | re.IGNORECASE)
    
    # Line breaks
    result = re.sub(r'<br\s*/?>', r'\\\\', result, flags=re.IGNORECASE)
    
    # Remove other HTML tags (clean up)
    result = re.sub(r'</?[a-z][a-z0-9]*[^>]*>', '', result, flags=re.IGNORECASE)
    
    return result


def _convert_code_blocks(text: str) -> str:
    """Convert Markdown code blocks to LaTeX verbatim environment.
    
    Handles both fenced code blocks (```lang...```) and inline code (`code`).
    """
    # Convert fenced code blocks: ```language\ncode\n``` -> verbatim
    def replace_code_block(match):
        code = match.group(2)
        # Unescape special chars that were escaped for LaTeX (inside verbatim they're literal)
        code = _unescape_for_verbatim(code)
        return f'\n\\begin{{verbatim}}\n{code.strip()}\n\\end{{verbatim}}\n'
    
    # Match ```language (optional) followed by code until ```
    result = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
    
    # Convert inline code: `code` -> \texttt{code}
    # Use texttt because \verb cannot be nested inside other commands like \textbf
    def replace_inline_code(match):
        code = match.group(1)
        # Keep LaTeX escaping for texttt (it handles escaped chars fine)
        # Just need to escape any remaining problematic chars
        return f'\\texttt{{{code}}}'
    
    result = re.sub(r'`([^`\n]+)`', replace_inline_code, result)
    
    return result


def _unescape_for_verbatim(text: str) -> str:
    """Unescape LaTeX special characters for use in verbatim environments."""
    text = text.replace(r'\_', '_')
    text = text.replace(r'\#', '#')
    text = text.replace(r'\&', '&')
    text = text.replace(r'\%', '%')
    text = text.replace(r'\$', '$')
    text = text.replace(r'\{', '{')
    text = text.replace(r'\}', '}')
    text = text.replace(r'\textbackslash{}', '\\')
    text = text.replace(r'\textasciitilde{}', '~')
    text = text.replace(r'\textasciicircum{}', '^')
    return text


def _convert_markdown_tables(text: str) -> str:
    """Convert Markdown pipe tables to LaTeX tabular."""
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a table header row
        if '|' in line and i + 1 < len(lines) and re.match(r'\s*\|?[\s\-:|]+\|', lines[i + 1]):
            # This is a table - collect all rows
            table_lines = [line]
            i += 1
            
            # Skip separator line
            if i < len(lines):
                i += 1
            
            # Collect data rows
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            
            # Convert to LaTeX
            latex_table = _markdown_table_to_latex(table_lines)
            result.append(latex_table)
        else:
            result.append(line)
            i += 1
    
    return '\n'.join(result)


def _sanitize_table_cell(cell: str) -> str:
    """Sanitize a table cell to remove problematic LaTeX environments.
    
    Table cells cannot contain itemize, enumerate, or other block environments.
    This function converts them to inline text.
    """
    # Remove itemize environments and convert \item to bullet character
    cell = re.sub(r'\\begin\{itemize\}', '', cell)
    cell = re.sub(r'\\end\{itemize\}', '', cell)
    cell = re.sub(r'\\begin\{enumerate\}', '', cell)
    cell = re.sub(r'\\end\{enumerate\}', '', cell)
    cell = re.sub(r'\\item\s*', '• ', cell)
    
    # Remove any stray begin/end environments that could break tables
    cell = re.sub(r'\\begin\{[^}]+\}', '', cell)
    cell = re.sub(r'\\end\{[^}]+\}', '', cell)
    
    # Clean up multiple spaces/newlines
    cell = re.sub(r'\s+', ' ', cell)
    
    return cell.strip()


def _markdown_table_to_latex(lines: list[str]) -> str:
    """Convert a Markdown table (list of lines) to LaTeX tabularx with auto-width."""
    if not lines:
        return ""
    
    # Parse header
    header_cells = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
    num_cols = len(header_cells)
    
    if num_cols == 0:
        return '\n'.join(lines)  # Not a valid table
    
    # Sanitize header cells
    header_cells = [_sanitize_table_cell(cell) for cell in header_cells]
    
    # Build LaTeX table using tabularx for automatic width adjustment
    # Use X columns which automatically adjust to fill the page width
    col_spec = '|' + 'X|' * num_cols
    
    latex = [
        '',
        r'\begin{table}[H]',
        r'\centering',
        r'\small',  # Slightly smaller font for tables
        r'\begin{tabularx}{\textwidth}{' + col_spec + '}',
        r'\hline',
        r'\textbf{' + r'} & \textbf{'.join(header_cells) + r'} \\',
        r'\hline',
    ]
    
    # Add data rows (skip first line which is header)
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if cells:
            # Sanitize each cell to remove problematic environments
            cells = [_sanitize_table_cell(cell) for cell in cells]
            # Pad with empty cells if needed
            while len(cells) < num_cols:
                cells.append('')
            latex.append(' & '.join(cells[:num_cols]) + r' \\')
            latex.append(r'\hline')
    
    latex.extend([
        r'\end{tabularx}',
        r'\end{table}',
        '',
    ])
    
    return '\n'.join(latex)


def _convert_bullet_lists(text: str) -> str:
    """Convert Markdown bullet lists to LaTeX itemize."""
    lines = text.split('\n')
    result = []
    in_list = False
    
    for line in lines:
        # Check for bullet item (- or *)
        match = re.match(r'^(\s*)[-*]\s+(.+)$', line)
        
        if match:
            if not in_list:
                result.append(r'\begin{itemize}')
                in_list = True
            result.append(r'    \item ' + match.group(2))
        else:
            if in_list and line.strip():  # Non-empty line that's not a list item
                result.append(r'\end{itemize}')
                in_list = False
            result.append(line)
    
    if in_list:
        result.append(r'\end{itemize}')
    
    return '\n'.join(result)


def _convert_numbered_lists(text: str) -> str:
    """Convert Markdown numbered lists to LaTeX enumerate."""
    lines = text.split('\n')
    result = []
    in_list = False
    
    for line in lines:
        # Check for numbered item (1. or 1))
        match = re.match(r'^(\s*)\d+[.)]\s+(.+)$', line)
        
        if match:
            if not in_list:
                result.append(r'\begin{enumerate}')
                in_list = True
            result.append(r'    \item ' + match.group(2))
        else:
            if in_list and line.strip():  # Non-empty line that's not a list item
                result.append(r'\end{enumerate}')
                in_list = False
            result.append(line)
    
    if in_list:
        result.append(r'\end{enumerate}')
    
    return '\n'.join(result)


# ============================================
# IMAGE HANDLING
# ============================================

def download_image(url: str, output_dir: Path, timeout: int = 30) -> Optional[Path]:
    """Download an image from URL to local directory.
    
    Args:
        url: Image URL
        output_dir: Directory to save the image
        timeout: Request timeout in seconds
        
    Returns:
        Path to downloaded image, or None if download failed
    """
    if not url or not url.startswith(('http://', 'https://')):
        return None
    
    try:
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        parsed = urlparse(url)
        path_ext = Path(parsed.path).suffix.lower()
        
        # Default to .jpg if no extension
        if path_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf']:
            path_ext = '.jpg'
        
        filename = f"img_{url_hash}{path_ext}"
        output_path = output_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            return output_path
        
        # Download
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; CourseGenerator/1.0)'
        })
        response.raise_for_status()
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
        
    except Exception as e:
        print(f"Failed to download image {url}: {e}")
        return None


def extract_image_urls(course_data: dict) -> list[str]:
    """Extract all image URLs from course data.
    
    Args:
        course_data: Parsed course.json data
        
    Returns:
        List of unique image URLs
    """
    urls = set()
    
    for module in course_data.get('modules', []):
        for submodule in module.get('submodules', []):
            for section in submodule.get('sections', []):
                # Check HTML elements for images
                html_elements = section.get('html', []) or []
                _extract_urls_from_elements(html_elements, urls)
    
    return list(urls)


def _extract_urls_from_elements(elements: list, urls: set):
    """Recursively extract image URLs from HTML elements."""
    for element in elements:
        if isinstance(element, dict):
            # Check for image field
            image = element.get('image')
            if image and isinstance(image, dict):
                url = image.get('content')
                if url and isinstance(url, str):
                    urls.add(url)
            
            # Check content for nested elements
            content = element.get('content')
            if isinstance(content, list):
                _extract_urls_from_elements(content, urls)
            
            # Check for elements field (in blocks)
            block_elements = element.get('elements')
            if isinstance(block_elements, list):
                _extract_urls_from_elements(block_elements, urls)


# ============================================
# BIBTEX GENERATION
# ============================================

def generate_bibtex(books: list[dict]) -> str:
    """Generate BibTeX entries from bibliography data.
    
    Args:
        books: List of book dictionaries from bibliography.json
        
    Returns:
        BibTeX-formatted string
    """
    entries = []
    seen_keys = set()
    
    for book in books:
        key = _generate_cite_key(book, seen_keys)
        entry = _book_to_bibtex(book, key)
        if entry:
            entries.append(entry)
            seen_keys.add(key)
    
    return '\n\n'.join(entries)


def _generate_cite_key(book: dict, seen_keys: set) -> str:
    """Generate a unique BibTeX citation key.
    
    Format: AuthorYear (e.g., Sutton2018)
    """
    authors = book.get('authors', [])
    year = book.get('year', 'Unknown')
    
    if authors:
        # Get first author's last name
        first_author = authors[0]
        # Handle "Last, First" or "First Last" format
        if ',' in first_author:
            last_name = first_author.split(',')[0].strip()
        else:
            parts = first_author.split()
            last_name = parts[-1] if parts else 'Unknown'
    else:
        last_name = 'Unknown'
    
    # Clean the last name for use as key
    last_name = _normalize_for_key(last_name)
    base_key = f"{last_name}{year}"
    
    # Ensure uniqueness
    key = base_key
    counter = 1
    while key in seen_keys:
        key = f"{base_key}{chr(ord('a') + counter - 1)}"
        counter += 1
    
    return key


def _normalize_for_key(text: str) -> str:
    """Normalize text for use in BibTeX key (ASCII only, no spaces)."""
    # Remove accents
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
    # Remove non-alphanumeric
    return re.sub(r'[^a-zA-Z0-9]', '', ascii_text)


def _book_to_bibtex(book: dict, key: str) -> str:
    """Convert a single book dictionary to BibTeX entry."""
    title = book.get('title', 'Unknown Title')
    authors = book.get('authors', [])
    year = book.get('year', '')
    publisher = book.get('publisher', '')
    isbn = book.get('isbn') or book.get('isbn_13', '')
    doi = book.get('doi', '')
    url = book.get('url', '')
    edition = book.get('edition', '')
    
    # Format authors for BibTeX (Last, First and Last, First)
    author_str = ' and '.join(authors) if authors else 'Unknown Author'
    
    # Escape special characters in fields
    title = _escape_bibtex_field(title)
    author_str = _escape_bibtex_field(author_str)
    publisher = _escape_bibtex_field(publisher)
    
    lines = [f'@book{{{key},']
    lines.append(f'    author = {{{author_str}}},')
    lines.append(f'    title = {{{title}}},')
    
    if year:
        lines.append(f'    year = {{{year}}},')
    if publisher:
        lines.append(f'    publisher = {{{publisher}}},')
    if edition:
        lines.append(f'    edition = {{{edition}}},')
    if isbn:
        lines.append(f'    isbn = {{{isbn}}},')
    if doi:
        lines.append(f'    doi = {{{doi}}},')
    if url:
        lines.append(f'    url = {{{url}}},')
    
    lines.append('}')
    
    return '\n'.join(lines)


def _escape_bibtex_field(text: str) -> str:
    """Escape special characters in BibTeX field values.
    
    Note: Biber handles most special characters internally,
    so we only need to escape braces to maintain BibTeX structure.
    The & character is properly handled by biber/biblatex.
    """
    if not text:
        return ""
    # Only escape braces to maintain BibTeX structure
    # Biber handles other special characters internally
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    return text


# ============================================
# LANGUAGE MAPPING
# ============================================

LANGUAGE_MAP = {
    'español': 'spanish',
    'spanish': 'spanish',
    'english': 'english',
    'inglés': 'english',
    'french': 'french',
    'français': 'french',
    'german': 'german',
    'deutsch': 'german',
    'italian': 'italian',
    'italiano': 'italian',
    'portuguese': 'portuguese',
    'português': 'portuguese',
}


def get_babel_language(course_language: str) -> str:
    """Map course language to babel language name.
    
    Args:
        course_language: Language from course config
        
    Returns:
        Babel language name (defaults to 'english')
    """
    return LANGUAGE_MAP.get(course_language.lower().strip(), 'english')

