"""
Factory for PDF book templates.

Allows selecting different LaTeX templates for book generation.
"""

from pathlib import Path
from typing import Callable, Dict

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Registry of available templates
TEMPLATES: Dict[str, str] = {
    "academic": "academic.tex",
    # Future templates:
    # "modern": "modern.tex",
    # "classic": "classic.tex",
}


def get_template_path(template_name: str) -> Path:
    """Get the path to a LaTeX template file.
    
    Args:
        template_name: Name of the template (e.g., "academic")
        
    Returns:
        Path to the template file
        
    Raises:
        ValueError: If template is not found
    """
    template_name = template_name.lower()
    if template_name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    return TEMPLATES_DIR / TEMPLATES[template_name]


def available_templates() -> list[str]:
    """Return list of available template names."""
    return list(TEMPLATES.keys())

