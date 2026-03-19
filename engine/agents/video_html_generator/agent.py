"""Video HTML Generator Agent.

Deterministic agent that simplifies module JSON files by extracting only:
- title, id, index, description, duration, type at module level
- title, index, description, duration at submodule level  
- title, index, description, html (all elements, but content arrays truncated to first item) at section level

No LLM calls - purely structural transformation.
"""

import json
from pathlib import Path
from typing import Any


def simplify_html_element(element: dict[str, Any]) -> dict[str, Any]:
    """
    Simplify an HTML element by keeping only the first item if content is an array.
    
    Args:
        element: HTML element dictionary
        
    Returns:
        Simplified element with content truncated to first item if it's an array
    """
    simplified = dict(element)  # Copy all fields
    
    content = element.get("content")
    if isinstance(content, list) and len(content) > 0:
        # Keep only first item in content array (e.g., for timeline elements)
        simplified["content"] = [content[0]]
    
    return simplified


def simplify_section(section: dict[str, Any]) -> dict[str, Any]:
    """
    Simplify a section to contain only essential fields for video generation.
    
    Args:
        section: Full section dictionary from module JSON
        
    Returns:
        Simplified section with title, index, description, and all html elements
        (with content arrays truncated to first item)
    """
    simplified = {
        "title": section.get("title", ""),
        "index": section.get("index", 0),
        "description": section.get("description", ""),  # Website expects this field
    }
    
    # Keep all HTML elements, but simplify each one (truncate content arrays to first item)
    html = section.get("html", [])
    if html and len(html) > 0:
        simplified["html"] = [simplify_html_element(elem) for elem in html]
    else:
        simplified["html"] = []  # Empty array instead of None
    
    return simplified


def simplify_submodule(submodule: dict[str, Any]) -> dict[str, Any]:
    """
    Simplify a submodule to contain only essential fields for video generation.
    
    Args:
        submodule: Full submodule dictionary from module JSON
        
    Returns:
        Simplified submodule with title, index, description, duration, and sections
    """
    return {
        "title": submodule.get("title", ""),
        "index": submodule.get("index", 0),
        "description": submodule.get("description", ""),  # Website expects this field
        "duration": submodule.get("duration", 0),  # Website expects this field
        "sections": [
            simplify_section(section)
            for section in submodule.get("sections", [])
        ],
    }


def simplify_module(module_data: dict[str, Any]) -> dict[str, Any]:
    """
    Simplify a module JSON structure for video generation.
    
    Keeps only essential fields:
    - Module level: title, id, index, description, duration, type
    - Submodule level: title, index, description, duration
    - Section level: title, index, description, html[0]
    
    Args:
        module_data: Full module dictionary loaded from JSON
        
    Returns:
        Simplified module dictionary
    """
    return {
        "title": module_data.get("title", ""),
        "id": module_data.get("id", ""),
        "index": module_data.get("index", 0),
        "description": module_data.get("description", ""),  # Website expects this field
        "duration": module_data.get("duration", 0),  # Website expects this field
        "type": module_data.get("type", "module"),  # Website expects this field
        "submodules": [
            simplify_submodule(submodule)
            for submodule in module_data.get("submodules", [])
        ],
    }


def simplify_module_from_path(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Load a module JSON file, simplify it, and save to video_html_generator subfolder.
    
    Args:
        input_path: Path to the input module JSON file
        output_dir: Optional custom output directory. If None, uses 
                   video_html_generator/ subfolder in the same parent directory.
                   
    Returns:
        Path to the saved simplified JSON file
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the input JSON
    with open(input_path, "r", encoding="utf-8") as f:
        module_data = json.load(f)
    
    # Simplify the module
    simplified = simplify_module(module_data)
    
    # Determine output path
    if output_dir is None:
        # Default: video_html_generator/ subfolder in same parent directory
        output_dir = input_path.parent / "video_html_generator"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with same filename
    output_path = output_dir / input_path.name
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)
    
    return output_path

