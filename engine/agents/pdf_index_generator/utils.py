"""
PDF Index Generator Utilities

Helper functions for computing course layout and section weights.
"""

import math
from typing import List, Dict


def get_module_count(total_pages: int) -> int:
    """Determine number of modules using smooth logarithmic scaling."""
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")

    log_pages = math.log10(total_pages)
    n_modules = 2 + (log_pages * 3)
    return max(1, min(30, round(n_modules)))


def compute_layout(total_pages: int) -> tuple[int, int, int]:
    """
    Compute course layout: (n_modules, n_submodules, n_sections)
    Uses logarithmic scaling so density grows smoothly across course sizes.
    
    Note: For PDF extraction, this is used as a reference for content weights,
    not for determining structure (which comes from the PDF itself).
    """
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")

    log_pages = math.log10(total_pages)
    n_modules = get_module_count(total_pages)

    pages_per_module = total_pages / n_modules
    submodule_scale = math.log10(max(pages_per_module, 1))
    n_submodules = 1 + (submodule_scale * 2)
    n_submodules = max(1, min(8, round(n_submodules)))

    target_pages_per_section = 2 + (log_pages * 0.5)
    target_pages_per_section = max(2, min(6, target_pages_per_section))

    pages_per_submodule = pages_per_module / n_submodules
    raw_sections = pages_per_submodule / target_pages_per_section
    n_sections = max(1, round(raw_sections))

    return n_modules, n_submodules, n_sections


def compute_section_weights(module_durations: List[float]) -> Dict[int, float]:
    """
    Compute proportional weights for each module based on duration.
    
    Used to distribute content weight across modules for theory generation.
    
    Args:
        module_durations: List of module durations in hours
        
    Returns:
        Dictionary mapping module index to weight (0.0-1.0)
    """
    total_duration = sum(module_durations)
    
    if total_duration == 0:
        # Equal weights if no durations specified
        n = len(module_durations)
        return {i: 1.0 / n for i in range(n)}
    
    return {
        i: duration / total_duration 
        for i, duration in enumerate(module_durations)
    }


def compute_pages_per_module(
    total_pages: int,
    module_durations: List[float]
) -> Dict[int, int]:
    """
    Distribute total pages across modules proportionally by duration.
    
    Args:
        total_pages: Total pages for the course
        module_durations: List of module durations in hours
        
    Returns:
        Dictionary mapping module index to allocated pages
    """
    weights = compute_section_weights(module_durations)
    
    # Allocate pages proportionally
    pages_per_module = {
        i: max(1, round(total_pages * weight))
        for i, weight in weights.items()
    }
    
    # Adjust to ensure total matches (rounding errors)
    allocated = sum(pages_per_module.values())
    diff = total_pages - allocated
    
    if diff != 0:
        # Add/remove from the largest module
        largest_module = max(pages_per_module, key=pages_per_module.get)
        pages_per_module[largest_module] += diff
    
    return pages_per_module


def compute_words_per_section(
    total_pages: int,
    words_per_page: int,
    module_durations: List[float],
    sections_per_module: List[int]
) -> Dict[int, int]:
    """
    Compute target words per section for each module based on duration weights.
    
    Modules with more hours get more words per section.
    
    Args:
        total_pages: Total pages for the course
        words_per_page: Target words per page
        module_durations: List of module durations in hours
        sections_per_module: List of section counts per module
        
    Returns:
        Dictionary mapping module index to words per section
    """
    total_words = total_pages * words_per_page
    pages_allocation = compute_pages_per_module(total_pages, module_durations)
    
    words_per_section = {}
    for i, pages in pages_allocation.items():
        module_words = pages * words_per_page
        n_sections = sections_per_module[i] if i < len(sections_per_module) else 1
        words_per_section[i] = max(100, module_words // max(1, n_sections))
    
    return words_per_section
